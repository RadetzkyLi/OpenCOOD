#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   vis_risky_scene.py
@Date    :   2024-01-30
@Author  :   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Visualize Risky Testing Scene
'''

import os
os.environ['NUMEXPR_MAX_THREADS'] = '8' 
import open3d as o3d
import numpy as np
import random
import argparse
import pymeshlab
import carla
import logging
import yaml


# import necessary lib
from opencood.utils.pcd_utils import mask_points_by_range
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils import box_utils


# ===================================================================
# - Utility function
# ===================================================================
COLOR_MAPPING = {
    "black": (0,0,0),
    "white": (1,1,1),
    "gray": (0.96,0.96,0.96),
    "red": (1,0,0),
    "orange": (1,0.5,0),
    "yellow": (1,1,0),
    "green": (0,1,0),
    "cyan": (0,1,1),
    "blue": (0,0,1),
    "puple": (0.5,0,0.5)
}

def carla_locations_to_array(locs, to_homo=False):
    arr = []
    for loc in locs:
        arr.append([loc.x, loc.y, loc.z])
    arr = np.array(arr)
    if to_homo:
        arr_new = np.ones((len(arr),4))
        arr_new[:,:3] = arr
        arr = arr_new
    return arr

def load_yaml(path):
    with open(path, encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  
    return data

def load_anno_info(path):
    """pcd path"""
    if path.endswith(".pcd"):
        anno_path = path.split(".pcd")[0] + ".yaml"
    else:
        anno_path = path
    with open(anno_path,encoding='utf8') as f:
        anno_info = yaml.load(f, Loader=yaml.FullLoader)  # scene info
    return anno_info

def get_cav_rsu_list(scenario_root):
    cav_list,rsu_list = [],[]
    for item in os.listdir(scenario_root):
        if item.startswith("cav"):
            cav_list.append(int(item.split("cav_")[1]))
        if item.startswith("rsu"):
            rsu_list.append(int(item.split("rsu_")[1]))
    return cav_list,rsu_list

def load_point_cloud(path, color=[0,0,1], ego_pose=None):
    """
    Parameters
    ----------
    path: str

    color : list
        Default to [0,0,1] ("blue")

    ego_pose : list
        Ego's [x,y,z,roll,yaw,pitch]

    Returns
    -------
    pcd : o3d.PointCloud
    """
    cav_lidar_range = [-100, -100, -3, 100, 100, 3]
    cav_lidar_range = [-120, -120, -3, 120, 120, 3]
    cav_lidar_range = [-120, -120, -6, 120, 120, 3]  #  used for RSU
    cav_lidar_range = [-100, -100, -6, 100, 100, 3]  #  used for RSU
    # cav_lidar_range = [-70, -70, -6, 70, 70, 6]
    
    # load point cloud
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    
    if ego_pose:
        # get ego's pose
        anno_info = load_anno_info(path)
        # calculate the transformation matrix
        transformation_matrix = x1_to_x2(anno_info['lidar_pose'], ego_pose)
        # project the lidar to ego space
        points[:, :3] = box_utils.project_points_by_matrix_torch(points[:, :3],
                                                                transformation_matrix)
    # crop
    points = mask_points_by_range(points, cav_lidar_range)

    points[:,1] = -points[:,1]  # left-->right
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)

    return pcd

# =========================================================
# - visualization functions
# =========================================================

def create_3d_bbox(vertices, edges=None, color=[0, 1, 0]):
    """
    Args:
        vertices: {ndarrary} of shape (8,3)
        edges: {ndarray} of shape (12,2). By fault, we use order of CARLA.
        color: List[int], color of bounding box. Defaults to green ([0,1,0]).
    Returns:
        bbox: open3d.LineSet, bbox composed of 12 lines
    """
    assert len(vertices)==8, "8 vertices are required, got {0}".format(len(vertices))
    assert edges is None or len(edges)==12, "12 edges are required, got {0}".format(len(edges))

    if edges is None:
        # the order corresponds to CARLA
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        edges = np.array(edges, dtype=np.int64)
    colors = [color for _ in range(len(edges))]
    bbox = o3d.geometry.LineSet()
    bbox.lines  = o3d.utility.Vector2iVector(edges)
    bbox.colors = o3d.utility.Vector3dVector(colors)
    bbox.points = o3d.utility.Vector3dVector(vertices)
    return bbox


def create_3d_bbox_OPV2V(obj_info, mat_w2l, backend="open3d", color=[0,1,0]):
    """Create 3D bounding box for object annotated in OPV2V format.

    Parameters
    ----------
    obj_info: dict
        An object's annotation in OPV2V format.

    mat_w2l: ndarray
        Matrix transform a point from world to lidar, of shape (4,4).
    
    backend: "open3d"|"carla"
        Method to create bbox.

    color: List[int]
        Bounding box's color, defaults to [0, 1, 0] (green).

    Returns
    -------
    bbox: open3d.LineSet|open3d.OrientedBoundingBox
        Bouding box represented by 12 lines.
    """
    center = obj_info['center']
    angle = obj_info['angle']
    extent = obj_info['extent']
    loc = obj_info['location']
    veh_transform = carla.Transform(
        carla.Location(loc[0], loc[1], loc[2]),
        carla.Rotation(roll=angle[0], yaw=angle[1], pitch=angle[2])
    )
    
    if backend == "open3d":
        # matrix from vehicle to lidar
        M_v2l = np.dot(mat_w2l, veh_transform.get_matrix())
        center.append(1)
        center = np.dot(M_v2l, np.array(center).reshape(4,1))
        center = [center[0,0], center[1,0], center[2,0]]
        
        center[1] = -center[1]
        mat_l2r = np.eye(3) # left--> right
        mat_l2r[1,1] = -1
        R = np.dot(mat_l2r, M_v2l[:3,:3])
        bbox = o3d.geometry.OrientedBoundingBox(np.array(center), R, 2.0*np.array(extent))
        bbox.color = np.array(color)
        # indices = bbox.get_point_indices_within_bounding_box(pcd.points)
        # print("points in bbox", len(indices))
    elif backend == "carla":
        delta = 0.01
        bbox = carla.BoundingBox(
                carla.Location(center[0], center[1], center[2]),
                carla.Vector3D(extent[0]+delta, extent[1]+delta, extent[2]+delta)
            )
        vertices = bbox.get_world_vertices(veh_transform)
        vertices = carla_locations_to_array(vertices,to_homo=True)
        
        # from global space to local space
        vertices = np.dot(vertices, mat_w2l.T)
        vertices = vertices[:,:3]
        vertices[:,1] = -vertices[:,1]  # left-->right
        bbox = create_3d_bbox(vertices)
    else:
        raise ValueError("Unsupported backend: {0}".format(backend))
    return bbox


# =====================================================================
# - Visualize point cloud for test scenes
# =====================================================================
def viz_pointcloud_test_scene(pcd_path, save_path=None):
    """
    
    Parameters
    ----------
    pcd_path : str
        The pcd file path or the scene root
    """
    if pcd_path.endswith(".pcd"):
        scenario_root = os.path.dirname(os.path.dirname(pcd_path))
        timestamp = os.path.basename(pcd_path).split(".pcd")[0]
    else:
        # given the scene root, select the first timestamp
        scenario_root = pcd_path
        timestamp = list(os.listdir(os.path.join(scenario_root, "map")))[0].split(".yaml")[0]

    # 1) get ego, occluder, occludee from scene desc
    yaml_path = os.path.join(scenario_root, "data_protocal.yaml")
    scene_desc = load_yaml(yaml_path)["additional"]["scene_desc"]

    ego_id = scene_desc["id_mapping"][scene_desc["ego_id"]]
    occluder_id = scene_desc["occlusion_pairs"][0][0]  # suppose only one occlusion pair
    occluder_id = scene_desc["id_mapping"].get(occluder_id, None)
    occludee_id = scene_desc["occlusion_pairs"][0][1]
    occludee_id = scene_desc["id_mapping"].get(occludee_id, None)

    # get all agents
    cav_id_list,rsu_id_list = get_cav_rsu_list(scenario_root)
    logging.info("Connected CAV: {0}, RSU: {1}".format(len(cav_id_list), len(rsu_id_list)))
    assert ego_id in cav_id_list+rsu_id_list
    # just keep one rsu
    rsu_id_list = [rsu_id_list[0]] if len(rsu_id_list)>1 else rsu_id_list

    # get ego's data
    if ego_id in cav_id_list:
        yaml_path = os.path.join(scenario_root, "cav_"+str(ego_id), timestamp+".yaml")
    else:
        yaml_path = os.path.join(scenario_root, "rsu_"+str(ego_id), timestamp+".yaml")

    # 2) Load data
    # 2.1) load ego's infomation
    anno_info = load_anno_info(yaml_path)
    lidar_pose = anno_info['lidar_pose']
    lidar_transform = carla.Transform(
        carla.Location(lidar_pose[0], lidar_pose[1], lidar_pose[2]),
        carla.Rotation(roll=lidar_pose[3], yaw=lidar_pose[4], pitch=lidar_pose[5])
    )
    mat_w2l = np.array(lidar_transform.get_inverse_matrix())  # from world to lidar

    # 2.2) load all agent's point cloud data
    pcd_list = []
    pointcloud_color = COLOR_MAPPING["white"]
    for agent_id in cav_id_list+rsu_id_list:
        if agent_id in cav_id_list:
            pcd_path = os.path.join(scenario_root, "cav_"+str(agent_id), timestamp+".pcd")
        elif agent_id in rsu_id_list:
            pcd_path = os.path.join(scenario_root, "rsu_"+str(agent_id), timestamp+".pcd")
        else:
            logging.warning("Found no agent {0} in scenario root {1}".format(agent_id, scenario_root))
            continue
        
        # load point cloud and project to ego's space
        if agent_id == ego_id:
            pcd_list.append(load_point_cloud(pcd_path, color=pointcloud_color))
        else:
            pcd_list.append(
                load_point_cloud(pcd_path, ego_pose=anno_info['lidar_pose'], color=pointcloud_color)
            )

    # 2.3) load bounding boxes
    map_info_path = os.path.join(scenario_root, "map", timestamp+".yaml")
    backend = "open3d"
    bbox_list = []
    for obj_id,obj_info in load_yaml(map_info_path)["objects"].items():
        # decide color
        # ego: green; occluder: purple; occludee: red; other: blue
        if obj_id == ego_id:
            color = COLOR_MAPPING["green"]
        elif obj_id == occluder_id:
            color = COLOR_MAPPING["puple"]
        elif obj_id == occludee_id:
            color = COLOR_MAPPING["red"]
        else:
            color = COLOR_MAPPING["blue"]

        bbox_list.append(create_3d_bbox_OPV2V(obj_info, mat_w2l, backend=backend, color=color))


    # 3) Visualization
    # 3.1) set render option
    W,H = 1980,1080
    W,H = 1080,1080
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=W, height=H, visible=True)
    render_opt = visualizer.get_render_option()
    render_opt.background_color = np.array(COLOR_MAPPING["black"])
    render_opt.point_size = 1.0
    render_opt.line_width = 15.0

    # 3.2) add geometries
    for pcd in pcd_list:
        visualizer.add_geometry(pcd)
    for bbox in bbox_list:
        visualizer.add_geometry(bbox)

    visualizer.poll_events()
    visualizer.update_renderer()

    # 3.3) save as image
    if save_path:
        scenario_name = os.path.basename(scenario_root)
        if scenario_name == "":
            scenario_name = os.path.basename(os.path.dirname(scenario_root))
        save_path = os.path.join(save_path, scenario_name+"."+timestamp+".png")
        visualizer.capture_screen_image(save_path, do_render=True)
        visualizer.destroy_window()
    else:
        visualizer.run()


# =====================================================================
# - running
# =====================================================================

if __name__ == '__main__':
    # parse args
    argparser = argparse.ArgumentParser(
        description='Point Cloud Visualization')
    argparser.add_argument(
        '--path',
        help = 'data path of Point Cloud'
        )
    argparser.add_argument(
        '--save-path',
        default=None,
        help="Dir to save the image"
    )
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    viz_pointcloud_test_scene(args.path, args.save_path)
    

