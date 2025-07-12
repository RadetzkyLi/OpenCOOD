#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   vis_camera.py
@Date    :   2025-05-31
@Author  :   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Visualize camera data.
'''

import os
import argparse
import yaml
import logging
import numpy as np
import cv2

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

from opencood.utils import box_utils
from opencood.utils import camera_utils
from opencood.utils.transformation_utils import x1_to_x2

# =====================================================================
# - Common Constants and Funcions
# =====================================================================
MAX_OBJECT_NUM = 100
BBX_ORDER = "lwh"
CAMERA_NAMES_MULTI = ["cameraFront", "cameraRear", "cameraLeft", "cameraRight", "cameraForward", "cameraBackward"]
CAMERA_NAMES_OPV2V = ["camera0", "camera1", "camera2", "camera3"]
FILTER_RANGE_CAV = [-140, -40, -3, 140, 40, 3]
FILTER_RANGE_RSU = [-140, -100, -6, 140, 100, 1]

def load_yaml(path):
    with open(path, encoding='utf8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  
    return data

def load_anno_info(anno_path:str, image_path:str=None):
    if anno_path is None:
        # infer annotation path
        fname = os.path.basename(image_path)
        timestamp = fname.split("_")[0]
        anno_path = os.path.join(
            os.path.dirname(image_path),
            timestamp + ".yaml"
        )
        logging.info("Using inferred anno_path")
    else:
        logging.info("Using specified anno_path")

    if os.path.exists(anno_path):
        return load_yaml(anno_path)
    else:
        raise RuntimeError("The annotation path is not existed: {0}".format(anno_path))
    
def get_agent_from_path(path, dataset_format:str="multi-v2x"):
    """Get agent id and type from data path."""
    agent = os.path.basename(os.path.dirname(path))
    if dataset_format == "multi-v2x":
        if agent.startswith("cav"):
            agent_id = int(agent.split("cav_")[1])
            return "cav", agent_id
        elif agent.startswith("rsu"):
            agent_id = int(agent.split("rsu_")[1])
            return "rsu", agent_id
        else:
            raise ValueError("Unknown agent '{0}'".format(agent))
    elif dataset_format == "opv2v":
        agent_id = int(agent)
        return "rsu", agent_id

def get_camera_name_from_path(image_path:str):
    fname = os.path.basename(image_path).split(".")[0]
    camera_name = fname.split("_")[1]
    return camera_name

def get_camera_names(dataset_format:str):
    assert dataset_format in ["opv2v", "multi-v2x"]
    if dataset_format == "opv2v":
        return CAMERA_NAMES_OPV2V
    elif dataset_format == "multi-v2x":
        return CAMERA_NAMES_MULTI

def pose_to_carla_transform(pose):
    """From pose in carla world coordinate (x,y,z,roll,yaw,pitch)
    to carla.Transform."""
    return carla.Transform(
        carla.Location(pose[0], pose[1], pose[2]),
        carla.Rotation(roll=pose[3], yaw=pose[4], pitch=pose[5])
    )

def calculate_camera_extrinsic(anno:dict, camera_name:str, backend:str="opencood"):
    """Calculate camera extrinsic using pose and opencood's 
    transformation utils"""
    backend = "carla"
    assert backend in ["opencood", "carla"]
    
    key = "coords" if "coords" in anno[camera_name] else "cords"
    if backend == "opencood":
        extrinsic = x1_to_x2(anno["lidar_pose"], anno[camera_name][key])
    elif backend == "carla" and CARLA_AVAILABLE:
        lidar_transform = pose_to_carla_transform(anno['lidar_pose'])
        lidar_to_world = np.array(lidar_transform.get_matrix())
        camera_transform = pose_to_carla_transform(anno[camera_name]['coords'])
        world_to_camera = np.array(camera_transform.get_inverse_matrix())
        lidar_to_camera = np.dot(world_to_camera, lidar_to_world)
        extrinsic = lidar_to_camera
    else:
        raise RuntimeError("Package `carla` is not installed!")

    return extrinsic

# =====================================================================
# - Visualization 
# =====================================================================
def get_bbx(anno:dict, ego_id:int, ego_type:str, 
            camera_name:str,
            dataset_format:str="multi-v2x", 
            by_lidar:bool=True):
    """
    
    Parameters
    ----------

    by_lidar : bool
        If True, the bbx would be projected to LiDAR's frame first.
        Otherwise the bbs would be projected to camera's frame directly.
        Defaults to True.
    """
    object_key = "objects" if dataset_format == "multi-v2x" else "vehicles"
    coord_key = "coords" if dataset_format == "multi-v2x" else "cords"

    # get bbx in target frame
    output_dict = {}   # {obj_id: bbx}, bbx is in xyzlwhyaw
    box_utils.project_world_objects(
        anno[object_key],
        output_dict,
        lidar_pose=anno["lidar_pose"] if by_lidar else anno[camera_name][coord_key],
        lidar_range=FILTER_RANGE_CAV if ego_type == "cav" else FILTER_RANGE_RSU,
        order=BBX_ORDER
    )

    object_np = np.zeros((MAX_OBJECT_NUM, 7))
    mask = np.zeros(MAX_OBJECT_NUM)
    object_ids = []

    for i, (object_id, object_bbx) in enumerate(output_dict.items()):
        # exclude ego
        if ego_id == object_id:
            continue
        object_np[i] = object_bbx[0, :]
        mask[i] = 1
        object_ids.append(object_id)

    object_np = object_np[mask == 1]

    if not by_lidar:
        camera_extrinsic = np.eye(4, dtype=np.float32)
    else:
        camera_extrinsic = np.array(anno[camera_name]["extrinsic"])
        camera_extrinsic_cal = calculate_camera_extrinsic(anno, camera_name)
        if not np.allclose(camera_extrinsic, camera_extrinsic_cal, atol=0.001):
            logging.warning("Extrinsic: Saved vs calculated not same!!!")
            logging.debug(camera_extrinsic)
            logging.debug(camera_extrinsic_cal)
        else:
            logging.info("Extrinsic: Saved vs calculated OK.")

    camera_intrinsic = np.array(anno[camera_name]["intrinsic"])
    
    # convert to corner representation
    objects = box_utils.boxes_to_corners_3d(object_np, BBX_ORDER)
    # project objects coordinate from lidar space to camera space
    # shape: (N, 8, 3)
    object_camera = camera_utils.project_3d_to_camera(objects,
                                                      camera_intrinsic,
                                                      camera_extrinsic)

    return objects, object_camera, object_ids

def visualize_agent_camera_bbx(image_path:str, anno_path:str=None, 
                               dataset_format:str="multi-v2x",
                               draw_3d=True, 
                               color=(0, 255, 0), thickness=2):
    """
    Visualize bbx on the 2d image for a certain agent
    and a certain camera.

    Parameters
    ----------
    image_path : str
        The path of image.

    anno_path : str
        The annotation path. If `None`, it will be inferred from `image_path`.

    dataset_format : str
        The format of data, "multi-v2x" or "opv2v".
    
    draw_3d : bool
        Draw 2d bbx or 3d bbx on image.

    color : tuple
        Bbx draw color.

    thickness : int
        Draw thickness.

    Returns
    -------
    The drawn image.
    """
    assert os.path.exists(image_path), "The image not existed!!!"

    anno = load_anno_info(anno_path, image_path)
    camera_name = get_camera_name_from_path(image_path)
    ego_type, ego_id = get_agent_from_path(image_path, dataset_format)
    objects, object_camera, object_ids = get_bbx(anno, ego_id, ego_type, camera_name, dataset_format)
    logging.info("{0}: agent {1}, there are {2} objects: {3}".format(
        camera_name, ego_id, len(object_ids), object_ids))

    camera_rgb = cv2.imread(image_path)
    if draw_3d:
        draw_rgb = camera_utils.draw_3d_bbx(camera_rgb,
                                            object_camera,
                                            color,
                                            thickness)
    else:
        draw_rgb = camera_utils.draw_2d_bbx(camera_rgb,
                                            objects,
                                            color,
                                            thickness)
        
    return draw_rgb

def visualize_agent_bbx(image_path:str, anno_path:str=None, 
                        dataset_format:str="multi-v2x",
                        draw_3d=True, color=(0, 255, 0), thickness=2):
    """
    
    Returns
    -------
    A list of drawn image.
    """
    # find out all images of the agent
    fname = os.path.basename(image_path)
    suffix = fname.split(".")[-1]
    prefix = fname.split(".")[0].split("_")[0]  # the timestamp

    data_dir = os.path.dirname(image_path)
    image_paths = []
    camera_names = get_camera_names(dataset_format)
    for camera in camera_names:
        another_image_path = os.path.join(
            data_dir,
            prefix + "_" + camera + "." + suffix
        )
        if os.path.exists(another_image_path):
            image_paths.append(another_image_path)

    # draw each image
    draw_image_list = []
    for path in image_paths:
        draw_image = visualize_agent_camera_bbx(path, 
                                                anno_path,
                                                dataset_format,
                                                draw_3d,
                                                color,
                                                thickness)
        draw_image_list.append(draw_image)
    
    return draw_image_list


# =====================================================================
# - running
# =====================================================================
def main(args):
    if args.plot_all:
        draw_image_list = visualize_agent_bbx(
            args.image_path,
            args.anno_path,
            dataset_format=args.dataset_format,
            draw_3d=args.draw_3d
        )
        camera_utils.plot_agent(draw_image_list)
    else:
        draw_rgb = visualize_agent_camera_bbx(
            args.image_path,
            args.anno_path,
            dataset_format=args.dataset_format,
            draw_3d=args.draw_3d
        )
        camera_utils.plot_agent([draw_rgb])


if __name__ == '__main__':
    # parse args
    argparser = argparse.ArgumentParser(
        description='Camera Data Visualization')
    argparser.add_argument(
        '--image-path',
        help = 'data path of image'
        )
    argparser.add_argument(
        '--anno-path',
        default=None,
        help='Annotation file path for the image. If `None`, inferred from image path.'
    )
    argparser.add_argument(
        '--save-path',
        default=None,
        help="Dir to save the image. Not enabled."
    )
    argparser.add_argument(
        '--plot-all',
        action='store_true',
        help="If set, all cameras related to this agent are drawn."
    )
    argparser.add_argument(
        '--dataset-format',
        default="multi-v2x",
        help="One of ['multi-v2x', 'opv2v']. 'multi-v2x' for Muti-V2X and CoRTSG, 'opv2v' for OPV2V and V2XSet."
    )
    argparser.add_argument(
        '--draw-3d',
        default=True,
        type=bool,
        help="If set, draw 3D bbx."
    )
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    main(args)