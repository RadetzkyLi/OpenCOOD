# -*- coding: utf-8 -*-
# Author: Rongsong Li <rongsong.li@qq.com>
# License: TDG-Attribution-NonCommercial-NoDistrib
"""
A command line tool for visualizing saved 3d object detection results.
"""
import os
import numpy as np
import argparse
import open3d as o3d

from opencood.visualization import vis_utils
from opencood.utils import box_utils
from opencood.utils import common_utils

# ========================================================
# - arg parser
# ========================================================
def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--npy-dir', type=str, default=None,
                           help="root dir of the result")
    argparser.add_argument('--npz-dir', type=str, default=None,)
    argparser.add_argument('--index', type=int, 
                           help="index of the frame")
    
    args = argparser.parse_args()

    # visualize
    if args.npy_dir is not None:
        visualize_result_from_npy(args.npy_dir, args.index)
    elif args.npz_dir is not None:
        visualize_result_from_npz(args.npz_dir, args.index)


# =========================================================
# - visualization utility
# =========================================================
def bbx2oabb(bbx_corner, order='hwl', color=(0, 0, 1)):
    """
    Convert the torch tensor bounding box to o3d oabb for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    oabbs : list
        The list containing all oriented bounding boxes.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)
    oabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        # bbx[:, :1] = - bbx[:, :1]
        bbx[:, 1] = -bbx[:, 1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        oabb = tmp_pcd.get_oriented_bounding_box()
        oabb.color = color
        oabbs.append(oabb)

    return oabbs

def visualize_single_sample_output_gt(pred_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis=True,
                                      save_path='',
                                      mode='constant'):
    """
    Visualize the prediction, groundtruth with point cloud together.

    Parameters
    ----------
    pred_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.

    mode : str
        Color rendering mode.
    """

    def custom_draw_geometry(pcd, pred, gt):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # black
        opt.point_size = 1.0

        vis.add_geometry(pcd)
        for ele in pred:
            vis.add_geometry(ele)
        for ele in gt:
            vis.add_geometry(ele)

        vis.run()
        vis.destroy_window()

    if len(pcd.shape) == 3:
        pcd = pcd[0]
    origin_lidar = pcd
    if not isinstance(pcd, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(pcd)

    origin_lidar_intcolor = \
        vis_utils.color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                                else origin_lidar[:, 2], mode=mode)
    # left -> right hand
    # origin_lidar[:, :1] = -origin_lidar[:, :1]
    origin_lidar[:, 1] = -origin_lidar[:, 1]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    oabbs_pred = bbx2oabb(pred_tensor, color=(1, 0, 0)) if pred_tensor is not None else []
    oabbs_gt = bbx2oabb(gt_tensor, color=(0, 1, 0))

    visualize_elements = [o3d_pcd] + oabbs_pred + oabbs_gt
    if show_vis:
        custom_draw_geometry(o3d_pcd, oabbs_pred, oabbs_gt)
    if save_path:
        vis_utils.save_o3d_visualization(visualize_elements, save_path)


# =========================================================
# - visualization
# =========================================================
def visualize_result_from_npy(npy_dir, index):
    pcd_path = os.path.join(npy_dir, "%04d_pcd.npy"%index)
    gt_path = os.path.join(npy_dir, "%04d_gt.npy_test.npy"%index)
    pr_path = os.path.join(npy_dir, "%04d_pred.npy"%(index))
    visualize_single_sample_output_gt(
        np.load(pr_path),
        np.load(gt_path),
        np.load(pcd_path),
        show_vis=True
    )

def visualize_result_from_npz(npz_dir:str, index:int):
    npz_path = os.path.join(npz_dir, str(index)+".npz")
    data = np.load(npz_path, allow_pickle=True)
    visualize_single_sample_output_gt(
        data['pred'],
        data['gt'],
        data['pcd'],
        show_vis=True
    )


# ========================================================
# - running
# ========================================================
if __name__ == '__main__':
    main()