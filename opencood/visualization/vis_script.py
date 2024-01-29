# -*- coding: utf-8 -*-
# Author: Rongsong Li <rongsong.li@qq.com>
# License: TDG-Attribution-NonCommercial-NoDistrib
"""
A command line tool for visualizing 3d object detection.
"""
import os
import numpy as np
import argparse

from opencood.visualization import vis_utils

# ========================================================
# - arg parser
# ========================================================
def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--npy-dir', type=str, help="root dir of the result")
    argparser.add_argument('--index', type=int, help="index of the frame")
    
    args = argparser.parse_args()

    # visualize
    visualize_result_from_npy(args.npy_dir, args.index)

# =========================================================
# - visualization
# =========================================================
def visualize_result_from_npy(npy_dir, index):
    pcd_path = os.path.join(npy_dir, "%04d_pcd.npy"%index)
    gt_path = os.path.join(npy_dir, "%04d_gt.npy_test.npy"%index)
    pr_path = os.path.join(npy_dir, "%04d_pred.npy"%(index))
    vis_utils.visualize_single_sample_output_gt(
        np.load(pr_path),
        np.load(gt_path),
        np.load(pcd_path),
        show_vis=True
    )


# ========================================================
# - running
# ========================================================
if __name__ == '__main__':
    main()