# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm

import torch
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils


from opencood.utils import box_utils
from opencood.utils.common_utils import torch_tensor_to_numpy


def generate_gt_bbox_with_objectId(data_dict, order):
    """
    The base postprocessor will generate 3d groundtruth bounding box.

    This is variant of function `generate_gt_bbox` in opencood.data_utils.post_processor.base_postprocessor. 
    The modified version will generate 3d bounding box and its object ids.

    Parameters
    ----------
    data_dict : dict
        The dictionary containing the origin input data of model.
    order : str
        'lwh' or 'hwl'

    Returns
    -------
    gt_box3d_tensor : torch.Tensor
        The groundtruth bounding box tensor, shape (N, 8, 3).
    """
    gt_box3d_list = []
    # used to avoid repetitive bounding box
    object_id_list = []

    for cav_id, cav_content in data_dict.items():
        # used to project gt bounding box to ego space
        transformation_matrix = cav_content['transformation_matrix']

        object_bbx_center = cav_content['object_bbx_center']
        object_bbx_mask = cav_content['object_bbx_mask']
        object_ids = cav_content['object_ids']
        object_bbx_center = object_bbx_center[object_bbx_mask == 1]

        # convert center to corner
        object_bbx_corner = \
            box_utils.boxes_to_corners_3d(object_bbx_center,
                                            order)
        projected_object_bbx_corner = \
            box_utils.project_box3d(object_bbx_corner.float(),
                                    transformation_matrix)
        gt_box3d_list.append(projected_object_bbx_corner)

        # append the corresponding ids
        object_id_list += object_ids

    # gt bbx 3d
    gt_box3d_list = torch.vstack(gt_box3d_list)
    # some of the bbx may be repetitive, use the id list to filter
    gt_box3d_selected_indices = \
        [object_id_list.index(x) for x in set(object_id_list)]
    gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

    # filter the gt_box to make sure all bbx are in the range
    mask = \
        box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
    gt_box3d_tensor = gt_box3d_tensor[mask, :, :]

    # get object ids
    selected_object_ids = torch.IntTensor(object_id_list)[gt_box3d_selected_indices]
    selected_object_ids = selected_object_ids[mask.cpu()]

    return gt_box3d_tensor,selected_object_ids

def get_scenario_name(scenario_dir):
    return os.path.basename(scenario_dir)

def save_prediction_gt_v2(pred_tensor, score_tensor, gt_tensor, pcd, gt_ids_tensor, prefix, save_dir):
    """
    Save prediction and gt tensor to npy file 
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)
    score_np = torch_tensor_to_numpy(score_tensor)

    # np.save(os.path.join(save_dir, prefix + '__pcd.npy'), pcd_np)
    # np.save(os.path.join(save_dir, prefix + '__pred.npy'), pred_np)
    # np.save(os.path.join(save_dir, prefix + '__gt_test.npy'), gt_np)
    # np.save(os.path.join(save_dir, prefix + '__gt_id.npy'), gt_ids_tensor.cpu().numpy())
    # np.save(os.path.join(save_dir, prefix + '__score.npy'), score_np)

    np.savez(os.path.join(save_dir, prefix+".npz"), pred=pred_np, score=score_np, gt=gt_np, gt_ids=gt_ids_tensor.cpu().numpy())


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    parser.add_argument('--as_no_fusion', action='store_true',
                        help='If set there is no cooperation')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    if opt.as_no_fusion:
        print("There will be no fusion!!!")
        opencood_dataset.as_no_fusion = True
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    for i, batch_data in tqdm(enumerate(data_loader)):
        # print(i)
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
    
                # save extra info here
                # will save batch_data['ego']['object_ids']
                _,gt_ids_tensor = generate_gt_bbox_with_objectId(batch_data, opencood_dataset.post_processor.params['order'])
                
                timestamp = batch_data['ego']['extra_info']['timestamp']
                scanerio_name = get_scenario_name(batch_data['ego']['extra_info']['scenario_dir'])
                ego_id = str(batch_data['ego']['extra_info']['ego_id'])
                prefix = "%s__%s__%s"%(scanerio_name, ego_id, timestamp)
                save_prediction_gt_v2(pred_box_tensor, 
                                      pred_score,
                                      gt_box_tensor, 
                                      batch_data['ego']['origin_lidar'][0], 
                                      gt_ids_tensor, 
                                      prefix, 
                                      npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
