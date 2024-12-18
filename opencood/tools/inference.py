#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   inference.py
@Date    :   2024-02-02
@Author  :   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Inference (compatible for various dataset formats)
'''

import argparse
import os
import time
from tqdm import tqdm
import logging

import torch
# to avoid 'RuntimeError: received 0 items of ancdata'
torch.multiprocessing.set_sharing_strategy('file_system')
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


def test_parser():
    parser = argparse.ArgumentParser(description="V2X 3D Detection Inference")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='no, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can not be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npz', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npz file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    parser.add_argument('--dataset_format', type=str, default='test',
                        help='Format of dataset. "test" or "opv2v" or "multi-v2x"')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='If set, it will ovewrite `root_dir` in yaml.')
    opt = parser.parse_args()

    # logging
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    return opt

def update_hypes(hypes, opt):
    hypes['dataset_format'] = opt.dataset_format
    if opt.dataset_root is not None:
        hypes['root_dir'] = opt.dataset_root

    # prpreocess
    hypes['preprocess']['keep_key_objects_anyway'] = True

    # no async
    if 'wild_setting' in hypes:
        if 'async' in hypes['wild_setting']:
            hypes['wild_setting'].update({
                "async": False
            })

def get_dataset_name_from_root(root:str):
    name = os.path.basename(root)
    if name == '':
        name = os.path.basename(os.path.dirname(root))
    return name

def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'
    assert opt.dataset_format in ['test', 'opv2v', 'multi-v2x', 'v2v4real']

    hypes = yaml_utils.load_yaml(None, opt)
    update_hypes(hypes, opt)

    print('Dataset Building')
    print('Dataset name:', get_dataset_name_from_root(hypes['root_dir']))
    opencood_dataset = build_dataset(hypes, visualize=True, partname='test')
    print(f"{len(opencood_dataset)} samples found.")
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
    # device = 'cpu'

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 
                         'detail': {'id': [], 'category': [], 'is_key_object': [], 'is_recalled': [], 'translation_error': []}},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 
                         'detail': {'id': [], 'category': [], 'is_key_object': [], 'is_recalled': [], 'translation_error': []}},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 
                         'detail': {'id': [], 'category': [], 'is_key_object': [], 'is_recalled': [], 'translation_error': []}}
                }
    
    # Some test metrics
    test_metric_dict = opencood_dataset.get_empty_test_metric_dict()

    exclude_single_cav = True if opt.dataset_format == 'test' else False
    # valid test samples, i.e., at least two agents
    num_of_valid_samples = 0
    num_pred = 0

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
        # exclude single car case
        if exclude_single_cav and \
            batch_data['ego']['test_metric_dict']['num_of_conn_list'][0]<2:
            continue
        num_of_valid_samples += 1

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
            elif opt.fusion_method == 'no':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_no_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            # get key objects
            key_object_ids = batch_data['ego'].get('key_object_ids', [])
            gt_box_tensor,gt_object_id_tensor,gt_object_category_list = \
                inference_utils.generate_gt_bbx_with_id_and_category(
                    batch_data,
                    opencood_dataset.post_processor.params['order'],
                    key_object_ids
            )
            num_pred += len(pred_box_tensor) if pred_box_tensor is not None else 0

            eval_utils.caluclate_tp_fp_4test(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.3,
                                            gt_object_id_tensor,
                                            key_object_ids,
                                            gt_object_category_list)
            eval_utils.caluclate_tp_fp_4test(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.5,
                                            gt_object_id_tensor,
                                            key_object_ids,
                                            gt_object_category_list)
            eval_utils.caluclate_tp_fp_4test(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.7,
                                            gt_object_id_tensor,
                                            key_object_ids,
                                            gt_object_category_list)
            if opt.save_npz:
                npz_save_dir = os.path.join(opt.model_dir, 'npz')
                if not os.path.exists(npz_save_dir):
                    os.makedirs(npz_save_dir)

                prefix = str(i)
                inference_utils.save_prediction_gt_id(
                    pred_box_tensor,
                    pred_score,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'][0],
                    gt_object_id_tensor,
                    prefix,
                    npz_save_dir
                )

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

            # test metric
            opencood_dataset.append_test_metric_dict(
                test_metric_dict,
                batch_data['ego']['test_metric_dict'])
            
    eval_utils.eval_final_results_4test(result_stat,
                                        opt.model_dir,
                                        opt.global_sort_detections)
    
    print("Valid samples:", num_of_valid_samples)
    print("pred:", num_pred)
    # testing metric
    opencood_dataset.print_test_metric_dict(test_metric_dict)

    if opt.show_sequence:
        vis.destroy_window()

if __name__ == '__main__':
    main()
