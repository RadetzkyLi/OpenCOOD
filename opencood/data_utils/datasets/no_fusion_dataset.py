#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   no_fusion_dataset.py
@Date    :   2024-02-05
@Author  :   Runsheng Xu <rxx3386@ucla.edu>
@Modified:   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   No fusion
'''

import random
import math
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils import box_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import late_fusion_dataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import \
    mask_points_by_range, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


class NoFusionDataset(late_fusion_dataset.LateFusionDataset):
    """
    No fusion is a variant of late fusion, i.e., only ego's lidar
    is considered.
    
    """
    def __init__(self, params, visualize, partname):
        super().__init__(params, visualize, partname)


    def get_item_test(self, base_data_dict):
        processed_data_dict = OrderedDict()
        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                scenario_id = cav_content['scenario_id']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        # key objects
        key_object_ids = self.get_key_object_ids_in_scenario(scenario_id)

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # For no fusion, only ego is considered
            if cav_id != ego_id:
                continue

            # find the transformation matrix from current cav to ego.
            cav_lidar_pose = selected_cav_base['params']['lidar_pose']
            transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)

            selected_cav_processed = \
                self.get_item_single_car(cav_id, 
                                         selected_cav_base, 
                                         key_object_ids=key_object_ids)
            selected_cav_processed.update({'transformation_matrix':
                                               transformation_matrix})
            if cav_id == ego_id:
                selected_cav_processed["key_object_ids"] = key_object_ids

            update_cav = "ego" if cav_id == ego_id else cav_id
            processed_data_dict.update({update_cav: selected_cav_processed})

        # test metric dict, in which communication volume is computed 
        # after calling postprocess when validation/testing
        test_metric_dict = self.get_test_metric_dict_instance(
            len(base_data_dict)
        )
        processed_data_dict['ego'].update({'test_metric_dict': test_metric_dict})

        return processed_data_dict

    def __getitem__old(self, idx):
        """
        Only consider ego. Variant of Early Fusion.
        """
        base_data_dict = self.retrieve_base_data(idx)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                scenario_id = cav_content['scenario_id']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        projected_lidar_stack = []
        object_stack = []
        object_id_stack = []
        obejct_category_stack = []

        # get ego to process information
        selected_cav_base = base_data_dict[ego_id]
        selected_cav_processed = self.get_item_single_car(
            ego_id,
            selected_cav_base,
            ego_lidar_pose
        )
        # all these lidar and object coordinates are projected to ego
        # already.
        projected_lidar_stack.append(
            selected_cav_processed['projected_lidar'])
        
        object_stack.append(selected_cav_processed['object_bbx_center'])
        object_id_stack = selected_cav_processed['object_ids']
        obejct_category_stack = selected_cav_processed['object_categories']
            
        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]
        
        # make sure bouding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # convert list to numpy array, (N, 4)
        projected_lidar_stack = np.vstack(projected_lidar_stack)

        # data augmentation
        projected_lidar_stack, object_bbx_center, mask = \
            self.augment(projected_lidar_stack, object_bbx_center, mask)
        
        # we do lidar filtering in the stacked lidar
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                     self.params['preprocess'][
                                                         'cav_lidar_range'
                                                     ])
        
        # augmentation may remove some of the bbx out of range
        object_bbx_center_valid = object_bbx_center[mask == 1]
        object_bbx_center_valid, range_mask = \
            box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                     self.params['preprocess'][
                                                         'cav_lidar_range'],
                                                     self.params['postprocess'][
                                                         'order'],
                                                     return_mask=True
                                                     )
        mask[object_bbx_center_valid.shape[0]:] = 0
        object_bbx_center[:object_bbx_center_valid.shape[0]] = \
            object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0]:] = 0
        unique_indices = list(np.array(unique_indices)[range_mask])

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(projected_lidar_stack)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'object_categories': [obejct_category_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': lidar_dict,
             'label_dict': label_dict,
             'key_object_ids': self.get_key_object_ids_in_scenario(scenario_id)})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                                                   projected_lidar_stack})

        return processed_data_dict