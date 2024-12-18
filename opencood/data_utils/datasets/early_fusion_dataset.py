#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   early_fusion_dataset.py
@Date    :   2024-02-01
@Author  :   Runsheng Xu <rxx3386@ucla.edu>
@Modified:   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Early fusion dataset
'''

import random
import math
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils import box_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import \
    mask_points_by_range, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


class EarlyFusionDataset(basedataset.BaseDataset):
    """
    This dataset is used for early fusion, where each CAV transmit the raw
    point cloud to the ego vehicle.
    """
    def __init__(self, params, visualize, partname):
        super(EarlyFusionDataset, self).__init__(params, visualize, partname)
        
        train_flag = True if partname == 'train' else False
        self.pre_processor = build_preprocessor(params['preprocess'], train_flag)
        self.post_processor = build_postprocessor(params['postprocess'], train_flag)


    def get_transmitted_point_cloud(self, lidar_np, transformation_matrix):
        """
        Get the transmitted raw point cloud for early fusion.
        steps: 
        1. project to ego's space.
        2. crop to ego's evaluation range.

        Parameters
        ----------
        agent_id : 
            The agent's id
        lidar_np : np.ndarray
            The agent's raw lidar after removing points hitting self. 
            Shape: (N, C).

        transformation_matrix : np.ndarray
            From agent to ego. (4, 4). None means the lidar have been 
            projected.

        Returns
        -------
        transmitted_lidar_np : np.ndarray
            (N', C)
        """
        # 1) Project the lidar to ego's space
        if transformation_matrix is not None:
            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                         transformation_matrix)

        # 2) Crop
        transmitted_lidar_np = mask_points_by_range(lidar_np,
                                                    self.params['preprocess'][
                                                        'cav_lidar_range'
                                                    ])
        
        return transmitted_lidar_np

    def compute_communication_volume(self, selected_cav_info:dict, ego_id:int):
        """
        Compute communication volume for early fusion.
        We assume the agents share the raw lidar with each other.
        Only caculate this when testing.

        Parameters
        ----------
        selected_cav_info : dict
            Each cav's info including lidar and transformation matrix
            (cav to ego)

        ego_id : int
            The ego's id.

        Returns
        -------
        comm_vol : float
            The total communication volume in bytes, i.e., receive + send.
        """
        # only useful when testing
        if self.partname == 'train':
            return 0.0

        receive_num = 0
        send_num = 0
        for cav_id,cav_info in selected_cav_info.items():
            if cav_id == ego_id:
                continue
            # send to ego
            receive_num += len(self.get_transmitted_point_cloud(
                cav_info['projected_lidar'], 
                None
            ))
            # receive from ego
            send_num += len(self.get_transmitted_point_cloud(
                selected_cav_info[ego_id]['projected_lidar'],
                np.linalg.inv(selected_cav_info[cav_id]['transformation_matrix'])  # ego to other
            ))

        # channels of point cloud
        num_channels = selected_cav_info[ego_id]['projected_lidar'].shape[1]

        # assume the coordinates are of float32
        # count in bytes
        comm_vol = (receive_num + send_num) * num_channels * 32/8        
        
        return comm_vol
    
    def __getitem__(self, idx):
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
        # key objects
        key_object_ids = self.get_key_object_ids_in_scenario(scenario_id)
        
        # used for compute communication volume
        # structure: {cav_id: {'projected_lidar': xxx, 'transformation_matrix': xxx}}
        selected_cav_info = {}

        # loop over all agents to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            selected_cav_processed = self.get_item_single_car(
                cav_id,
                selected_cav_base,
                ego_lidar_pose,
                key_object_ids=key_object_ids
            )
            # all these lidar and object coordinates are projected to ego
            # already.
            projected_lidar_stack.append(
                selected_cav_processed['projected_lidar'])
            
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            obejct_category_stack += selected_cav_processed['object_categories']
            
            selected_cav_info[cav_id] = {
                "projected_lidar": selected_cav_processed['projected_lidar'],
                "transformation_matrix": selected_cav_processed['transformation_matrix']
            }

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
        # key objects will be kept anyway
        if self.params['preprocess']['keep_key_objects_anyway']:
            for key_object_id in key_object_ids:
                range_mask[unique_indices.index(object_id_stack.index(key_object_id))] = True
            object_bbx_center_valid = object_bbx_center[mask == 1][range_mask]

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
        
        # test metric dict
        test_metric_dict = self.get_test_metric_dict_instance(
            len(base_data_dict),
            comm_vol=self.compute_communication_volume(selected_cav_info, ego_id)
        )

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'object_categories': [obejct_category_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': lidar_dict,
             'label_dict': label_dict,
             'key_object_ids': key_object_ids,
             'test_metric_dict': test_metric_dict})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                                                   projected_lidar_stack})

        return processed_data_dict
    
    def get_item_single_car(self, cav_id:int, selected_cav_base, ego_pose, key_object_ids:list=[]):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        cav_id : int
            The selcected CAV's id.
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.
        key_object_ids : list
            The objects we focus on.
       
        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        if self.dataset_format == 'v2v4real':
            return self.get_item_single_car_v2v4real(cav_id, selected_cav_base, ego_pose, key_object_ids)
        else:
            return self.get_item_single_car_standard(cav_id, selected_cav_base, ego_pose, key_object_ids)

    def get_item_single_car_standard(self, cav_id:int, selected_cav_base, ego_pose, key_object_ids:list=[]):
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose)

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids, object_categories = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       ego_pose,
                                                       return_object_categories=True,
                                                       key_object_ids=key_object_ids)
        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = self.mask_ego_points(lidar_np, selected_cav_base["scenario_id"], cav_id)
        # project the lidar to ego space
        lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'object_categories': object_categories,
             'projected_lidar': lidar_np,
             'transformation_matrix': transformation_matrix})

        return selected_cav_processed
    
    def get_item_single_car_v2v4real(self, cav_id:int, selected_cav_base, ego_pose, key_object_ids:list=[]):
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = selected_cav_base['params'][
            'transformation_matrix']

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids, object_categories = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       transformation_matrix,
                                                       return_object_categories=True,
                                                       key_object_ids=key_object_ids)
        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = self.mask_ego_points(lidar_np, selected_cav_base["scenario_id"], cav_id)
        # project the lidar to ego space
        lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'object_categories': object_categories,
             'projected_lidar': lidar_np,
             'transformation_matrix': transformation_matrix})

        return selected_cav_processed
    

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']
            key_object_ids = cav_content['key_object_ids']
            object_categories = cav_content['object_categories']

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})
            if self.visualize:
                origin_lidar = [cav_content['origin_lidar']]

            # processed lidar dictionary
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar']])
            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'object_categories': object_categories,
                                        'key_object_ids': key_object_ids,
                                        'transformation_matrix': transformation_matrix_torch,
                                        'test_metric_dict': cav_content['test_metric_dict']})

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor
    