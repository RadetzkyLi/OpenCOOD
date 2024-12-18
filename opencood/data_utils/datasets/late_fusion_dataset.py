#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   late_fusion_dataset.py
@Date    :   2024-02-02
@Author  :   Runsheng Xu <rxx3386@ucla.edu>
@Modified:   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Late fusion
'''


import random
from collections import OrderedDict

import numpy as np
import torch

from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils
from opencood.utils.pcd_utils import \
    mask_points_by_range, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


class LateFusionDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """
    def __init__(self, params, visualize, partname='test'):
        super(LateFusionDataset, self).__init__(params, visualize, partname)
        
        train_flag = True if partname == 'train' else False
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train_flag)
        self.post_processor = build_postprocessor(params['postprocess'], 
                                                  train_flag)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        if self.partname == 'train':
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(base_data_dict)

        return reformat_data_dict
    
    def process_ego_pose_for_generating_object_center(self, ego_pose):
        """
        For OPV2V, V2XSet and Multi-V2X, the `ego_pose` is a list of size 6 and nothing needs doing;
        for V2V4Real, the `pose` is a matrix of shape (4,4), we need to convert it to identity matrix.

        Parameters
        ----------
        ego_pose : list|ndarray
            The ego's lidar pose.

        Returns
        -------
        ego_pose : list|ndarray
            The ego's lidar pose that needs to be used for generating object center.
        """
        if self.dataset_format == 'v2v4real':
            ego_pose = np.identity(4)
        else:
            ego_pose = ego_pose
        
        return ego_pose


    def get_item_single_car(self, cav_id:int, selected_cav_base, key_object_ids:list=[]):
        """
        Process a single CAV's information for the train/test pipeline.

        Parameters
        ----------
        cav_id : int
            The selected CAV's id.
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        key_object_ids : list
            The objects we focus on.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # remove points that hit ego vehicle
        lidar_np = self.mask_ego_points(lidar_np, selected_cav_base["scenario_id"], cav_id)
        
        # generate the bounding box(n, 7) under the cav's space
        object_bbx_center, object_bbx_mask, object_ids, object_categories = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       self.process_ego_pose_for_generating_object_center(
                                                           selected_cav_base['params']['lidar_pose']),
                                                       return_object_categories=True,
                                                       key_object_ids=key_object_ids)
        
        # data augmentation
        lidar_np, object_bbx_center, object_bbx_mask = \
            self.augment(lidar_np, object_bbx_center, object_bbx_mask)

        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(lidar_np)
        selected_cav_processed.update({'processed_lidar': lidar_dict})

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()
        selected_cav_processed.update({'anchor_box': anchor_box})

        selected_cav_processed.update({'object_bbx_center': object_bbx_center,
                                       'object_bbx_mask': object_bbx_mask,
                                       'object_ids': object_ids,
                                       'object_categories': object_categories})

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=object_bbx_mask)
        selected_cav_processed.update({'label_dict': label_dict})

        return selected_cav_processed

    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()

        # during training, we return a random cav's data
        if not self.visualize:
            selected_cav_id, selected_cav_base = \
                random.choice(list(base_data_dict.items()))
        else:
            selected_cav_id, selected_cav_base = \
                list(base_data_dict.items())[0]

        selected_cav_processed = self.get_item_single_car(selected_cav_id, selected_cav_base)
        processed_data_dict.update({'ego': selected_cav_processed})

        return processed_data_dict

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

        # for late fusion, we also need to stack the lidar for better
        # visualization
        if self.visualize:
            projected_lidar_list = []
            origin_lidar = []

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']
            object_categories = cav_content['object_categories']

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})
            if self.visualize:
                transformation_matrix = cav_content['transformation_matrix']
                origin_lidar = [cav_content['origin_lidar']]

                projected_lidar = cav_content['origin_lidar']
                projected_lidar[:, :3] = \
                    box_utils.project_points_by_matrix_torch(
                        projected_lidar[:, :3],
                        transformation_matrix)
                projected_lidar_list.append(projected_lidar)

            # processed lidar dictionary
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar']])
            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(
                    np.array(cav_content['transformation_matrix'])).float()

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'object_categories': object_categories,
                                        'transformation_matrix': transformation_matrix_torch})

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})

        if self.visualize:
            projected_lidar_stack = torch.from_numpy(
                np.vstack(projected_lidar_list))
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})

        # get key object ids for ego
        output_dict['ego'].update({
            "key_object_ids": batch['ego']['key_object_ids']
        })
        # test metric
        output_dict['ego'].update({
            "test_metric_dict": batch['ego']['test_metric_dict']
        })
       
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

        # compute communication volume in bytes
        comm_vol = self.compute_communication_volume_for_late_fusion(output_dict)
        # update test_metric_dict
        data_dict['ego']['test_metric_dict'].update({
            'comm_vol_list': [comm_vol]
        })

        return pred_box_tensor, pred_score, gt_box_tensor
    
    @staticmethod
    def compute_communication_volume_for_late_fusion(output_dict):
        """
        Compute the total communication volume in bytes for late fusion.
        We assume all agents share predicted bbx with each other.

        Parameters
        ----------
        output_dict : dict
            The `output_dict` after calling dataset.postprocess in which
            the number of prediction for each agent is counted.

        Returns
        -------
        comm_vol : float
            The total communication volume in bytes, i.e., receive + send.
        """
        ego_num_of_pred = output_dict['ego']['num_of_pred']
        others_num_of_pred = \
            np.sum([item['num_of_pred'] for item in output_dict.values()]) \
            - ego_num_of_pred
        
        receive_num = others_num_of_pred
        send_num = (len(output_dict)-1) * ego_num_of_pred
        
        # each bbx denoted by xyzlwhr
        # the value is of float32
        comm_vol = (receive_num + send_num) * 7 * 32/8

        return comm_vol