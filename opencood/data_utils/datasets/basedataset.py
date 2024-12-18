#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   basedataset.py
@Date    :   2024-02-01
@Author  :   Runsheng Xu <rxx3386@ucla.edu>
@Modified:   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Base dataset for V2X perception
'''

import os
import sys
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.scenario_database_manager import build_scenario_database_manager
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2

class BaseDataset:
    """
    Base dataset for all kinds of fusion. Mainly used to initialize the
    database and associate the __get_item__ index with the correct timestamp
    and scenario.
    The dataset is designed for testing scenarios.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the raw point cloud will be saved in the memory
        for visualization.

    Attributes
    ----------
    database_manager : class
        Used to manage scenario data, e.g., retriving samples.

    pre_processor : opencood.pre_processor
        Used to preprocess the raw data.

    post_processor : opencood.post_processor
        Used to generate training labels and convert the model outputs to
        bbx formats.

    data_augmentor : opencood.data_augmentor
        Used to augment data.

    """
    def __init__(self, params, visualize, partname='test') -> None:
        assert partname in ['train', 'val', 'test'], "Unexpected `partname`!"

        self.params = params
        self.visualize = visualize
        self.partname = partname
        self.dataset_format = \
            params.get("dataset_format", "opv2v")  # "opv2v", "v2v4real", "multi-v2x", "test" supported

        self.pre_processor = None
        self.post_processor = None
        train_flag = True if partname == 'train' else False
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train_flag)
        
        # simulate lidar intensity
        if 'intensity_simulation_method' not in params['preprocess']:
            params['preprocess']['intensity_simulation_method'] = 'carla'  # 'carla', None, 'none' supported
        # featurs of point cloud
        if 'num_point_features' not in params['preprocess']['args']:
            params['preprocess']['args']['num_point_features'] = 4

        # keep key objects anyway
        if "keep_key_objects_anyway" not in params['preprocess']:
            params['preprocess']['keep_key_objects_anyway'] = False

        # wild settings
        self.init_wild_settings(params)

        # database manager
        self.database_manager = build_scenario_database_manager(params, partname, self.dataset_format)


    def init_wild_settings(self, params):
        """
        """
        # if the training/testing include noisy setting
        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            # whether to add time delay
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            # localization error
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            # transmission data size
            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed'] \
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb (Megabits)
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

        if 'train_params' not in params or\
                'max_cav' not in params['train_params']:
            self.max_cav = 7
        else:
            self.max_cav = params['train_params']['max_cav']


    def __len__(self):
        return self.database_manager.get_number_of_total_samples()

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    def is_rsu_agent(self, scenario_id:int, agent_id:int):
        """
        Judge whether the given agent is RSU.

        Parameters
        ----------
        agent_id: int
            The agent's id.

        idx: int
            The data index.

        Returns
        -------
        flag: bool
            If the agent is rsu, return True.
        """
        flag = self.database_manager.is_rsu_agent(scenario_id, agent_id)
        return flag
    
    def get_connected_agents_for_timestamp(self, scenario_id, ego_id, timestamp):
        """
        Get connected agents for given agent at given timestamp.
        If the number of total agents exceeds the `max_agent`, 
        random select some of them to use.
        TODO: adaptively select which connectable agents to connect.
        """
        # agents that ego can connect to
        cav_id_list,rsu_id_list = self.database_manager.get_connected_agents_for_timestamp(
            scenario_id, ego_id, timestamp
        )

        num_cav = len(cav_id_list)
        num_rsu = len(rsu_id_list)
        # If ego plus conn exceed max_cav, random select some conn.
        if num_cav+num_rsu> self.max_cav - 1:
            if self.partname == 'val' or self.partname == 'test':
                seed = 55
                random.seed(seed)
            conn_list = random.sample(cav_id_list+rsu_id_list, self.max_cav-1)
        else:
            conn_list = cav_id_list + rsu_id_list

        return conn_list

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix. If set to false, meaning when other cavs
            project their LiDAR point cloud to ego, they are projecting to
            past ego pose.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        scenario_id,ego_id,timestamp = self.database_manager.get_ids_from_sample_index(idx)
        cur_scenario_database = self.database_manager.scenario_database[scenario_id]

        # get connected agents
        conn_list = self.get_connected_agents_for_timestamp(
            scenario_id, ego_id, timestamp
        )

        data = OrderedDict()
        # if the number of point hit on a object exceed or equal to thr_visibility,
        # the object is regarded as visible
        thr_visibility = 1 

        # reorder so that ego be the first
        for agent_id in [ego_id] + conn_list:
            ego_flag = True if agent_id == ego_id else False
            # calculate time delay
            time_delay = self.time_delay_calculation(ego_flag=ego_flag)
            timestamp_delay = self.database_manager.calc_timestamp_with_gap(timestamp, -time_delay)
            # print(timestamp, time_delay, timestamp_delay)

            # in case some agents have no delayed timestamp
            if not ego_flag and \
                not self.database_manager.is_agent_has_timestamp(scenario_id, agent_id, timestamp_delay):
                continue

            data[agent_id] = OrderedDict()
            data[agent_id]['ego'] = ego_flag
            data[agent_id]['scenario_id'] = scenario_id

            # add time delay vehicle parameters
            data[agent_id]['time_delay'] = time_delay
            # load corresponding data into the dictionary
            data[agent_id]['params'] = self.reform_param(
                cur_scenario_database[agent_id],
                cur_scenario_database[ego_id],
                timestamp,
                timestamp_delay,
                cur_ego_pose_flag,
                data[agent_id]['ego'],
                thr_visibility=thr_visibility,
                scenario_id=scenario_id
            )
            data[agent_id]['lidar_np'] = \
                pcd_utils.pcd_to_np(
                    cur_scenario_database[agent_id][timestamp_delay]['lidar'],
                    self.params['preprocess']['intensity_simulation_method'],
                    self.params['preprocess']['args']['num_point_features']
                )

        return data
    
    def time_delay_calculation(self, ego_flag:bool):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : float
            The time delay in seconds.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0.0
        
        # time delay real mode
        if self.async_mode == 'real':
            # in the real mode, time delay = systematic async time + data
            # transmission time + backbone computation time
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == 'sim':
            # in the simulation mode, the time delay is constant
            time_delay = np.abs(self.async_overhead)

        time_delay = time_delay / 1000.0
        return time_delay if self.async_flag else 0.0

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        # there is no need to add noise for real world data
        if self.dataset_format == 'v2v4real':
            return pose
        
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0],
                      pose[1] + xyz_noise[1],
                      pose[2] + xyz_noise[2],
                      pose[3],
                      pose[4] + ryp_std[1],
                      pose[5]]
        return noise_pose

    def reform_param(self, cav_content, ego_content, timestamp_cur,
                     timestamp_delay, cur_ego_pose_flag, is_ego,
                     thr_visibility:int=1, 
                     scenario_id:int=None):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose for other CAVs.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        is_ego : bool
            Whether the agent is the ego.

        thr_visibility : int
            If hit lidar points on an object reached `thr_visibility`, it
            is regarded as visible.

        scenario_id : int
            It cannot be None when `keep_key_objects_anyway` is Ture.

        Return
        ------
        The merged parameters.
        """
        cur_params = load_yaml(cav_content[timestamp_cur]['yaml'])
        delay_params = load_yaml(cav_content[timestamp_delay]['yaml'])

        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if not is_ego and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose,
                                                      self.xyz_noise_std,
                                                      self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose,
                                                    self.xyz_noise_std,
                                                    self.ryp_noise_std)

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                 cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                            cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params['vehicles'] = \
            self.get_visible_objects(cav_content, cur_params, timestamp_cur, 
                                     thr_visibility, scenario_id)
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

        # get key objects
        if is_ego:
            self.get_key_objects_for_agent(delay_params['vehicles'], 
                                           scenario_id, 
                                           ego_content[timestamp_cur]['yaml'])

        return delay_params
    
    def get_visible_objects(self, cav_content, cav_params, timestamp:str, thr:int=1, 
                            scenario_id:int=None):
        """
        Obtain the visible objects of the agent at given timestamp.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.
        
        cav_params: dict
            The agent's annotations loaded from a yaml file.

        timestamp: str
            The specific timestamp, e.g., "000068".

        thr: int
            If an object is hit by at least `thr` laser beams, the object
            is regarded as visible.

        scenario_id : int
            It cannot be None if `keep_key_objects_anyway` is True.

        keep_key_objects_anyway : bool
            If set, the key objects will be kept as gt even if they are not 
            observed by any agents.

        Returns
        -------
        visible_objects: dict
            The agent's visible objects.
        """
        # the dataset is OPV2V or V2XSet or V2V4Real 
        if cav_content[timestamp]["view"] is None:
            visible_objects = cav_params["vehicles"]
        # the dataset is Multi-V2X
        else:
            hit_objects = load_yaml(cav_content[timestamp]["view"])["visible_objects"]
            visible_object_ids = [el['object_id'] for el in hit_objects 
                                    if el["visible_points"] >= thr]
            visible_objects = {obj_id:obj_info for obj_id,obj_info in cav_params["objects"].items()
                               if obj_id in visible_object_ids}
            self.correct_object_center(visible_objects)
        return visible_objects
    
    @staticmethod
    def correct_object_center(objects_info):
        """To fix the CARLA bug: cycle and motor's bounding box's center's y value
        exceed zero greatly, e.g., 125 meters, which is wrong!
        To repair this, once bbox's center's y value exceed 0.5 meters, we reset 
        it to zero.
        """
        thr = 0.5  # meters
        for obj_id in objects_info.keys():
            if objects_info[obj_id]['center'][1] > thr:
                objects_info[obj_id]['center'][1] = 0.0
    
    def get_key_object_ids_in_scenario(self, scenario_id:int):
        """Get key objects' ids for given scenario.
        Now, occludees are taken as key objects.

        Parameters
        ----------
        scenario_id : int

        Returns
        -------
        key_object_ids : list[int]
        """
        key_object_ids = self.database_manager.get_key_object_ids_in_scenario(scenario_id)
        return key_object_ids[:]
    
    def get_key_objects_for_agent(self, object_dict, scenario_id, cur_agent_yaml_path:str):
        """
        Get key objects' info for agent.
        """
        if not self.params['preprocess']['keep_key_objects_anyway']:
            return
        
        key_objects = self.database_manager.get_key_objects_for_agent(scenario_id, cur_agent_yaml_path)
        object_dict.update(key_objects)
        
    
    def mask_ego_points(self, points, scenario_id, ego_id):
        """
        Remove points of the ego vehicle itself.

        Parameters
        ----------
        points : np.ndarray
            Lidar points under lidar sensor coordinate system.

        scenario_id : str
            The scenario the ego lie in

        ego_id : int
            The ego's id.

        Returns
        -------
        points : np.ndarray
            Filtered lidar points.

        """
        ego_extent = self.database_manager.get_agent_extent(scenario_id, ego_id)
        # there is no need to mask points for rsu
        if ego_extent is None:
            return points
        
        if self.dataset_format == 'opv2v' or self.dataset_format == 'v2v4real':
            mask = (points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) \
                & (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1)
        elif self.dataset_format == 'test' or self.dataset_format == 'multi-v2x':
            mask = (points[:, 0] >= -ego_extent[0]) & (points[:, 0] <= ego_extent[0]) &\
                    (points[:, 1] >= -ego_extent[1]) & (points[:, 1] <= ego_extent[1])
        points = points[np.logical_not(mask)]

        return points

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask


    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for early and late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def visualize_result(self, pred_box_tensor,
                         gt_tensor,
                         pcd,
                         show_vis,
                         save_path,
                         dataset=None):
        # visualize the model output
        self.post_processor.visualize(pred_box_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis,
                                      save_path,
                                      dataset=dataset)
        

    @staticmethod
    def get_empty_test_metric_dict():
        """
        Returns a test metric template.
        """
        return {
            "num_of_conn_list": [],   # num of connected agents for each sample
            "comm_vol_list": [],      # total communication volume in bytes for each sample
        }
    
    @staticmethod
    def get_test_metric_dict_instance(num_of_conn:int=0, comm_vol:float=0):
        """
        Instantialize a test metric dict (for one sample).
        The test metric dict is expected to be used only when inference
        with batch size = 1.

        Parameters
        ----------
        num_of_conn : int
            The number of connected agents (including ego) in that sample.
        comm_vol : float
            The total communication volume in bytes for ego in that sample.

        Returns
        -------
        test_metric_dict : dict
            E.g.: {
                "num_of_conn_list": [2],
                "comm_vol_list": [2.2]
            }

        """
        return {
            "num_of_conn_list": [num_of_conn],
            "comm_vol_list": [comm_vol]
        }

    
    @staticmethod
    def append_test_metric_dict(test_metric_dict, aother):
        """
        Update 

        Parameters
        ----------
        test_metric_dict : dict

        aother : dict
            another test metric dict
        """
        test_metric_dict['num_of_conn_list'] += aother['num_of_conn_list']
        test_metric_dict['comm_vol_list'] += aother['comm_vol_list']

    @staticmethod
    def print_test_metric_dict(test_metric_dict):
        # connected agents
        num_of_conn_list = np.array(test_metric_dict['num_of_conn_list'])
        print("#agent: max={0}, min={1}, avg.={2:.3f}".format(
            np.max(num_of_conn_list), np.min(num_of_conn_list), np.mean(num_of_conn_list)
        ))

        # communication volume
        comm_vol_list = np.array(test_metric_dict['comm_vol_list'])
        one_MB = 1048576  # bytes
        one_KB = 1024     # bytes
        print("comm_vol(KB): [min,max,avg.]={0:.3f}, {1:.3f}, {2:.3f}".format(
            np.min(comm_vol_list)/one_KB,
            np.max(comm_vol_list)/one_KB,
            np.mean(comm_vol_list)/one_KB
        ))

        # mask = num_of_conn_list>1  # in case of single car
        # if np.sum(mask) > 0:
        #     comm_vol_list = comm_vol_list[mask]
        #     print("comm_vol(KB, masked by conn): max={0:.3f}, min={1:.3f}, avg.={2:.3f}".format(
        #         np.max(comm_vol_list)/one_KB, 
        #         np.min(comm_vol_list)/one_KB, 
        #         np.mean(comm_vol_list)/one_KB
        #     ))
