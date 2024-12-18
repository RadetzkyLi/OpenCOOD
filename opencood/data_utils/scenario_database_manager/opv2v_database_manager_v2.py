#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   opv2v_database_manager_v2.py
@Date    :   2024-02-25
@Author  :   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Database manager for Multi-V2X dataset.
'''

import os
import sys
import json
from collections import OrderedDict

from opencood.data_utils.scenario_database_manager.opv2v_database_manager_4test import Opv2vDatabaseManager4Test
from opencood.data_utils.scenario_database_manager.manager_utils import ManagerUtility
from opencood.hypes_yaml.yaml_utils import load_yaml


class Opv2vDatabaseManagerV2(Opv2vDatabaseManager4Test):
    """
    The manager for customized Multi-V2X dataset which has 
    a similar format as OPV2V and V2XSet.
    
    """
    def __init__(self, params, partname='train') -> None:
        super().__init__(params, partname)

        # structure: [{rsu_id: rsu_pose}, ...]
        self.rsu_pose_list = self.get_rsu_pose_list_in_dataset()
        # structure: {scenario_id: agent}

    @staticmethod
    def load_camera_files(agent_dir: str, timestamp: str, agent_type:str):
        """Retrive camera files for agent
        
        Parameters
        ----------
        agent_dir: str
            The full path of agent
        timestamp: str
            Current timestamp
        agent_type: str
            Can be "cav" or "rsu".
        
        Returns
        -------
        file_list: list
            The list containing all camera png file paths.
        """
        if agent_type == "cav":
            camera_front = os.path.join(agent_dir, timestamp + "_cameraFront.jpg")
            camera_rear = os.path.join(agent_dir, timestamp + "_cameraRear.jpg")
            camera_left = os.path.join(agent_dir, timestamp + "_cameraLeft.jpg")
            camera_right = os.path.join(agent_dir, timestamp + "_cameraRight.jpg")
            file_list = [camera_front, camera_rear, camera_left, camera_right]
        elif agent_type == "rsu":
            camera_forward = os.path.join(agent_dir, timestamp + "_cameraForward.jpg")
            camera_backward = os.path.join(agent_dir, timestamp + "_cameraBackward.jpg")
            file_list = [camera_forward, camera_backward]
        else:
            raise ValueError("Unknown agent_type '%s'"%agent_type)
        return file_list
    
    @staticmethod
    def get_timestamps_in_agent(agent_dir:str, partname:str):
        """Get the timestamps from agent_dir based on given train_ratio
        for training or test set. 
        
        Parameters
        ----------
        agent_dir: str
            Dir of agent data
        
        train_ratio: float
            The proceding part of all timestamps is taken as training set, and 
            the last part as test.
        
        partname: str
            One of "train", "val" and "test".

        Returns
        -------
        timestamps: list
            Sorted timestamps, e.g.: ["000068", "000070", "000072" ...]
        """
        timestamps = [x.split(".pcd")[0] for x in os.listdir(agent_dir)
                    if x.endswith(".pcd")]
        timestamps.sort()

        train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
        N = len(timestamps)
        n_train = int(train_ratio * N)
        n_val = int(val_ratio * N)
        n_test = N-n_train-n_val

        # for debug
        # n_train = 5
        # n_val = 2
        # n_test = 2

        if partname == 'train':
            timestamps = timestamps[0:n_train]
        elif partname == 'val':
            timestamps = timestamps[n_train:(n_train+n_val)]
        elif partname == 'test':
            timestamps = timestamps[(n_train+n_val):(n_train+n_val+n_test)]
        else:
            raise ValueError("The `partname` must be one of 'train', 'val' and 'test'")
        return timestamps
    
    def load_agent_config(self):
        """
        load agent config
        """
        pr = self.params['pr_setting']['value']
        path = self.params['pr_setting']['path']
        # structure: [{"scenario_name":xxx, "cav_list": [], "rsu_list": [], "ego_list": []}]
        with open(path, 'r') as f:
            content = f.read()
            data = json.loads(content)[str(pr)]
        self.agent_config_list = data

    def initialize_database_from_agent_config(self, root):
        """
        """
        scenario_database = OrderedDict()
        sample_id_to_index = OrderedDict()
        num_total_samples = 0
        scenario_name_list = []  # multiple scenarios can share a same name
        # structure: [{cav_id: [length/2, width/2, height/2]}]
        cav_extent_list = []

        # go through all agent config, each config as a scenario
        for scenario_id,agent_config in enumerate(self.agent_config_list):
            scenario_name = agent_config['scenario_name']
            scenario_dir = os.path.join(root, scenario_name)
    
            # structure: {agent_id: timestamp: {"yaml":xxx, "lidar": xxx, "camera0": xxx, "view": xxx}}
            scenario_data = OrderedDict()
            scenario_name_list.append(scenario_name)

            # CAV's extent
            cav_extent_list.append(ManagerUtility.get_cav_extent_in_scenario(scenario_dir))

            # go through all agents
            for agent_id in agent_config["cav_list"] + agent_config["rsu_list"]:
                agent_type = "cav" if agent_id in agent_config["cav_list"] else "rsu"
                agent_dir = os.path.join(scenario_dir, self.get_agent_name_from_type(agent_type, agent_id))

                # go through all timestamps
                scenario_data[agent_id] = OrderedDict()
                for timestamp in self.get_timestamps_in_agent(agent_dir, self.partname):
                    # get file paths for point cloud, camera and annotation
                    yaml_file = os.path.join(agent_dir, timestamp + ".yaml")
                    lidar_file = os.path.join(agent_dir, timestamp + '.pcd')
                    camera_files = self.load_camera_files(agent_dir, timestamp, agent_type)
                    view_file = os.path.join(agent_dir, timestamp + "_view.yaml")
                    scenario_data[agent_id][timestamp] = {
                        "yaml": yaml_file,
                        "lidar": lidar_file,
                        "camera0": camera_files,
                        "view": view_file
                    }
                
                    # update dataset id mapping
                    if agent_id in agent_config["ego_list"]:
                        key = self.get_sample_key_from_ids(
                            scenario_id,
                            agent_id,
                            timestamp
                        )
                        sample_id_to_index[key] = num_total_samples
                        num_total_samples += 1
            
            scenario_database[scenario_id] = scenario_data

        # check
        assert len(scenario_database) == len(self.agent_config_list)
        assert len(scenario_name_list) == len(self.agent_config_list)
        assert len(cav_extent_list) == len(self.agent_config_list)
            
        self.scenario_database = scenario_database
        self.scenario_name_list = scenario_name_list
        self.sample_id_to_index = sample_id_to_index
        self.cav_extent_list = cav_extent_list

    def initialize_database(self, root: str):
        """
        Load agent config list and then initialize
        """
        # load agent config
        self.load_agent_config()

        # init
        self.initialize_database_from_agent_config(root)


    def get_rsu_pose_list_in_dataset(self):
        """
        The pose of rsu is fixed, preload to save time.
        This should be called after `self.agent_config_list` is constructed.

        Parameters
        ----------

        Returns
        -------
        rsu_pose_list : list|None
            Each element denotes all RSUs' poses in a scenario. e.g.:
            {
                rsu_id: rsu_pose
            }
        """
        if self.agent_config_list is None:
            print("Warning: rsu pose only can be gotten after initializing agent_config_list")
            return None
        
        rsu_pose_list = []
        for scenario_id,agent_config in enumerate(self.agent_config_list):
            id_to_pose_dict = {}  # current scenario's rsus' pose
            for rsu_id in agent_config["rsu_list"]:
                timestamp = list(self.scenario_database[scenario_id][rsu_id].keys())[0]
                yaml_path = self.scenario_database[scenario_id][rsu_id][timestamp]["yaml"]
                id_to_pose_dict[rsu_id] = load_yaml(yaml_path)["lidar_pose"]
            rsu_pose_list.append(id_to_pose_dict)

        return rsu_pose_list

    def filter_agents_by_range(self, scenario_id, ego_id, timestamp:str, all_cav_ids, all_rsu_ids):
        """
        Filter by range.
        Cache the RSU's lidar pose
        """
        # get ego's pose
        yaml_path = self.scenario_database[scenario_id][ego_id][timestamp]['yaml']
        ego_lidar_pose = load_yaml(yaml_path)["lidar_pose"]
        
        cav_id_list,rsu_id_list = [],[]

        # 1) handle RSU
        # filter by timestamp
        all_rsu_ids = [rsu_id for rsu_id in all_rsu_ids
                       if timestamp in self.scenario_database[scenario_id][rsu_id]]
        for rsu_id in all_rsu_ids:
            lidar_pose = self.rsu_pose_list[scenario_id][rsu_id]
            d = self.calc_dist_between_pose(ego_lidar_pose, lidar_pose)
            if d <= self.comm_range:
                rsu_id_list.append(rsu_id)

        # 2) handle CAV
        all_cav_ids = [cav_id for cav_id in all_cav_ids 
                       if timestamp in self.scenario_database[scenario_id][cav_id]]
        
        for cav_id in all_cav_ids:
            yaml_path = self.scenario_database[scenario_id][cav_id][timestamp]["yaml"]
            lidar_pose = load_yaml(yaml_path)["lidar_pose"]
            d = self.calc_dist_between_pose(ego_lidar_pose, lidar_pose)
            if d <= self.comm_range:
                cav_id_list.append(cav_id)

        return cav_id_list,rsu_id_list