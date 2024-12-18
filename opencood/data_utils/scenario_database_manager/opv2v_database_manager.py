#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   opv2v_database_manager.py
@Date    :   2024-01-31
@Author  :   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Manager scenario database for dataset
            in OPV2V format.
'''

import os
import sys
import logging
import numpy as np
import random
from collections import OrderedDict
from joblib import delayed,Parallel

from opencood.data_utils.scenario_database_manager.base_database_manager import BaseDatabaseManager
from opencood.hypes_yaml.yaml_utils import load_yaml


class Opv2vDatabaseManager(BaseDatabaseManager):
    """
    The manager can be used for OPV2V and V2XSet.
    
    """
    def __init__(self, params, partname='train') -> None:
        super().__init__(params, partname)

        self.root = params['root_dir']

        # maximum number of agents
        if 'train_params' not in params or\
                'max_cav' not in params['train_params']:
            self.max_agent = 7
        else:
            self.max_agent = params['train_params']['max_cav']
        
        # key data structures
        # structure: {scenario_id: agent_id: {timestamp: {xxx}, }}
        self.scenario_database = None
        # structure: {sample_key: index}
        self.sample_id_to_index = None
        # structure: [{"scenario_name":xxx, "cav_list": [], "rsu_list": [], "ego_list": []}]
        self.agent_config_list = None
        # structure: [[ids of key objects], ...]
        self.key_objects_list = []

        # Initialize database
        self.initialize_database(self.root)


    @staticmethod
    def load_camera_files(agent_dir:str, timestamp:str):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        agent_dir : str
            The full file path of current agent.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(agent_dir,
                                    timestamp + '_camera0.png')
        camera1_file = os.path.join(agent_dir,
                                    timestamp + '_camera1.png')
        camera2_file = os.path.join(agent_dir,
                                    timestamp + '_camera2.png')
        camera3_file = os.path.join(agent_dir,
                                    timestamp + '_camera3.png')
        if not os.path.exists(camera0_file):
            return []
        return [camera0_file, camera1_file, camera2_file, camera3_file]
    
    @staticmethod
    def get_agents_in_scenario(scenario_dir:str):
        """Get all agents' ids in scenario, including CAV and RSU"""
        agent_list = [int(x) for x in os.listdir(scenario_dir)
                      if os.path.isdir(
                          os.path.join(scenario_dir, x)
                      )]
        cav_id_list = [x for x in agent_list if x>=0]
        rsu_id_list = [x for x in agent_list if x<0]
        return cav_id_list,rsu_id_list
    
    @staticmethod
    def get_timestamps_in_agent(agent_dir:str):
        """Get the timestamps from agent_dir based on given train_ratio
        for training or test set. 
        
        Parameters
        ----------
        agent_dir: str
            Dir of agent data
        
        train_ratio: float

        Returns
        -------
        timestamps: list
            Sorted timestamps, e.g.: ["000068", "000070", "000072" ...]
        """
        timestamps = [x.split(".pcd")[0] for x in os.listdir(agent_dir)
                    if x.endswith(".pcd")]
        timestamps.sort()
        return timestamps
    
    @staticmethod
    def get_sample_key_from_ids(scenario_id:int, agent_id:int, timestamp:str):
        """Returns the unique sample identifier."""
        key = "%d\t%d\t%s"%(scenario_id, agent_id, timestamp)
        return key
    
    @staticmethod
    def decode_sample_key(key:str):
        """Returns a sample's scenario id, agent id and timestamp
        
        Parameters
        ----------
        key : str

        Returns
        -------
        scenario_id : int

        agent_id : int

        timestamp : str
        """
        arr = key.split("\t")
        return int(arr[0]), int(arr[1]), arr[2]
    
    @staticmethod
    def calc_dist_between_pose(pose, other):
        """
        Parameters
        ----------
        pose : list|array
            [x, y, z, roll, yaw, pitch]
        other : list|array
            Same as `pose`.

        Returns
        -------
        d : float   
            The distance in x-y plane
        """
        if isinstance(pose, list):
            d = np.sqrt(
                (pose[0] - other[0])**2 +
                (pose[1] - other[1])**2
            )
        else:
            d = np.sqrt(
                (pose[0, -1] - other[0, -1])**2 +
                (pose[1, -1] - other[1, -1])**2
            )
        return d
    
    def get_agent_name_from_id(self, scenario_id:int, agent_id:int):
        """
        """
        # For OPV2V and V2XSet, agent_name = str(agent_id)
        return str(agent_id)

    def get_agent_type_from_id(self, scenario_id:int, agent_id:int):
        """
        For OPV2V and V2XSet, negative agent id is always RSU.

        Parameters
        ----------
        agent_id : int

        Returns
        -------
        agent_type : str
            One of {"cav", "rsu"}
        """
        if agent_id < 0:
            return "rsu"
        return "cav"


    def initialize_database(self, root: str):
        """Get scenario database.
        
        """
        # structure: {scenario_id: agent_id: {timestamp: {xxx}, }}
        scenario_database = OrderedDict()
        # structure: {sample_key: sample_index}
        sample_id_to_index = OrderedDict()
        num_total_samples = 0
        scenario_name_list = []  # multiple scenarios can share a same name
        # structure: [{"scenario_name":xxx, "cav_list": [], "rsu_list": [], "ego_list": []}]
        agent_config_list = []

        for scenario_id,scenario_name in enumerate(sorted(os.listdir(root))):
            scenario_dir = os.path.join(root, scenario_name)

            # if scenario_id > 1:
            #     break

            scenario_data = OrderedDict()
            scenario_name_list.append(scenario_name)

            # Find ego id
            # we regard the agent with the minimum id as the ego
            cav_id_list,rsu_id_list = self.get_agents_in_scenario(scenario_dir)
            cav_id_list = sorted(cav_id_list)
            ego_id = cav_id_list[0]

            # record agent config
            agent_config_list.append({
                "scenario_name": scenario_name,
                "cav_list": cav_id_list,
                "rsu_list": rsu_id_list,
                "ego_list": [ego_id]
            })

            # go through all agents
            all_agents = cav_id_list + rsu_id_list
            for j,agent_id in enumerate(all_agents):
                if j > self.max_agent - 1:
                    logging.warning("Too many agents ({0})! Truncate to `max_agent`({1})".format(
                        len(all_agents), self.max_agent
                    ))
                    break
                agent_dir = os.path.join(scenario_dir, self.get_agent_name_from_id(scenario_id, agent_id))

                scenario_data[agent_id] = OrderedDict()
                # go through all timestamps
                for timestamp in self.get_timestamps_in_agent(agent_dir):
                    # get file paths for point cloud, camera and annotation
                    yaml_file = os.path.join(agent_dir, timestamp + ".yaml")
                    lidar_file = os.path.join(agent_dir, timestamp + '.pcd')
                    camera_files = self.load_camera_files(agent_dir, timestamp)
                    view_file = os.path.join(agent_dir, timestamp + "_view.yaml")
                    view_file = view_file if os.path.exists(view_file) else None
                    scenario_data[agent_id][timestamp] = {
                        "yaml": yaml_file,
                        "lidar": lidar_file,
                        "camera0": camera_files,
                        "view": view_file
                    }
                
                    # update dataset id mapping
                    if agent_id == ego_id:
                        key = self.get_sample_key_from_ids(
                            scenario_id,
                            agent_id,
                            timestamp
                        )
                        sample_id_to_index[key] = num_total_samples
                        num_total_samples += 1

            scenario_database[scenario_id] = scenario_data

        self.scenario_database = scenario_database
        self.sample_id_to_index = sample_id_to_index
        self.agent_config_list = agent_config_list

    def get_number_of_total_samples(self):
        """
        Returns sample length
        """
        return len(self.sample_id_to_index)

    def get_ids_from_sample_index(self, index:int):
        """
        Return sample key for given index

        Parameters
        ----------
        index : int
            The sample index

        Returns
        -------
        scenario_id : int

        agent_name : str
            E.g., "cav_119".

        timestamp : str
            E.g., "000088"
        """
        key = list(self.sample_id_to_index.keys())[index]
        return self.decode_sample_key(key)
    
    def calc_timestamp_with_gap(self, timestamp:str, time_gap:float):
        """Calculate the timestamp for given time gap. If time gap is
        positive, returns the later timestamp; if time gap is negative,
        returns the previous timestamp.

        Parameters
        ----------
        timestamp : str
            e.g., "000088".

        time_gap : float
            e.g., 0.5 seconds

        Returns
        -------
        target_timestamp : str
            e.g., "000093"
        """
        timestamp_int = int(timestamp)
        # quantilization
        delta_timestamp_int = int(time_gap * self.sampling_frequency)
        # For OPV2V, V2XSet and Multi-V2X, the data is stremed at 20Hz
        # but saved at 10Hz. So, we need to times 2 so that the 
        # minimum integer timestamp gap is 2.
        delta_timestamp_int = int(2 * delta_timestamp_int)

        target_timestamp_int = timestamp_int + delta_timestamp_int
        target_timestamp_int = max(target_timestamp_int, 0)
        # convert to string
        target_timestamp = "%06d"%target_timestamp_int
        return target_timestamp
    
    def filter_agents_by_range(self, scenario_id, ego_id, timestamp:str, all_cav_ids, all_rsu_ids):
        """
        
        """
        # get ego's pose
        yaml_path = self.scenario_database[scenario_id][ego_id][timestamp]['yaml']
        ego_lidar_pose = load_yaml(yaml_path)["lidar_pose"]

        cav_id_list,rsu_id_list = [],[]
        for agent_id in all_cav_ids+all_rsu_ids:
            if timestamp not in self.scenario_database[scenario_id][agent_id]:
                continue
            
            yaml_path = self.scenario_database[scenario_id][agent_id][timestamp]['yaml']
            lidar_pose = load_yaml(yaml_path)["lidar_pose"]
            
            d = self.calc_dist_between_pose(ego_lidar_pose, lidar_pose)
            if d <= self.comm_range:
                if agent_id in all_cav_ids:
                    cav_id_list.append(agent_id)
                else:
                    rsu_id_list.append(agent_id)

        return cav_id_list,rsu_id_list
    
    def get_connected_agents_for_timestamp(self, scenario_id:int, agent_id:int, timestamp:str):
        """Returns connected agent list for given timestamp.
        
        Parameters
        ----------
        scenario_id : int

        agent_id : int

        timestamp : str

        return_cav_

        Returns
        -------
        cav_id_list : list[int]
            Ids of CAVs that the given agent can connect to at given
            timestamp. The ego is excluded.

        rsu_id_list : list[int]
            Ids of RSUs that the given agent can connect to at given
            timestamp. The ego is excluded.
        """
        cur_agent_config = self.agent_config_list[scenario_id]
        agent_type = self.get_agent_type_from_id(scenario_id, agent_id)
        ego_id = agent_id

        # Get all agents: CAV and RSU
        all_cav_ids,all_rsu_ids = [],[]
        # v2v
        if self.comm_type == "v2v":
            if agent_type == "rsu":
                return []
            all_cav_ids = cur_agent_config["cav_list"]
        # v2i
        elif self.comm_type == "v2i":
            if agent_type == 'rsu':
                all_rsu_ids = cur_agent_config["cav_list"]
            elif agent_type == "cav":
                all_cav_ids = cur_agent_config["rsu_list"]
            else:
                raise ValueError("Unknown agent type: {0}".format(agent_type))
        # v2x
        elif self.comm_type == 'v2x':
            all_cav_ids = cur_agent_config["cav_list"] 
            all_rsu_ids = cur_agent_config["rsu_list"]
        # single 
        elif self.comm_type == 'none' or self.comm_type is None:
            return []
        else:
            raise ValueError("Unexpected comm_type : {0}".format(self.comm_type))
        
        # exclude ego
        all_cav_ids = list(set(all_cav_ids) - set([ego_id]))
        
        # filter agents by range
        cav_id_list,rsu_id_list = self.filter_agents_by_range(
            scenario_id, ego_id, timestamp, all_cav_ids, all_rsu_ids)

        return cav_id_list,rsu_id_list

    def get_connected_agents_for_timestamp_random(self, scenario_id:int, agent_id:int, timestamp:str):
        """Returns connected agent list for given timestamp.
        
        Parameters
        ----------
        scenario_id : int

        agent_id : int

        timestamp : str

        return_cav_

        Returns
        -------
        cav_id_list : list[int]
            Ids of CAVs that the given agent can connect to at given
            timestamp. The ego is excluded.

        rsu_id_list : list[int]
            Ids of RSUs that the given agent can connect to at given
            timestamp. The ego is excluded.
        """
        cur_agent_config = self.agent_config_list[scenario_id]
        agent_type = self.get_agent_type_from_id(scenario_id, agent_id)

        if agent_type == "rsu":
            random.shuffle(cur_agent_config["rsu_list"])
            for cur_id in cur_agent_config["rsu_list"]:
                if timestamp in self.scenario_database[scenario_id][cur_id]:
                    ego_id = cur_id
                    break
            else:
                assert False, "No valid agent_id."
        elif agent_type == "cav":
            random.shuffle(cur_agent_config["cav_list"])
            for cur_id in cur_agent_config["cav_list"]:
                if timestamp in self.scenario_database[scenario_id][cur_id]:
                    ego_id = cur_id
                    break
            else:
                assert False, "No valid agent_id."
        else:
            raise ValueError("Unknown agent type: {0}".format(agent_type))

        # Get all agents: CAV and RSU
        all_cav_ids, all_rsu_ids = [],[]
        # v2v
        if self.comm_type == "v2v":
            if agent_type == "rsu":
                return []
            all_cav_ids = cur_agent_config["cav_list"]
        # v2i
        elif self.comm_type == "v2i":
            if agent_type == 'rsu':
                all_rsu_ids = cur_agent_config["cav_list"]
            elif agent_type == "cav":
                all_cav_ids = cur_agent_config["rsu_list"]
            else:
                raise ValueError("Unknown agent type: {0}".format(agent_type))
        # v2x
        elif self.comm_type == 'v2x':
            all_cav_ids = cur_agent_config["cav_list"] 
            all_rsu_ids = cur_agent_config["rsu_list"]
        # single 
        elif self.comm_type == 'none' or self.comm_type is None:
            return []
        else:
            raise ValueError("Unexpected comm_type : {0}".format(self.comm_type))
        
        # exclude ego
        all_cav_ids = list(set(all_cav_ids) - set([ego_id]))
        
        # filter agents by range
        cav_id_list, rsu_id_list = self.filter_agents_by_range(scenario_id, ego_id, timestamp, all_cav_ids, all_rsu_ids)

        return cav_id_list, rsu_id_list, ego_id

    def get_connected_agents_for_sample_index(self, index:int):
        """
        Parameters
        ----------
        index : int
            Sample's integer index

        Returns
        -------
        cav_id_list : list[int]
            Ids of CAVs that the given agent can connect to at given
            timestamp. The ego is excluded.

        rsu_id_list : list[int]
            Ids of RSUs that the given agent can connect to at given
            timestamp. The ego is excluded.
        """
        scenario_id,agent_id,timestamp = self.get_ids_from_sample_index(index)
        return self.get_connected_agents_for_timestamp(scenario_id, agent_id, timestamp)
    
    def is_agent_has_timestamp(self, scenario_id:int, agent_id:int, timestamp:str):
        """
        Returns True if target agent has given timestamp
        """
        if timestamp in self.scenario_database[scenario_id][agent_id]:
            return True
        return False

    def is_rsu_agent(self, scenario_id:int, agent_id:int):
        """Returns True if given agent is RSU"""
        return agent_id < 0

    def get_agent_extent(self, scenario_id:int, agent_id:int):
        """
        Returns None if the given agent out range of given scenario.
        """
        # For OPV2V and V2XSet, the ego is always 'vehicle.lincoln.mkz_2017'.
        extent = [2.4508416652679443, 1.0641621351242065, 0.7553732395172119]
        
        if agent_id < 0:  # the agent is RSU
            return None
        return extent
    
    def get_key_object_ids_in_scenario(self, scenario_id:int):
        """
        Get key object ids for given scenario.
        """
        if len(self.key_objects_list) > 0:
            return self.key_objects_list[scenario_id]
        return []
    
    def get_key_objects_for_agent(self, scenario_id:int, cur_agent_yaml_path:str):
        """

        Returns
        -------
        key_objects : dict
            {object_id: object_info}
        """
        key_object_ids = self.get_key_object_ids_in_scenario(scenario_id)
        if len(key_object_ids) == 0:
            return {}
        
        fname = os.path.basename(cur_agent_yaml_path)
        scenario_dir = os.path.dirname(os.path.dirname(cur_agent_yaml_path))
        map_yaml_path = os.path.join(scenario_dir, 'map', fname)
        all_objects = load_yaml(map_yaml_path)['objects']
        key_objects = {_id: all_objects[_id] for _id in key_object_ids}
        
        return key_objects

