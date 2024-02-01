#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   opv2v_database_manager_v2.py
@Date    :   2024-01-31
@Author  :   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Manage dataset in Multi-V2X format (a variant of OPV2V)
'''

import os
import sys
import numpy as np
from collections import OrderedDict

from opencood.data_utils.scenario_databse_manager.base_database_manager import BaseDatabaseManager
from opencood.hypes_yaml.yaml_utils import load_yaml


class ManagerUtility:
    """
    Some common utility for database manager
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_agents_in_scenario(scenario_root):
        cav_list = [int(x.split("cav_")[1]) for 
                    x in os.listdir(scenario_root) if x.startswith("cav")]
        rsu_list = [int(x.split("rsu_")[1]) for 
                    x in os.listdir(scenario_root) if x.startswith("rsu")]
        return cav_list,rsu_list

    @staticmethod
    def get_timestamps_in_agent(agent_dir):
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
    def load_camera_files(agent_dir, timestamp, agent_type):
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
            The list containing all camera jpg file paths.
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
    def get_sample_key_from_ids(scenario_id, agent_name, timestamp):
        """
        
        """
        key = "%d\t%s\t%s"%(scenario_id, agent_name, timestamp)
        return key

    @staticmethod
    def decode_sample_key(key):
        """
        
        Parameters
        ----------
        key : str

        Returns
        -------
        scenario_id : int

        agent_name : str

        timestamp : str
        """
        arr = key.split("\t")
        scenario_id = int(arr[0])
        agent_name = arr[1]
        timestamp = arr[2]
        return scenario_id,agent_name,timestamp
        

    @staticmethod
    def get_cav_extent_in_scenario(scenario_root):
        """Get extent of cav in given scenario.

        Parameters
        ----------
        scenario_root: str
            The root of a scenario data.

        Returns
        -------
        cav_extent: dict
            E.g.: {
                110: [2.5, 1.1, 0.8]  # half size in meters in x, y and z direction
            }
        """
        cav_list,_ = ManagerUtility.get_agents_in_scenario(scenario_root)
        fnames = list(os.listdir(os.path.join(scenario_root, "map")))
        fnames.sort()
        cav_extent = {}
        for fname in fnames:
            file_path = os.path.join(scenario_root, "map", fname)
            object_info = load_yaml(file_path)["objects"]
            for cav in cav_list:
                if cav not in cav_extent and cav in object_info:
                    cav_extent[cav] = object_info[cav]["extent"]
            if len(cav_extent) == len(cav_list):
                break
        return cav_extent

class Opv2vDatabaseManagerV2(BaseDatabaseManager):
    """
    Manager dataset in Multi-V2X format.


    """
    def __init__(self, params, partname='train') -> None:
        super().__init__(params, partname)

        self.root = params["root_dir"]

        # some wild settings
        wild_setting = params.get("wild_setting", {})
        self.sampling_frequency = wild_setting.get("sampling_frequency", 10)  # Hz
        self.comm_type = wild_setting.get("comm_type", "vx2")                 # 'v2v', 'v2i' supported
        self.comm_range = wild_setting.get("comm_range", 70)                  # meters


        # key data structures
        # structure: {scenario_id: agent_name: {timestamp: {xxx}, }}
        self.scenario_database = None
        # structure: {sample_key: index}
        self.sample_id_mapping = None
        # structure: [{"scenario_name":xxx, "cav_list": [], "rsu_list": [], "ego_list": []}]
        self.agent_config_list = None
        # structure: [{cav_id: [length/2, width/2, height/2]}]
        self.cav_extent_list = None


    def get_all_agent_config(self):
        """
        Returns agent config for all scenarios.
        """
        return None
        
    
    @staticmethod
    def get_agent_name_from_id(agent_id:int, agent_type:str):
        return "{0}_{1}".format(agent_type, agent_id)
    
    @staticmethod
    def get_agent_id_from_name(agent_name):
        return int(agent_name.split("_")[1])
    
    @staticmethod
    def get_agent_type_from_name(agent_name):
        """
        Only support "cav" and "rsu"
        """
        if agent_name.startswith("cav_"):
            return "cav"
        elif agent_name.startswith("rsu_"):
            return "rsu"
        else:
            raise ValueError("Unexpected agent name: {0}".format(agent_name))
        
    def get_agent_type_from_id(self, scenario_id:int, agent_id:int):
        """
        Returns None if the given agent doesn't belong to either 
        CAV nor RSU of given scenario
        """
        cur_agent_config = self.agent_config_list[scenario_id]
        if agent_id in cur_agent_config["cav_list"]:
            return "cav"
        elif agent_id in cur_agent_config["rsu_list"]:
            return "rsu"
        else:
            return None
        
    def get_agent_name(self, scenario_id:int, agent_id:int):
        agent_type = self.get_agent_type_from_id(scenario_id, agent_id)
        agent_name = self.get_agent_name_from_id(agent_id, agent_type)
        return agent_name
        
    @staticmethod
    def calc_dist_between_pose(pose, other):
        d = np.sqrt(
            (pose[0] - other[0])**2 +
            (pose[1] - other[1])**2 +
            (pose[2] - other[2])**2
        )
        return d

    def initialize_database(self, root):
        """
        """
        scenario_database = OrderedDict()
        sample_id_mapping = OrderedDict()
        num_total_samples = 0
        all_agent_config = self.get_all_agent_config()
        scenario_name_list = []  # multiple scenarios can share a same name

        for town_agent_config in all_agent_config:
            scenario_name = town_agent_config["town_name"]
            scenario_dir = os.path.join(root, scenario_name)
    
            # record data mete info
            scenario_data = OrderedDict()
            # go through all scenarios in a town
            for agent_config in town_agent_config["agent_config"]:
                scenario_id = len(scenario_name_list)
                scenario_name_list.append(scenario_name)
                scenario_data[scenario_id] = OrderedDict()

                # go through all agents
                for agent_id in agent_config["cav_list"] + agent_config["rsu_lsit"]:
                    agent_type = "cav" if agent_id in agent_config["cav_list"] else "rsu"
                    agent_name = self.get_agent_name_from_id(agent_id)
                    agent_dir = os.path.join(scenario_dir, self.get_agent_name_from_id(agent_id, agent_type))

                    # go through all timestamps
                    scenario_data[scenario_id][agent_name] = OrderedDict()
                    for timestamp in ManagerUtility.get_timestamps_in_agent(agent_dir):
                        # get file paths for point cloud, camera and annotation
                        yaml_file = os.path.join(agent_dir, timestamp + ".yaml")
                        lidar_file = os.path.join(agent_dir, timestamp + '.pcd')
                        camera_files = ManagerUtility.load_camera_files(agent_dir, timestamp, agent_type)
                        view_file = os.path.join(agent_dir, timestamp + "_view.yaml")
                        scenario_data[agent_name][timestamp] = {
                            "yaml": yaml_file,
                            "lidar": lidar_file,
                            "camera0": camera_files,
                            "view": view_file
                        }
                    
                        # update dataset id mapping
                        if agent_id in agent_config["ego_list"]:
                            key = ManagerUtility.get_sample_key_from_ids(
                                scenario_id,
                                agent_name,
                                timestamp
                            )
                            sample_id_mapping[key] = num_total_samples
                            num_total_samples += 1
            
            scenario_database[scenario_id] = scenario_data
            
        self.scenario_database = scenario_database
        self.sample_id_mapping = sample_id_mapping

    def initialize_database_for_test_scenes(self, root):
        """Initialize test scenes for testing scenarios.
        
        """
        # structure: {scenario_id: agent_name: {timestamp: {xxx}, }}
        scenario_database = OrderedDict()
        # structure: {sample_key: sample_index}
        sample_id_mapping = OrderedDict()
        num_total_samples = 0
        scenario_name_list = []  # multiple scenarios can share a same name
        # structure: [{"scenario_name":xxx, "cav_list": [], "rsu_list": [], "ego_list": []}]
        agent_config_list = []
        # structure: [{cav_id: [length/2, width/2, height/2]}]
        cav_extent_list = []

        for scenario_id,scenario_name in enumerate(os.listdir(root)):
            scenario_data = OrderedDict()
            scenario_name_list.append(scenario_name)

            # find ego id
            scenario_dir = os.path.join(root, scenario_name)
            yaml_file = os.path.join(scenario_dir, "data_protocal.yaml")
            additional = load_yaml(yaml_file)["additional"]
            id_mapping = additional["scene_desc"]["id_mapping"]
            ego_id = id_mapping[additional["scene_desc"]["ego_id"]]

            # find all agents
            cav_list,rsu_list = ManagerUtility.get_agents_in_scenario(scenario_dir)
            agent_config_list.append({
                "scenario_name": scenario_name,
                "cav_list": cav_list,
                "rsu_list": rsu_list,
                "ego_list": [ego_id]
            })
            cav_extent_list.append(ManagerUtility.get_cav_extent_in_scenario(scenario_dir))

            # go through all agents
            for agent_name in os.listdir(scenario_dir):
                if not (agent_name.startswith("cav") or agent_name.startswith("rsu")):
                    continue
                agent_dir = os.path.join(scenario_dir, agent_name)
                agent_type = self.get_agent_type_from_name(agent_name)

                # go through all timestamps
                for timestamp in ManagerUtility.get_timestamps_in_agent(agent_dir):
                    # get file paths for point cloud, camera and annotation
                    yaml_file = os.path.join(agent_dir, timestamp + ".yaml")
                    lidar_file = os.path.join(agent_dir, timestamp + '.pcd')
                    camera_files = ManagerUtility.load_camera_files(agent_dir, timestamp, agent_type)
                    view_file = os.path.join(agent_dir, timestamp + "_view.yaml")
                    scenario_data[agent_name][timestamp] = {
                        "yaml": yaml_file,
                        "lidar": lidar_file,
                        "camera0": camera_files,
                        "view": view_file
                    }
                
                    # update dataset id mapping
                    if self.get_agent_id_from_name(agent_name) == ego_id:
                        key = ManagerUtility.get_sample_key_from_ids(
                            scenario_id,
                            agent_name,
                            timestamp
                        )
                        sample_id_mapping[key] = num_total_samples
                        num_total_samples += 1

            scenario_database[scenario_id] = scenario_data

        self.scenario_database = scenario_database
        self.sample_id_mapping = sample_id_mapping
        self.agent_config_list = agent_config_list
        self.cav_extent_list = cav_extent_list

    def get_number_of_total_samples(self):
        """
        Returns sample length
        """
        return len(self.sample_id_mapping)

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
        key = self.sample_id_mapping.keys()[index]
        return ManagerUtility.decode_sample_key(key)

    def calc_timestamp_with_gap(self, timestamp:float, time_gap:float):
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
        delta_timestamp_int = int(time_gap/self.sampling_frequency)

        target_timestamp_int = timestamp_int + delta_timestamp_int
        target_timestamp_int = max(target_timestamp_int, 0)
        # convert to string
        target_timestamp = "%06d"%target_timestamp_int
        return target_timestamp
    
    def get_connected_agents_for_timestamp(self, scenario_id, agent_name, timestamp):
        """Returns connected agent list for given timestamp.
        
        Parameters
        ----------
        scenario_id : int

        agent_name : str

        timestamp : str

        Returns
        -------
        conn_list : list[int]
            Ids of agents that the given agent can connect to at given
            timestamp. The ego is not included.
        """
        cur_agent_config = self.agent_config_list[scenario_id]
        scenario_name = cur_agent_config["scenario_name"]
        agent_type = self.get_agent_type_from_name(agent_name)

        # get all agents
        if self.comm_type == "v2v":
            if agent_type == "rsu":
                return []
            all_agents = cur_agent_config["cav_list"]
        elif self.comm_type == "v2i":
            if agent_type == 'rsu':
                all_agents = cur_agent_config["cav_list"]
            elif agent_type == "cav":
                all_agents = cur_agent_config["rsu_list"]
            else:
                raise ValueError("Unknown agent type: {0}".format(agent_type))
        elif self.comm_type == 'v2x':
            all_agents = cur_agent_config["cav_list"] + cur_agent_config["rsu_list"]
        else:
            raise ValueError("Unexpected comm_type : {0}".format(self.comm_type))
        
        # get ego's pose
        fname = timestamp + ".yaml"
        yaml_path = os.path.join(self.root, scenario_name, agent_name, fname)
        cur_ego_lidar_pose = load_yaml(yaml_path)["lidar_pose"]
        
        conn_list = []
        # filter agents by range
        for agent_id in all_agents:
            agent_type = "cav" if agent_id in cur_agent_config["cav_list"] else "rsu"
            yaml_path = os.path.join(self.root, scenario_name, 
                                     self.get_agent_name_from_id(agent_id, agent_type),
                                     fname)
            if not os.path.exists(yaml_path):
                continue

            lidar_pose = load_yaml(yaml_path)["lidar_pose"]
            d = self.calc_dist_between_pose(cur_ego_lidar_pose, lidar_pose)
            if d <= self.comm_range:
                conn_list.append(agent_id)

        return conn_list
    
    def get_connected_agents_for_sample_index(self, index:int):
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        conn_list : list[int]
            The connected agents for given sample.
        """
        scenario_id,agent_name,timestamp = self.get_ids_from_sample_index(index)
        return self.get_connected_agents_for_timestamp(scenario_id, agent_name, timestamp)
    
    def is_agent_has_timestamp(self, scenario_id:int, agent_id:int, timestamp:str):
        """
        Returns True if target agent has given timestamp
        """
        agent_type = self.get_agent_type_from_id(scenario_id, agent_id)
        agent_name = self.get_agent_name_from_id(agent_id, agent_type)
        key = ManagerUtility.get_sample_key_from_ids(scenario_id, agent_name, timestamp)

        if key in self.sample_id_mapping:
            return True
        return False
    
    def get_agent_extent(self, scenario_id:int, agent_id:int):
        """
        Returns None if the given agent out range of given scenario.
        """
        extent = self.cav_extent_list[scenario_id].get(agent_id, None)
        return extent