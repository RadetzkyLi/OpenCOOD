#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   opv2v_database_manager_4test.py
@Date    :   2024-01-31
@Author  :   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Manage dataset for Risky Testing Scenario
'''

import os
import sys
import numpy as np
from collections import OrderedDict
from joblib import Parallel,delayed

from opencood.data_utils.scenario_database_manager.opv2v_database_manager import Opv2vDatabaseManager
from opencood.data_utils.scenario_database_manager.manager_utils import ManagerUtility
from opencood.hypes_yaml.yaml_utils import load_yaml



class Opv2vDatabaseManager4Test(Opv2vDatabaseManager):
    """
    Manage generated testing scenarios, which is in 
    multi-v2x format with some variations.

    """
    def __init__(self, params, partname='train') -> None:
        # additional key data structures
        # structure: [{cav_id: [length/2, width/2, height/2]}]
        self.cav_extent_list = None
        # structure: [name, ...]
        self.scenario_name_list = None
        
        super().__init__(params, partname)

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
        
    @staticmethod
    def get_agent_name_from_type(agent_type:str, agent_id:int):
        """
        """
        assert agent_type in ['cav', 'rsu']

        agent_name = "%s_%d"%(agent_type, agent_id)
        return agent_name

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
        
    @staticmethod
    def get_agent_id_from_name(agent_name):
        """"""
        return int(agent_name.split("_")[1])
        
    def get_agent_name_from_id(self, scenario_id: int, agent_id: int):
        """
        In Multi-V2X format, agent_name = {agent_type}_{agent_id}
        """
        agent_type = self.get_agent_type_from_id(scenario_id, agent_id)
        if agent_type is None:
            raise ValueError("Agent {0} is not CAV nor RSU in scenario {1}"
                             .format(agent_id, scenario_id))
        return agent_type + "_" + str(agent_id)
    
    def get_scenario_name_from_id(self, scenario_id: int):
        """Get corresponding scenario name"""
        return self.scenario_name_list[scenario_id]

    @staticmethod
    def get_one_scenario_data(root:str, scenario_id, scenario_name:str):
        """
        Get one scenario's data.
        """
        scenario_dir = os.path.join(root, scenario_name)

        scenario_data = OrderedDict()
        
        # Find ego id
        # ego is determinded when generating testing scenarios
        yaml_file = os.path.join(scenario_dir, "data_protocal.yaml")
        additional = load_yaml(yaml_file)["additional"]
        id_mapping = additional["scene_desc"]["id_mapping"]
        ego_id = id_mapping[additional["scene_desc"]["ego_id"]]

        # find all agents
        cav_list,rsu_list = ManagerUtility.get_agents_in_scenario(scenario_dir)
        agent_config = {
            "scenario_name": scenario_name,
            "cav_list": cav_list,
            "rsu_list": rsu_list,
            "ego_list": [ego_id]
        }

        # go through all agents
        for agent_id in cav_list + rsu_list:
            agent_type = "cav" if agent_id in cav_list else "rsu"
            agent_dir = os.path.join(scenario_dir, 
                                    Opv2vDatabaseManager4Test.get_agent_name_from_type(
                                        agent_type, agent_id))

            scenario_data[agent_id] = OrderedDict()
            # go through all timestamps
            for timestamp in Opv2vDatabaseManager4Test.get_timestamps_in_agent(agent_dir):
                # get file paths for point cloud, camera and annotation
                yaml_file = os.path.join(agent_dir, timestamp + ".yaml")
                lidar_file = os.path.join(agent_dir, timestamp + '.pcd')
                camera_files = ManagerUtility.load_camera_files(agent_dir, timestamp, agent_type)
                view_file = os.path.join(agent_dir, timestamp + "_view.yaml")
                scenario_data[agent_id][timestamp] = {
                    "yaml": yaml_file,
                    "lidar": lidar_file,
                    "camera0": camera_files,
                    "view": view_file
                }
            
        output_dict = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "scenario_data": scenario_data,
            "key_objects": ManagerUtility.get_key_object_ids_in_scenario(scenario_dir),
            "agent_config": agent_config,
            "cav_extent": ManagerUtility.get_cav_extent_in_scenario(scenario_dir),
        }
        return output_dict
    
    def initialize_database_parallel(self, root:str, n_jobs:int=8):
        """
        Read the dataset in parallel manner.
        For testing scenarios, we need to open yaml to get
        key objects and so on for each scenario. Parallel will
        save much time than sequential opening. 
        """
        # structure: {scenario_id: agent_name: {timestamp: {xxx}, }}
        scenario_database = OrderedDict()
        # structure: {sample_key: sample_index}
        sample_id_to_index = OrderedDict()
        # structure: [name1, name2, ...]
        scenario_name_list = []  # multiple scenarios can share a same name
        # structure: [{"scenario_name":xxx, "cav_list": [], "rsu_list": [], "ego_list": []}]
        agent_config_list = []
        # structure: [{cav_id: [length/2, width/2, height/2]}]
        cav_extent_list = []
        # structure: [[key_object_id,]]
        key_objects_list = []

        res = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=0)(
            delayed(Opv2vDatabaseManager4Test.get_one_scenario_data)
            (root, scenario_id, scenario_name)
            for scenario_id,scenario_name in enumerate(sorted(os.listdir(root)))
        )

        # Merge the result
        # sort by scenario id
        res = sorted(res, key=lambda x:x["scenario_id"])

        num_of_total_samples = 0
        for scenario_data_dict in res:
            scenario_name_list.append(scenario_data_dict['scenario_name'])
            scenario_database[scenario_data_dict['scenario_id']] = \
                scenario_data_dict['scenario_data']
            key_objects_list.append(scenario_data_dict['key_objects'])
            agent_config_list.append(scenario_data_dict['agent_config'])
            cav_extent_list.append(scenario_data_dict['cav_extent'])

            # sample id to sample index
            for agent_id,agent_data_dict in scenario_data_dict['scenario_data'].items():
                for timestamp in sorted(list(agent_data_dict.keys())):
                    # update dataset id mapping
                    if agent_id in scenario_data_dict['agent_config']['ego_list']:
                        key = self.get_sample_key_from_ids(
                            scenario_data_dict['scenario_id'],
                            agent_id,
                            timestamp
                        )
                        sample_id_to_index[key] = num_of_total_samples
                        num_of_total_samples += 1

            # if num_of_total_samples > 10:  # for debug
            #     break

        # update to class attributes
        self.scenario_database = scenario_database
        self.sample_id_to_index = sample_id_to_index
        self.agent_config_list = agent_config_list
        self.cav_extent_list = cav_extent_list
        self.key_objects_list = key_objects_list
        self.scenario_name_list = scenario_name_list

    def initialize_database_sequential(self, root):
        """Initialize scenario database in sequential manner
          for testing scenarios.
        
        """
        # structure: {scenario_id: agent_name: {timestamp: {xxx}, }}
        scenario_database = OrderedDict()
        # structure: {sample_key: sample_index}
        sample_id_to_index = OrderedDict()
        num_total_samples = 0
        # structure: [name1, name2, ...]
        scenario_name_list = []  # multiple scenarios can share a same name
        # structure: [{"scenario_name":xxx, "cav_list": [], "rsu_list": [], "ego_list": []}]
        agent_config_list = []
        # structure: [{cav_id: [length/2, width/2, height/2]}]
        cav_extent_list = []
        # structure: [[key_object_id,]]
        key_objects_list = []

        for scenario_id,scenario_name in enumerate(sorted(os.listdir(root))):
            scenario_dir = os.path.join(root, scenario_name)

            # if scenario_id > 10:
            #     break

            scenario_data = OrderedDict()
            scenario_name_list.append(scenario_name)
            key_objects_list.append(ManagerUtility.get_key_object_ids_in_scenario(scenario_dir))

            # Find ego id
            # ego is determinded when generating testing scenarios
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
            for agent_id in cav_list + rsu_list:
                agent_type = "cav" if agent_id in cav_list else "rsu"
                agent_dir = os.path.join(scenario_dir, 
                                         self.get_agent_name_from_type(agent_type, agent_id))

                scenario_data[agent_id] = OrderedDict()
                # go through all timestamps
                for timestamp in self.get_timestamps_in_agent(agent_dir):
                    # get file paths for point cloud, camera and annotation
                    yaml_file = os.path.join(agent_dir, timestamp + ".yaml")
                    lidar_file = os.path.join(agent_dir, timestamp + '.pcd')
                    camera_files = ManagerUtility.load_camera_files(agent_dir, timestamp, agent_type)
                    view_file = os.path.join(agent_dir, timestamp + "_view.yaml")
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
        self.cav_extent_list = cav_extent_list
        self.key_objects_list = key_objects_list
        self.scenario_name_list = scenario_name_list

    def initialize_database(self, root: str):
        """
        Default to parallel
        """
        self.initialize_database_parallel(root)

    def get_number_of_total_samples(self):
        """
        Returns sample length
        """
        return len(self.sample_id_to_index)
    
    def is_rsu_agent(self, scenario_id:int, agent_id:int):
        """
        Parameters
        ----------
        scenario_id : int

        agent_id : int
            The agent's id.
        
        Returns
        -------
        flag : bool
            Returns True if the corresponding agent is RSU.
        """
        if agent_id in self.agent_config_list[scenario_id]["rsu_list"]:
            return True
        return False
    
    def get_agent_extent(self, scenario_id:int, agent_id:int):
        """
        Returns None if the given agent is RSU
        """
        if agent_id in self.agent_config_list[scenario_id]["rsu_list"]:
            extent = None
        elif agent_id in self.agent_config_list[scenario_id]["cav_list"]:
            extent = self.cav_extent_list[scenario_id][agent_id]
        else:
            raise ValueError("Found no agent {0} in scenario {1}"
                             .format(agent_id, scenario_id))
        return extent