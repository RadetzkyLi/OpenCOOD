#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   manager_utils.py
@Date    :   2024-02-25
@Author  :   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Common utils for database manager
'''

import os

from opencood.hypes_yaml.yaml_utils import load_yaml



class ManagerUtility:
    """
    Some common utility for database manager
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_agents_in_scenario(scenario_root):
        """Get CAV and RSU in a scenario. """
        cav_list = [int(x.split("cav_")[1]) for 
                    x in os.listdir(scenario_root) if x.startswith("cav")]
        rsu_list = [int(x.split("rsu_")[1]) for 
                    x in os.listdir(scenario_root) if x.startswith("rsu")]
        return cav_list,rsu_list

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
            The list containing all camera png file paths.
        """
        if agent_type == "cav":
            camera_front = os.path.join(agent_dir, timestamp + "_cameraFront.png")
            camera_rear = os.path.join(agent_dir, timestamp + "_cameraRear.png")
            camera_left = os.path.join(agent_dir, timestamp + "_cameraLeft.png")
            camera_right = os.path.join(agent_dir, timestamp + "_cameraRight.png")
            file_list = [camera_front, camera_rear, camera_left, camera_right]
        elif agent_type == "rsu":
            camera_forward = os.path.join(agent_dir, timestamp + "_cameraForward.png")
            camera_backward = os.path.join(agent_dir, timestamp + "_cameraBackward.png")
            file_list = [camera_forward, camera_backward]
        else:
            raise ValueError("Unknown agent_type '%s'"%agent_type)
        return file_list

    @staticmethod
    def get_cav_extent_in_scenario(scenario_root):
        """Get extent of cav in given scenario.

        Parameters
        ----------
        scenario_root: str
            The root of a scenario data.

        target_cav_id_list : str
            The CAVs we want to know extent. If None, find all 

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

            # all CAVs' extent have been found
            if len(cav_extent) == len(cav_list):
                break

        return cav_extent
    
    @staticmethod
    def get_key_object_ids_in_scenario(scenario_root:str):
        """Get key objects' ids in given scenario.
        The occludees are taken as key objects.
        
        Parameters
        ----------
        scenario_root : str

        Returns
        -------
        key_object_ids : list[int]
        """
        yaml_path = os.path.join(scenario_root, "data_protocal.yaml")
        scene_desc = load_yaml(yaml_path)["additional"]["scene_desc"]
        occludee_ids = [ele[1] for ele in scene_desc["occlusion_pairs"]]

        key_object_ids = []
        for occludee_id in set(occludee_ids):
            actor_id = scene_desc["id_mapping"].get(occludee_id, None)
            # the occludee is not static
            if actor_id is not None:
                key_object_ids.append(actor_id)
        
        return key_object_ids
