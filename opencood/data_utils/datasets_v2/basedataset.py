# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modified by: Rongsong Li <rongsong.li@qq.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Basedataset class for all kinds of fusion.

The class BaseDataset works for Multi-V2X dataset.
"""

import os
import math
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import Dataset

import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


# ====================================================================
# - utility functions
# ====================================================================
def _get_agents_in_scenario(scenario_root):
    cav_list = [int(x.split("cav_")[1]) for 
                x in os.listdir(scenario_root) if x.startswith("cav")]
    rsu_list = [int(x.split("rsu_")[1]) for 
                x in os.listdir(scenario_root) if x.startswith("rsu")]
    return cav_list,rsu_list

def _get_agent_config_for_pr(path, pr):
    """Get the agent config for specific penetration rate.
    
    Parameters
    ----------
    path : str
        Path of the penetration rate config file.

    pr : float
        The target penetration rate

    Returns
    -------
    agent_config : list
        Each element in the list is a dict, e.g.: {
            "scenario_name": 'xxx',
            "agent_config": [{
                "seed": 43, 
                "cav_list": [248, 269, 325], 
                "ego_list": [269],
                }, 
            ...],
        },

    """
    return load_yaml(path)[str(pr)]

def _get_cav_extent_in_scenario(scenario_root):
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
    cav_list = [int(x.split("cav_")[1]) for 
                x in os.listdir(scenario_root) if x.startswith("cav")]
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

def _get_timestamps_in_agent(agent_dir, partname="train"):
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

    # set the following for debug
    # n_train = 1
    # n_val = 1
    # n_test = 1

    return timestamps

    if partname == 'train':
        timestamps = timestamps[0:n_train]
    elif partname == 'val':
        timestamps = timestamps[n_train:(n_train+n_val)]
    elif partname == 'test':
        timestamps = timestamps[(n_train+n_val):(n_train+n_val+n_test)]
    else:
        raise ValueError("The `part` must be one of 'train', 'val' and 'test'")
    return timestamps

def _load_camera_files(agent_dir, timestamp, agent_type="cav"):
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


def _get_scenario_database(root_dir, all_agent_config, partname='train'):
    """Get scenario database based on given config.
    
    Parameters
    ----------
    root_dir: str
        Dir of the dataset
    config: list
        The config of each scenario under specific penetration rate, e.g.: [
            {
                "scenario_name": 'xxx',
                "agent_config": [{
                    "seed": 43, 
                    "cav_list": [248, 269, 325], 
                    "rsu_list": [889], # added after this function
                    "ego_list": [269],
                    "len_list": [300], # frames of each ego, added after this function 
                }, 
                ...],
                "length": 3000 # total frames of the scenario, added after this function
            },
            ...
        ]
    partname: str
        One of "train", "val" and "test". 
    
    Returns
    -------
    scenario_database: dict
        Database similar to that of OpenCOOD, e.g.: {
            "Town01__2023": {
                269: {
                    "000068": {
                        "yaml": xxx,
                        "lidar": xxx,
                        "camera0": xxx,
                        "view": xxx
                    }
                }
            }
        }

    Notes
    -----
    1. `agent_config` will be modified here.
    """
    scenario_database = OrderedDict()
    total_length = 0
    for config in all_agent_config:
        scenario_folder = os.path.join(root_dir, config["scenario_name"])
        cav_list = set([])

        _,rsu_list = _get_agents_in_scenario(scenario_folder)
        
        # reform agent_config
        for cur_config in config["agent_config"]:
            cav_list = cav_list | set(cur_config["cav_list"])
            cur_config["len_list"] = []
            cur_config["rsu_list"] = rsu_list
            for cav in cur_config["ego_list"]:
                # retrive train/test data
                timestamps = _get_timestamps_in_agent(os.path.join(scenario_folder, "cav_%d"%cav),
                                                      partname=partname)
                n = len(timestamps)
                cur_config["len_list"].append(n)
                total_length += n
        config["length"] = total_length
        cav_list = list(cav_list)

        # record meta data
        data = OrderedDict()
        for agent in cav_list + rsu_list:
            if agent in cav_list:
                agent_type = "cav"
            else:
                agent_type = "rsu"
            data[agent] = OrderedDict()
            agent_dir = os.path.join(scenario_folder, "%s_%d"%(agent_type, agent))
            timestamps = _get_timestamps_in_agent(agent_dir, partname)
            for timestamp in timestamps:
                data[agent][timestamp] = OrderedDict()
                yaml_file = os.path.join(agent_dir, timestamp + '.yaml')
                lidar_file = os.path.join(agent_dir, timestamp + '.pcd')
                camera_files = _load_camera_files(agent_dir, timestamp, agent_type)
                view_file = os.path.join(agent_dir, timestamp + "_view.yaml")
                data[agent][timestamp]["yaml"] = yaml_file
                data[agent][timestamp]["lidar"] = lidar_file
                data[agent][timestamp]["camera0"] = camera_files
                data[agent][timestamp]["view"] = view_file
        scenario_database[config["scenario_name"]] = data
        
    return scenario_database




# ====================================================================
# - Dataset Class
# ====================================================================

class BaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to initialize the
    database and associate the __get_item__ index with the correct timestamp
    and scenario.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the raw point cloud will be saved in the memory
        for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    pre_processor : opencood.pre_processor
        Used to preprocess the raw data.

    post_processor : opencood.post_processor
        Used to generate training labels and convert the model outputs to
        bbx formats.

    data_augmentor : opencood.data_augmentor
        Used to augment data.

    """

    def __init__(self, params, visualize, partname="train"):
        assert partname in ['train', 'val', 'test'], "Unexpected `partname`!"

        self.params = params
        self.visualize = visualize
        self.partname = partname

        self.pre_processor = None
        self.post_processor = None
        train_flag = True if partname == 'train' else False
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train_flag)

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

        # simulate lidar intensity
        if 'simulate_pcd_intensity' not in params['preprocess']:
            params['preprocess']['simulate_pcd_intensity'] = None
        

        # CAV penetration rate
        self.penetration_rate = params['pr_setting']['value']
        self.agent_config = _get_agent_config_for_pr(
            params['pr_setting']['path'],
            self.penetration_rate
        )
        # self.agent_config = load_yaml(params['pr_setting']['path'])[
        #     str(self.penetration_rate)
        # ]

        # Structure: {scenario_name : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        root_dir = params['root_dir']
        self.scenario_database = _get_scenario_database(root_dir, self.agent_config, partname)
        
        # Structure: {scenario_name: {cav_1: [x, y, z]}}
        self.cav_extent = OrderedDict()
        for scenario_name in os.listdir(root_dir):
            scenario_root = os.path.join(root_dir, scenario_name)
            if not os.path.isdir(scenario_root):
                continue
            self.cav_extent[scenario_name] = \
                _get_cav_extent_in_scenario(os.path.join(root_dir, scenario_name))


    def __len__(self):
        return self.agent_config[-1]["length"]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    def decode_data_index(self, idx):
        """
        Given the index, return the scenario name, ego id and timestamp index.
        """
        scenario_name = None
        ego = None
        timestamp_index = None
        cur_agent_config = None
        for i in range(len(self.agent_config)):
            n = self.agent_config[i]['length']
            if idx < n:
                scenario_name = self.agent_config[i]["scenario_name"]
                prev_length = self.agent_config[i-1]["length"] if i>0 else 0
                cur_length = prev_length
                for config in self.agent_config[i]["agent_config"]:
                    for cav,length in zip(config["ego_list"], config["len_list"]):
                        if idx < cur_length + length:
                            ego = cav 
                            timestamp_index = idx - cur_length
                            cur_agent_config = config
                            break
                        cur_length += length
                    if cur_agent_config is not None:
                        break
                break
        return scenario_name, cur_agent_config, ego, timestamp_index
    
    @staticmethod
    def return_timestamp_key(scenerio_database, agent, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        agent : int
            The agent's id.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        timestamp_keys = list(scenerio_database[agent].keys())
        timestamp_keys.sort()
        timestamp_key = timestamp_keys[timestamp_index]
        return timestamp_key


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
        scenario_name,cur_agent_config,ego_id,timestamp_index = self.decode_data_index(idx)
        scenario_database = self.scenario_database[scenario_name]
        timestamp_key = self.return_timestamp_key(scenario_database, 
                                                  ego_id, 
                                                  timestamp_index)
        
        # calculate delay
        timestamp_delay = self.time_delay_calculation(ego_flag=False)
        if timestamp_index - timestamp_delay <= 0:
            timestamp_delay = timestamp_index
        timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
        timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                        ego_id,
                                                        timestamp_index_delay)

        data = OrderedDict()
        visible_objects = []
        # if the number of point hit on a object exceed or equal to thr_visibility,
        # the object is regarded as visible
        thr_visibility = 1 

        # load files for all connected agents
        conn_agents = load_yaml(scenario_database[ego_id][timestamp_key]['yaml'])["conn_agents"]
        used_agents = set(cur_agent_config["cav_list"]) | set(cur_agent_config["rsu_list"])
        conn_agents = set(conn_agents) & used_agents
        # reorder so as that ego be the first
        conn_agents = [ego_id] + list(conn_agents-set([ego_id]))
        for agent in conn_agents:
            # in case at some timestamp, the agent is not existed on the map
            if timestamp_key_delay not in scenario_database[agent]:
                continue

            data[agent] = OrderedDict()
            data[agent]['ego'] = True if agent == ego_id else False

            # add time delay vehicle parameters
            data[agent]['time_delay'] = timestamp_delay
            # load the corresponding data into the directory
            data[agent]['params'] = self.reform_param(
                scenario_database[agent],
                scenario_database[ego_id],
                timestamp_key,
                timestamp_key_delay,
                cur_ego_pose_flag,
                data[agent]['ego']
            )
            data[agent]['lidar_np'] = \
                pcd_utils.pcd_to_np(scenario_database[agent][timestamp_key_delay]['lidar'])
            
            # visible objects
            cur_vis = self.get_visible_objects(scenario_database[agent], 
                                                timestamp_key_delay, 
                                                thr_visibility)
            visible_objects.extend(cur_vis)

        # (optional) filter objects by visibility
        eval_objects = list(data[ego_id]['params']['vehicles'].keys())
        eval_objects = set(eval_objects) & set(visible_objects)
        data[ego_id]['params']['vehicles'] = {
            k:v for k,v in data[ego_id]['params']['vehicles'].items() if k in eval_objects
        }

        return data
    
    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
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

        # the data is 10 hz for both opv2v and v2x-set
        # todo: it may not be true for other dataset like DAIR-V2X and V2X-Sim
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

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
                     timestamp_delay, cur_ego_pose_flag, is_ego):
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
        delay_params['vehicles'] = cur_params['objects']
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

        return delay_params
    
    def get_visible_objects(self, cav_content, timestamp, thr=1):
        """
        Obtain the visible objects of the agent at given timestamp.

        Parameters
        ----------
        cav_content: dict
            Dictionary that contains all file paths in the current cav/rsu.

        timestamp: str
            The specific timestamp, e.g., "000068".

        thr: int
            If an object is hit by at least `thr` laser beams, the object
            is regarded as visible.

        Returns
        -------
        vis_objs: list
            The agent's visible objects.
        """
        hit_objects = load_yaml(cav_content[timestamp]["view"])["visible_objects"]
        vis_objs = [el['object_id'] for el in hit_objects 
                    if el["visible_points"]>=thr]
        return vis_objs
    
    def mask_ego_points(self, points, scenario_name, ego_id):
        """
        Remove points of the ego vehicle itself.

        Parameters
        ----------
        points : np.ndarray
            Lidar points under lidar sensor coordinate system.

        scenario_name : str
            The scenario the ego lie in

        ego_id : int
            The ego's id.

        Returns
        -------
        points : np.ndarray
            Filtered lidar points.

        """
        # there is no need to mask points for rsu
        if ego_id not in self.cav_extent[scenario_name]:
            return points
        
        ego_extent = self.cav_extent[scenario_name][ego_id]
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