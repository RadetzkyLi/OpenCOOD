#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   pcd_intensity_utils.py
@Date    :   2024-01-29
@Author  :   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Simulate pointcloud intensity
'''

import numpy as np


def simulate_pointcloud_intensity_carla(xyz, a:float=0.004):
    """
    Simulate the intensity of point cloud. 
        I/I_0 = exp(-a*d)

    Parameters
    ----------
    xyz : ndarray
        The lidar data in numpy format, shape: (n,3)

    a : float
        Attenuation coefficient. This may depend on the sensor's 
        wavelenght, and the conditions of the atmosphere. Defaults
        to 0.004 as CARLA 0.9.12

    Returns
    -------
    pcd_np : np.ndarray
        The lidar data in numpy format, shape: (n,4)

    Refs
    ----
    1. https://carla.readthedocs.io/en/latest/ref_sensors/#lidar-sensor
    """
    d = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)
    i_arr = np.exp(-a * d)
    pcd_np = np.hstack((xyz, i_arr.reshape(i_arr.shape[0], 1)))

    return pcd_np



def simulate_pcd_intensity(xyz, method='carla', **kwargs):
    """The interface for simulating pointcloud intensity. Default to
    that of CARLA.
    
    Parameters
    ----------
    xyz : ndarray
        The lidar data in numpy format.

    method : str
        the simulation method.

    **kwargs : dict
        arguments for method.

    Returns
    -------
    pcd_np : np.ndarray
        The lidar data in numpy format, shape: (n,4)
    """
    if method is None:
        return xyz
    elif method == 'carla':
        a = float(kwargs.get('a', 0.004))
        return simulate_pointcloud_intensity_carla(xyz, a)
    else:
        raise ValueError("Unexpected point cloud intensity simulation method: {0}".format(method))
