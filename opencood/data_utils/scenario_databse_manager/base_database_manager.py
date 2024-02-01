#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   base_database_manager.py
@Date    :   2024-01-31
@Author  :   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   Manage scenario database
'''

import os
import sys
import numpy as np 


class BaseDatabaseManager:
    """
    Manage the v2x scenario database. A scenario database
    is composed of multiple scenario data, in which each
    scenario is composed of data recordings of multiple
    CAVs and RSUs. The manager is in charge of :
        1. retriving samples for given integer index;
        2. retriving samples for given time gap

    
    """
    def __init__(self, params, partname='train') -> None:
        self.sampling_frequency = None  # Hz
        self.partname = partname
        self.params = params
        

    def construct_database(self, root):
        pass

    def get_sample(self, index:int):
        """Return corresponding sample for given index.
        """
        pass

    def calc_timestamp_with_gap(self, timestamp, time_gap:float):
        pass
    