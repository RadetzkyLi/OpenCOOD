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
        self.partname = partname
        self.params = params

        self.root = ""

        self.sampling_frequency = 10    # Hz
        self.comm_type = "v2x"          # 'none', 'v2v', 'v2i' and 'v2x' supported
        self.comm_range = 70            # meters
        

    def initialize_database(self, root:str):
        """Get scenario database.
        
        Parameters
        ----------
        root : str
            dataset root.
        """
        pass

    def get_sample(self, index:int):
        """Return corresponding sample for given index.
        """
        pass

    def get_number_of_total_samples(self):
        """
        Returns sample length
        """
        pass

    def calc_timestamp_with_gap(self, timestamp, time_gap:float):
        pass
    
    def is_rsu_agent(self, scenario_id, agent_id):
        """Judge whether the given agent is RSU.
        Paramters
        ---------
        scenario_id : int
            
        agent_id : int

        """
        pass