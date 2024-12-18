#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   __init__.py
@Date    :   2024-01-31
@Author  :   Rongsong Li <rongsong.li@qq.com>
@Version :   1.0
@Desc    :   build scenario database manager 
'''


from opencood.data_utils.scenario_database_manager.opv2v_database_manager \
    import Opv2vDatabaseManager
from opencood.data_utils.scenario_database_manager.opv2v_database_manager_4test \
    import Opv2vDatabaseManager4Test
from opencood.data_utils.scenario_database_manager.opv2v_database_manager_v2 \
    import Opv2vDatabaseManagerV2

__all__ = {
    'opv2v': Opv2vDatabaseManager,
    'v2v4real': Opv2vDatabaseManager,
    'test': Opv2vDatabaseManager4Test,
    'multi-v2x': Opv2vDatabaseManagerV2
}


def build_scenario_database_manager(dataset_cfg, partname:str, dataset_format:str):
    error_message = "Dataset format '{0}' is not found! "\
                    "Please add your support for such format "\
                    "in opencood/data_utils/scenario_database_manager/__init__.py".format(
                        dataset_format
                    )
    assert dataset_format in __all__, error_message

    manager = __all__[dataset_format](
        params = dataset_cfg,
        partname = partname
    )

    return manager