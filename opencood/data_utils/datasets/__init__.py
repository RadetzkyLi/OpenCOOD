# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modified by: Rongsong Li <rongsong.li@qq.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.datasets.no_fusion_dataset import NoFusionDataset

__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'NoFusionDataset': NoFusionDataset,
}


# the final range for evaluation
GT_RANGE_V2V4Real = [-100, -40, -5, 100, 40, 3]  # used for V2V4Real
GT_RANGE_OPV2V    = [-140, -40, -3, 140, 40, 1]  # used for OPV2V and V2XSet
GT_RANGE_MultiV2X = [-140, -40, -3, 140, 40, 3]  # used for multi-v2x and testing scenes
GT_RANGE = GT_RANGE_OPV2V

# The communication range for cavs
COM_RANGE = 70



def build_dataset(dataset_cfg, visualize=False, partname="train"):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in list(__all__.keys()), error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        partname=partname
    )

    # adjust GT_RANGE according to dataset format
    dataset_format = dataset_cfg.get("dataset_format", "opv2v")
    global GT_RANGE
    if dataset_format == "v2v4eal":
        GT_RANGE = GT_RANGE_V2V4Real
    elif dataset_format == "multi-v2x" or dataset_format == "test":
        GT_RANGE = GT_RANGE_MultiV2X

    return dataset
