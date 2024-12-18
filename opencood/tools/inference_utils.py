# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modified by: Rongsong Li <rongsong.li@qq.com>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils import box_utils
from opencood.utils.common_utils import torch_tensor_to_numpy


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor

def inference_no_fusion(batch_data, model, dataset):
    """
    Model inference for no fusion.
    """
    return inference_late_fusion(batch_data, model, dataset)


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for intermediate fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%05d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%05d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%05d_gt.npy_test' % timestamp), gt_np)



# ====================================================
# New added features for testing
# ====================================================
def generate_gt_bbx_with_id(data_dict, order):
    """
    The base postprocessor will generate 3d groundtruth bounding box.

    This is variant of function `generate_gt_bbx` in opencood.data_utils.post_processor.base_postprocessor. 
    The modified version will generate 3d bounding box and its object ids.

    Parameters
    ----------
    data_dict : dict
        The dictionary containing the origin input data of model.
    order : str
        'lwh' or 'hwl'

    Returns
    -------
    gt_box3d_tensor : torch.Tensor
        The groundtruth bounding box tensor, shape (N, 8, 3).
    gt_object_id_tensor : torch.Tensor
        The groundtruth object id tensor, shape (N,)
    """
    gt_box3d_list = []
    # used to avoid repetitive bounding box
    gt_id_list = []

    for cav_id, cav_content in data_dict.items():
        # used to project gt bounding box to ego space
        transformation_matrix = cav_content['transformation_matrix']

        object_bbx_center = cav_content['object_bbx_center']
        object_bbx_mask = cav_content['object_bbx_mask']
        object_ids = cav_content['object_ids']
        object_bbx_center = object_bbx_center[object_bbx_mask == 1]

        # convert center to corner
        object_bbx_corner = \
            box_utils.boxes_to_corners_3d(object_bbx_center,
                                            order)
        projected_object_bbx_corner = \
            box_utils.project_box3d(object_bbx_corner.float(),
                                    transformation_matrix)
        gt_box3d_list.append(projected_object_bbx_corner)

        # append the corresponding ids
        gt_id_list += object_ids

    # gt bbx 3d
    gt_box3d_list = torch.vstack(gt_box3d_list)
    # some of the bbx may be repetitive, use the id list to filter
    gt_box3d_selected_indices = \
        [gt_id_list.index(x) for x in set(gt_id_list)]
    gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

    # filter the gt_box to make sure all bbx are in the range
    mask = \
        box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
    gt_box3d_tensor = gt_box3d_tensor[mask, :, :]

    # get object ids
    gt_object_id_tensor = torch.IntTensor(gt_id_list)[gt_box3d_selected_indices]
    gt_object_id_tensor = gt_object_id_tensor[mask.cpu()]

    return gt_box3d_tensor,gt_object_id_tensor


def generate_gt_bbx_with_id_and_category(data_dict, order:str, key_object_ids:list=[]):
    """Generate groundtruth bounding boxes with corresponding id and category.
    Modified from opencood.data_utils.post_processor.base_postprocessor.generate_gt_bbx.

    Parameters
    ----------
    data_dict : dict
        The dictionary containing the origin input data of model.
    order : str
        'lwh' or 'hwl'
    key_object_ids : list
        Key object will not be filtered.
    
    Returns
    -------
    gt_object_id_tensor : torch.Tensor
        The groundtruth object id tensor, shape (N,)
    gt_object_category_list : list
        The groundtruth object category list, shape (N,)
    """
    gt_box3d_list = []
    # used to avoid repetitive bounding box
    gt_id_list = []
    gt_object_category_list = []

    for cav_id, cav_content in data_dict.items():
        # used to project gt bounding box to ego space
        transformation_matrix = cav_content['transformation_matrix']

        object_bbx_center = cav_content['object_bbx_center']
        object_bbx_mask = cav_content['object_bbx_mask']
        object_ids = cav_content['object_ids']
        object_bbx_center = object_bbx_center[object_bbx_mask == 1]

        # convert center to corner
        object_bbx_corner = \
            box_utils.boxes_to_corners_3d(object_bbx_center,
                                            order)
        projected_object_bbx_corner = \
            box_utils.project_box3d(object_bbx_corner.float(),
                                    transformation_matrix)
        gt_box3d_list.append(projected_object_bbx_corner)

        # append the corresponding ids
        gt_id_list += object_ids
        gt_object_category_list += cav_content['object_categories']

    # gt bbx 3d
    gt_box3d_list = torch.vstack(gt_box3d_list)
    # some of the bbx may be repetitive, use the id list to filter
    gt_box3d_selected_indices = \
        [gt_id_list.index(x) for x in set(gt_id_list)]
    gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

    # filter the gt_box to make sure all bbx are in the range
    mask = \
        box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor)
    # key objects will be saved anyway
    for _id in key_object_ids:
        mask[gt_box3d_selected_indices.index(gt_id_list.index(_id))] = True
    gt_box3d_tensor = gt_box3d_tensor[mask, :, :]

    # get object ids
    gt_object_id_tensor = torch.IntTensor(gt_id_list)[gt_box3d_selected_indices]
    gt_object_id_tensor = gt_object_id_tensor[mask.cpu()]
    # get object categories
    gt_object_category_list = [gt_object_category_list[i] for i in gt_box3d_selected_indices]
    gt_object_category_list = [obj_cat for obj_cat,flag in 
                               zip(gt_object_category_list, mask.cpu().numpy())
                               if flag]

    return gt_box3d_tensor, gt_object_id_tensor,gt_object_category_list



def save_prediction_gt_id(pred_tensor, score_tensor, gt_tensor, pcd, gt_ids_tensor, prefix, save_dir):
    """
    Save prediction and gt tensor to npz file 
    """
    pred_np = torch_tensor_to_numpy(pred_tensor) if pred_tensor is not None else None
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)
    score_np = torch_tensor_to_numpy(score_tensor) if pred_tensor is not None else None
    gt_ids_np = torch_tensor_to_numpy(gt_ids_tensor)

    # np.save(os.path.join(save_dir, prefix + '__pcd.npy'), pcd_np)
    # np.save(os.path.join(save_dir, prefix + '__pred.npy'), pred_np)
    # np.save(os.path.join(save_dir, prefix + '__gt_test.npy'), gt_np)
    # np.save(os.path.join(save_dir, prefix + '__gt_id.npy'), gt_ids_tensor.cpu().numpy())
    # np.save(os.path.join(save_dir, prefix + '__score.npy'), score_np)

    np.savez(os.path.join(save_dir, prefix+".npz"), 
             pred=pred_np, 
             score=score_np, 
             gt=gt_np, 
             pcd=pcd_np,
             gt_ids=gt_ids_np)
