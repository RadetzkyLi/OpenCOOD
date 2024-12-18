# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# Modified by: Rongsong Li <rongsong.li@qq.com>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch
import shapely

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

        result_stat[iou_thresh]['score'] += det_score.tolist()

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def calculate_ap(result_stat, iou, global_sort_detections):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
        
    iou : float
        The threshold of iou.

    global_sort_detections : bool
        Whether to sort the detection results globally.
    """
    iou_5 = result_stat[iou]

    if global_sort_detections:
        fp = np.array(iou_5['fp'])
        tp = np.array(iou_5['tp'])
        score = np.array(iou_5['score'])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()
        
    else:
        fp = iou_5['fp']
        tp = iou_5['tp']
        assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, global_sort_detections):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30, global_sort_detections)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50, global_sort_detections)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70, global_sort_detections)

    dump_dict.update({'ap30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    
    output_file = 'eval.yaml' if not global_sort_detections else 'eval_global_sort.yaml'
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))

    print('The Average Precision at IOU 0.3 is %.2f, '
          'The Average Precision at IOU 0.5 is %.2f, '
          'The Average Precision at IOU 0.7 is %.2f' % (ap_30, ap_50, ap_70))



# ===============================================================
# - New added features for testing
# ===============================================================
def caluclate_tp_fp_4test(det_boxes, det_score, gt_boxes, result_stat, iou_thresh:float, 
                          gt_object_ids, key_object_ids, gt_object_categories:list):
    """
    Calculate the true positive and false positive numbers of the current
    frames.
    For testing, the true positive numbers of key objects of the current
    frames are also counted. (batch size=1 is assumed)

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    gt_object_ids : torch.Tensor
        The groundtruth objct id.
    key_object_ids : list
        The key objects' id of current frame.
    gt_object_categories : list
        The groundtruth object category.
    """
    # fp, tp and gt number in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]

    # per object detail
    gt_object_detail = {
        "id": [],
        "is_key_object": [],
        "category": [],
        "is_recalled": [],
        'translation_error': []
    }

    gt_id_list = common_utils.torch_tensor_to_numpy(gt_object_ids).tolist()
    gt_object_categories = gt_object_categories[:]
   
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            cur_gt_polygon = gt_polygon_list.pop(gt_index)

            # per object detail
            cur_gt_id = gt_id_list.pop(gt_index)
            cur_gt_category = gt_object_categories.pop(gt_index)
            gt_object_detail['id'].append(cur_gt_id)
            gt_object_detail['category'].append(cur_gt_category)
            gt_object_detail['is_key_object'].append(1 if cur_gt_id in key_object_ids else 0)
            gt_object_detail['is_recalled'].append(1)
            gt_object_detail['translation_error'].append(calculate_translation_error(det_polygon, cur_gt_polygon))

        result_stat[iou_thresh]['score'] += det_score.tolist()
    
    # remaining gt
    for cur_gt_id,object_cat in zip(gt_id_list, gt_object_categories):
        gt_object_detail['id'].append(cur_gt_id)
        gt_object_detail['category'].append(object_cat)
        gt_object_detail['is_key_object'].append(1 if cur_gt_id in key_object_ids else 0)
        gt_object_detail['is_recalled'].append(0)

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt
    result_stat[iou_thresh]['detail']['id'] += gt_object_detail['id']
    result_stat[iou_thresh]['detail']['category'] += gt_object_detail['category']
    result_stat[iou_thresh]['detail']['is_key_object'] += gt_object_detail['is_key_object']
    result_stat[iou_thresh]['detail']['is_recalled'] += gt_object_detail['is_recalled']
    result_stat[iou_thresh]['detail']['translation_error'] += gt_object_detail['translation_error']


def calculate_recall_per_category(result_stat, iou_thresh:float):
    """
    Calculate the recall rate for each category.
    The category cannot be named by 'overall'.

    Parameters
    ----------
    result_stat : dict

    iou_thresh : float

    Returns
    -------
    category_recall_stat : dict
        e.g., {"car": {"tp": 3, "gt": 5, "recall": 0.600}}

    key_object_recall_stat : dict
        similar as `category_recall_stat`.
    """
    category_recall_stat = {}
    key_object_recall_stat = {}

    for category,is_recalled,is_key_object in zip(result_stat[iou_thresh]['detail']['category'],
                                                  result_stat[iou_thresh]['detail']['is_recalled'],
                                                  result_stat[iou_thresh]['detail']['is_key_object']):
        # all object
        if category in category_recall_stat:
            category_recall_stat[category]['gt'] += 1
        else:
            category_recall_stat[category] = {
                "gt": 1, "tp": 0
            }
            category_recall_stat[category]['tp'] = 0
        if is_recalled:
            category_recall_stat[category]['tp'] += 1
        
        # key object
        if is_key_object:
            if category in key_object_recall_stat:
                key_object_recall_stat[category]['gt'] += 1
            else:
                key_object_recall_stat[category] = {
                    "gt": 1, "tp": 0
                }
            if is_recalled:
                key_object_recall_stat[category]['tp'] += 1

    # calculate recall rate
    digits = 3
    tp,gt = 0,0
    category_recall_stat = dict(sorted(category_recall_stat.items(), key=lambda d: d[0]))
    for category,stat in category_recall_stat.items():
        stat['recall'] = round(stat['tp']/stat['gt'], digits)
        tp += stat['tp']
        gt += stat['gt']
    category_recall_stat['overall'] = {
        "tp": tp, 'gt': gt, 'recall': round(tp/gt, digits) if gt>0 else 0.0
    }

    tp,gt = 0,0
    key_object_recall_stat = dict(sorted(key_object_recall_stat.items(), key=lambda d: d[0]))
    for category,stat in key_object_recall_stat.items():
        stat['recall'] = round(stat['tp']/stat['gt'], digits)
        tp += stat['tp']
        gt += stat['gt']
    key_object_recall_stat['overall'] = {
        "tp": tp, "gt": gt, "recall": round(tp/gt, digits) if gt>0 else 0.0
    }

    return category_recall_stat, key_object_recall_stat

def calculate_ate(result_stat, iou_thresh:float):
    data = result_stat[iou_thresh]['detail']['translation_error']
    return np.array(data).mean()

def calculate_translation_error(box1, box2):
    return shapely.distance(box1.centroid, box2.centroid)

def print_category_recall(category_recall_stat):
    ENDC = '\033[0m'
    OKGREEN = '\033[92m'
    header = OKGREEN +"%12s\t%6s\t%6s\t%s"%("Category", 'TP', 'GT', 'Recall') + ENDC + "\n"
    body = header
    line_format = "%12s\t%6d\t%6d\t%.3f\n"
    for category,stat in category_recall_stat.items():
        body += line_format%(category, stat['tp'], stat['gt'], stat['recall'])
    print(body)

def eval_final_results_4test(result_stat, save_path, global_sort_detections):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30, global_sort_detections)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50, global_sort_detections)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70, global_sort_detections)

    translation_error_30 = calculate_ate(result_stat, 0.30)
    translation_error_50 = calculate_ate(result_stat, 0.50)
    translation_error_70 = calculate_ate(result_stat, 0.70)

    dump_dict.update({'ap30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70,
                      'translation_error_30': translation_error_30,
                      'translation_error_50': translation_error_50,
                      'translation_error_70': translation_error_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    
    output_file = 'eval.yaml' if not global_sort_detections else 'eval_global_sort.yaml'
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))

    print('The Average Precision at IOU 0.3 is %.3f, '
          'The Average Precision at IOU 0.5 is %.3f, '
          'The Average Precision at IOU 0.7 is %.3f' % (ap_30, ap_50, ap_70))
    
    print('The Average Translation Error at IOU 0.3 is %.3f meter, '
          'The Average Translation Error at IOU 0.5 is %.3f meter, '
          'The Average Translation Error at IOU 0.7 is %.3f meter' % (translation_error_30, translation_error_50, translation_error_70))

    # recall of objects of various categories
    category_recall_30,key_object_recall_30 = calculate_recall_per_category(result_stat, 0.30)
    category_recall_50,key_object_recall_50 = calculate_recall_per_category(result_stat, 0.50)
    category_recall_70,key_object_recall_70 = calculate_recall_per_category(result_stat, 0.70)

    print("Per category recall at IOU 0.3 is:")
    print_category_recall(category_recall_30)

    print("Per category recall at IOU 0.5 is:")
    print_category_recall(category_recall_50)

    print("Per category recall at IOU 0.7 is:")
    print_category_recall(category_recall_70)

    if len(key_object_recall_30)>1:
        print("Key object recall at IOU 0.3 is:")
        print_category_recall(key_object_recall_30)

        print("Key object recall at IOU 0.5 is:")
        print_category_recall(key_object_recall_50)

        print("Key object recall at IOU 0.7 is:")
        print_category_recall(key_object_recall_70)
