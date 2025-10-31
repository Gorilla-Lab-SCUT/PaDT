import re
import os
import PIL.Image
import json
import torch
import numpy as np
from collections import defaultdict
from datasets import Dataset, load_dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as cocomask



def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x1_prime, y1_prime, w1_prime, h1_prime = bbox2

    bbox1_coords = [x1, y1, x1 + w1, y1 + h1]
    bbox2_coords = [x1_prime, y1_prime, x1_prime + w1_prime, y1_prime + h1_prime]

    inter_x1 = max(bbox1_coords[0], bbox2_coords[0])
    inter_y1 = max(bbox1_coords[1], bbox2_coords[1])
    inter_x2 = min(bbox1_coords[2], bbox2_coords[2])
    inter_y2 = min(bbox1_coords[3], bbox2_coords[3])

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    inter_area = inter_width * inter_height

    bbox1_area = w1 * h1
    bbox2_area = w1_prime * h1_prime

    union_area = bbox1_area + bbox2_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def calculate_ciou(pred: np.ndarray, gt: np.ndarray):
    i = np.logical_and(pred, gt).sum()
    u = np.logical_or(pred, gt).sum()
    return i/u if u>0 else 0.0



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        suffix = sys.argv[1]
        split = sys.argv[2]
    else:
        suffix = 'padt_pro_3b'
        split = 'refcoco_val'

    output_dir = '../outputs/refcoco'  # Add the output directory, default is logs

    log_result_paths = [os.path.join(output_dir, f'{split}_{i}_pred_results_{suffix}.json') for i in range(8)]

    preds = []
    for log_result_file in log_result_paths:
        if os.path.exists(log_result_file) is False:
            continue
        with open(log_result_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                preds.append(item)
    
    data_files = [f'PaDT-MLLM/RefCOCO/{split}.json']  # Add the data root
    image_folders = ['../../dataset/coco/train2014']  # Add the image root
    assert len(data_files) == len(image_folders), "Number of data files must match number of image folders"


    gt_dict = defaultdict(list)
    accuracy = defaultdict(int)
    mask_cious = defaultdict(float)
    for data_file, image_folder in zip(data_files, image_folders):
        if os.path.exists(data_file) is False:
            data = load_dataset("/".join(data_file.split("/")[:-1]), data_files=data_file.split("/")[-1])['train'].to_list()
        else:
            data = [json.loads(i) for i in open(data_file, 'r').readlines()]

        for item in data:
            bbox_name = '%d_%s' % (item['id'], item['objects'][0]['label'])
            image = PIL.Image.open(os.path.join(image_folder, item['image']))
            width, height = image.size
            # (x1, y1, x2, y2) -> (x1, y1, w, h)
            gt_bbox = item['objects'][0]['bbox']
            gt_bbox = [round(gt_bbox[0] * width), round(gt_bbox[1] * height), round((gt_bbox[2] - gt_bbox[0]) * width), round((gt_bbox[3] - gt_bbox[1]) * height)]
            rle = item['objects'][0]['rle']
            mask = cocomask.decode(rle)
            gt_dict[bbox_name] = [gt_bbox, mask]
            accuracy[bbox_name] = 0.

    pred_dict = defaultdict(list)
    for pred in preds:
        bbox_name = '%d_%s' % (pred['image_id'], pred['category'])
        if bbox_name in gt_dict:
            gt_bbox, gt_mask = gt_dict[bbox_name]
            pred_bbox = pred['bbox']
            pred_mask = cocomask.decode(pred['mask'])

            ciou = calculate_ciou(pred_mask > 0, gt_mask > 0)
            iou = calculate_iou(gt_bbox, pred_bbox)

            accuracy[bbox_name] = max(iou, accuracy[bbox_name])
            mask_cious[bbox_name] = max(ciou, mask_cious[bbox_name])
            pred_dict[bbox_name] = [pred_bbox, pred_mask]

    all_ious = np.array([i for i in accuracy.values()])
    all_mask_cious = np.array([i for i in mask_cious.values()])
    ap = (all_ious >= 0.5).mean()
    mean_cious = all_mask_cious.mean()
    print('The results using our validation set.')
    print('REC AP_50:', ap, '| RES CIoU:', mean_cious)

    # align to VLM-R1
    vlm_eval_ap = []
    vlm_eval_ciou = []
    vlm_json_files = ['../../dataset/RefCOCO/rec_jsons_processed/%s.json' % (split.replace('refcoco+', 'refcocop'))]
    with open(vlm_json_files[0], 'r') as f:
        items = json.load(f)
        for idx, item in enumerate(items):
            image_id = int(item['image'].split('_')[-1].split('.')[0])
            category = item['normal_caption']
            vlm_eval_ap.append(accuracy['%d_%s' % (image_id, category)] >= 0.5)
            vlm_eval_ciou.append(mask_cious['%d_%s' % (image_id, category)])

    print('\nThe results using VLM-R1 validation set. [The results present in our paper]')
    print('REC AP_50:', np.array(vlm_eval_ap).mean().item(), '| RES CIoU:', np.array(vlm_eval_ciou).mean().item())

