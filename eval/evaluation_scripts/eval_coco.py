import re
import os
import PIL
import json
import torch
import numpy as np
from datasets import Dataset, load_dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        suffix = sys.argv[1]
    else:
        suffix = 'padt_pro_3b'

    output_dir = '../outputs/coco'  # Add the output directory, default is logs

    log_result_paths = [os.path.join(output_dir, f'coco_{i}_pred_results_{suffix}.json') for i in range(8)]

    preds = []
    for log_result_file in log_result_paths:
        if os.path.exists(log_result_file) is False:
            continue
        with open(log_result_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                preds.append(item)
    
    data_files = ['PaDT-MLLM/COCO/instances_val2017.json']  # Add the data root
    ori_coco_json = '../../dataset/coco/annotations/instances_val2017.json'
    ori_coco_obj = COCO(ori_coco_json)

    gts = {
        'info': ori_coco_obj.dataset['info'],
        'licenses': ori_coco_obj.dataset['licenses'],
        'images': ori_coco_obj.dataset['images'],
        'annotations': [],
        'categories': ori_coco_obj.dataset['categories'],
    }

    name_to_cat_id = {cat['name']: cat['id'] for cat in ori_coco_obj.dataset['categories']}

    ann_ids = 1
    for data_file in data_files:
        if os.path.exists(data_file) is False:
            data = load_dataset("/".join(data_file.split("/")[:-1]), data_files=data_file.split("/")[-1])['train'].to_list()
        else:
            data = [json.loads(i) for i in open(data_file, 'r').readlines()]
        
        for item in data:
            for obj_item in item['objects']:
                cat_id = name_to_cat_id[obj_item['label']]
                img_h, img_w = ori_coco_obj.imgs[item['id']]['height'], ori_coco_obj.imgs[item['id']]['width']
                # (x1, y1, x2, y2) -> (x1, y1, w, h)
                bbox = [round(obj_item['bbox'][0] * img_w), round(obj_item['bbox'][1] * img_h), round((obj_item['bbox'][2] - obj_item['bbox'][0]) * img_w), round((obj_item['bbox'][3] - obj_item['bbox'][1]) * img_h)]
                gts['annotations'].append({
                    'id': ann_ids,
                    'image_id': item['id'],
                    'category_id': cat_id,
                    'iscrowd': obj_item['iscrowd'],
                    'area': obj_item['area'],
                    'bbox': bbox
                })
                ann_ids += 1

    new_preds = []
    for pred in preds:
        try:
            pred['category_id'] = name_to_cat_id[pred['category'].lower()]
            del pred['category']
        except:
            continue
        new_preds.append(pred)

    coco_gt = COCO()
    coco_gt.dataset = gts
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(new_preds)

    iou_type = 'bbox'
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    # coco_eval.params.iouThrs=[0.5]
    # coco_eval.params.iouThrs=[0.75]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mean_average_precision = coco_eval.stats[0]
    print(f"\nMean Average Precision (mAP): {mean_average_precision:.3f}")
