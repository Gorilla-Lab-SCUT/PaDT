import os
import cv2
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from collections import defaultdict


def filter_coco_annotations_with_api(input_json_path, output_filtered_json, max_bboxes_per_class_per_image=10, is_train=False, drop_rate=0.5, max_class_in_prompt=100):
    os.makedirs(os.path.dirname(output_filtered_json), exist_ok=True)
    with open(output_filtered_json, 'w') as f:
        f.write("")

    print(f"Loading: {input_json_path}...")
    coco = COCO(input_json_path)
    print(f"Finish.")

    # Get all image ids and category ids
    image_ids = coco.getImgIds()
    category_ids = set(coco.getCatIds())
    categories_info = coco.loadCats(category_ids)
    
    id_to_category_name = {cat['id']: cat['name'] for cat in categories_info}

    category_index = np.array(list(category_ids))

    this_turn_pass_number = 0
    total_number = 0
    skip_resolution = 0

    # check whether bbox number per category exceed max_bboxes_per_class_per_image
    for img_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        annotations = coco.loadAnns(ann_ids)
        image_category_bbox_counts = defaultdict(int)

        image_info = coco.loadImgs(img_id)[0]
        ori_h, ori_w = image_info['height'], image_info['width']
        max_side = max(ori_h, ori_w)
        # round(640 * 2 / 28) * 28
        if max_side > 1288:
            skip_resolution += 1
            continue

        for ann in annotations:
            category_id = ann['category_id']
            image_category_bbox_counts[category_id] += 1

        remove_category_ids = set()
        if is_train:
            np.random.shuffle(category_index)
            remove_category_ids.update(category_index[max_class_in_prompt:])
            if (np.random.rand(1) < drop_rate).item():
                remove_category_ids.update(category_index[:int(drop_rate*min(len(category_index), max_class_in_prompt))].tolist())

        category_to_index = {val: idx for idx, val in enumerate(category_index)}
        answer_list = []
        for category_id, count in sorted(image_category_bbox_counts.items(), key=lambda item: category_to_index[item[0]]):
            if category_id in remove_category_ids:
                remove_category_ids.remove(category_id)

            if count > max_bboxes_per_class_per_image:
                remove_category_ids.add(category_id)
                continue

            this_cat_objects = []
            ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[category_id])
            annotations = coco.loadAnns(ann_ids)

            for ann in annotations:
                if 'segmentation' in ann:
                    mask = coco.annToMask(ann)
                    ori_h, ori_w = mask.shape[:2]

                    resized_h, resized_w = int(round(ori_h / 28) * 28), int(round(ori_w / 28) * 28)
                    resized_mask = cv2.resize(mask * 255, (resized_w, resized_h))
                    patch_mask = resized_mask.reshape(resized_h // 28, 28, resized_w//28, 28).transpose(0, 2, 1, 3).mean(axis=-1).mean(axis=-1) >= 255 / 28

                    if patch_mask.sum() < 1:
                        this_turn_pass_number += 1
                        continue

                    save_rle = cocomask.encode(mask.astype(np.uint8))
                    save_rle['counts'] = save_rle['counts'].decode()

                    this_cat_objects.append({
                        'patches': np.where(patch_mask.reshape(-1))[0].tolist(),
                        'bbox': [
                            ann['bbox'][0] * (resized_w / ori_w) / resized_w, 
                            ann['bbox'][1] * (resized_h / ori_h) / resized_h, 
                            (ann['bbox'][0] + ann['bbox'][2]) * (resized_w / ori_w) / resized_w, 
                            (ann['bbox'][1] + ann['bbox'][3]) * (resized_h / ori_h) / resized_h
                        ],
                        'iscrowd': ann['iscrowd'],
                        'area': ann['area'],
                        'rle': save_rle
                    })
                else:
                    mask = np.zeros((ori_h, ori_w)).astype(np.uint8)
                    x1, y1, w, h = ann['bbox']
                    x1, y1, x2, y2 = round(x1), round(y1), round(x1 + w), round(y1 + h)
                    mask[y1: y2, x1: x2] = 1

                    resized_h, resized_w = int(round(ori_h / 28) * 28), int(round(ori_w / 28) * 28)
                    resized_mask = cv2.resize(mask * 255, (resized_w, resized_h))

                    patch_mask = resized_mask.reshape(resized_h // 28, 28, resized_w//28, 28).transpose(0, 2, 1, 3).mean(axis=-1).mean(axis=-1) >= 255 / 28

                    if patch_mask.sum() < 1:
                        this_turn_pass_number += 1
                        continue
                    
                    this_cat_objects.append({
                        'patches': np.where(patch_mask.reshape(-1))[0].tolist(),
                        'bbox': [
                            ann['bbox'][0] * (resized_w / ori_w) / resized_w, 
                            ann['bbox'][1] * (resized_h / ori_h) / resized_h, 
                            (ann['bbox'][0] + ann['bbox'][2]) * (resized_w / ori_w) / resized_w, 
                            (ann['bbox'][1] + ann['bbox'][3]) * (resized_h / ori_h) / resized_h
                        ],
                        'iscrowd': ann['iscrowd'],
                        'area': ann['area'],
                    })

                total_number += 1

            if len(this_cat_objects) > 0:
                answer_list.append({
                    'label': id_to_category_name[category_id],
                    'objects': this_cat_objects
                })
        

        # create answer template
        object_num_per_category = [len(item['objects']) for item in answer_list]
        if len(object_num_per_category) == 0:
            answer_template = 'No objects from the list are present in the image'
        elif len(object_num_per_category) == 1:
            if sum(object_num_per_category) == 1:
                answer_template = "There is "
            else:
                answer_template = "There are "
        else:
            answer_template = "In this image, there are "

        objects = []
        for cat_idx, cat_answer_list in enumerate(answer_list):
            answer_template += str(len(cat_answer_list['objects'])) + " \"%s\" (" % cat_answer_list['label']
            for obj_idx, obj in enumerate(cat_answer_list['objects']):
                obj['label'] = cat_answer_list['label']
                answer_template += "<|Obj_%d|>" % len(objects)
                objects.append(obj)
                if obj_idx < len(cat_answer_list['objects']) - 1:
                    answer_template += ", "
                else:
                    answer_template += ")"
            if cat_idx < len(answer_list) - 1:
                answer_template += ", "

        if len(object_num_per_category) == 1:
            answer_template += " in this image."                
        else:
            answer_template += "."
        
        # make conversation.
        target_list = list(category_ids - remove_category_ids)
        target_list = sorted(target_list, key=lambda item: category_to_index[item])

        ref_list_in_group = coco.loadCats(target_list)
        ref_list_in_group = [cat['name'] for cat in ref_list_in_group]

        data_item = {
            'id': img_id,
            'image': coco.loadImgs([img_id])[0]['file_name'],
            'conversations': [
                {
                    "from": "human",
                    "value": "Please carefully check the image and detect the following objects: " + json.dumps(ref_list_in_group) + "."
                }
            ],
            "answer_template": answer_template,
            "objects": objects,
            "task": 'ovd',
        }

        with open(os.path.join(output_filtered_json), "a+") as f:
            f.write(json.dumps(data_item) + "\n")

    print(this_turn_pass_number)
    print(total_number)
    print(skip_resolution)


if __name__ == '__main__':
    os.makedirs('../../dataset/coco/processed', exist_ok=True)
    # create validation set
    input_coco_json = '../../dataset/coco/annotations/instances_val2017.json'
    output_filtered_json = '../../dataset/coco/processed/instances_val2017.json'
    is_train = False  # will randomly drop some target categories in target list.
    filter_coco_annotations_with_api(input_coco_json, output_filtered_json, max_bboxes_per_class_per_image=50, is_train=is_train)

    # create train set
    input_coco_json = '../../dataset/coco/annotations/instances_train2017.json'
    output_filtered_json = '../../dataset/coco/processed/instances_train2017.json'
    is_train = True  # will randomly drop some target categories in target list.
    filter_coco_annotations_with_api(input_coco_json, output_filtered_json, max_bboxes_per_class_per_image=50, is_train=is_train)