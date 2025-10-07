import os
import re
import cv2
import json
import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from collections import defaultdict

def prepare_ric_annotations_with_api(input_json_path, output_filtered_json):
    os.makedirs(os.path.dirname(output_filtered_json), exist_ok=True)
    with open(output_filtered_json, 'w') as f:
        f.write("")

    print(f"Loading: {input_json_path}...")
    coco = COCO(input_json_path)
    print("Finish.")

    # Get all image ids and category ids
    image_ids = coco.getImgIds()

    this_turn_pass_number = 0
    total_number = 0

    for img_id in tqdm.tqdm(image_ids):
        ann_ids_img = coco.getAnnIds(imgIds=[img_id])

        image_info = coco.loadImgs(img_id)[0]
        captions = image_info['captions']

        for caption in captions:
            if caption[-1] != '.' and caption[-1] != '"':
                print('Caption not end:', caption)
                continue

            unexpected_pattern_1 = r'(\(\d+(,\s*\d+)*\))'
            unexpected_results_1 = re.findall(unexpected_pattern_1, caption)
            for ur1 in unexpected_results_1:
                ur_str = ur1[0]
                ur_str_replaced = ur_str
                ur_idxs = re.findall(r'(\d+)', ur_str)
                for ur_idx in ur_idxs:
                    if int(ur_idx) in ann_ids_img:
                        ur_str_replaced = ur_str_replaced.replace(ur_idx, '<box_id: %s/>' % ur_idx)
                caption = caption.replace(ur_str, ur_str_replaced)

            unexpected_pattern_2 = r'(<box_id:\s*[^>\d]+(\d+)/?>)'
            unexpected_results_2 = re.findall(unexpected_pattern_2, caption)
            for ur2 in unexpected_results_2:
                caption = caption.replace(ur2[0], '<box_id: %s/>' % ur2[1])

            # (<box_id: 405710/, 714044/>) (<box_id: 1057221,1561272>) (<box_id: person_ann_id/2016798, 2153731, 184066>)
            unexpected_pattern_3 = r'(<box_id:\s*[^>\d]*\d+/?(,\s*\d+/?)+>)'
            unexpected_results_3 = re.findall(unexpected_pattern_3, caption)
            for ur3 in unexpected_results_3:
                ur_str = ur3[0]
                ur_idxs = re.findall(r'(\d+)', ur_str)
                caption = caption.replace(ur_str, ', '.join(['<box_id: %s/>' % i for i in ur_idxs]))

             # (<box_id: 405710/)
            unexpected_pattern_4 = r'(<box_id:\s*[^>\d]*(\d+)/(?!>))'
            unexpected_results_4 = re.findall(unexpected_pattern_4, caption)
            for ur4 in unexpected_results_4:
                ur_str = ur4[0]
                caption = caption.replace(ur_str, '<box_id: %s/>' % ur4[1])

            pattern = r'(<box_id:\s*(\d+)/?>)'
            results = re.findall(pattern, caption)
            ann_ids = [int(i[1]) for i in results]

            pattern_without_matching = r'<box_id:\s*\d+/?>'
            caption_parts = re.split(pattern_without_matching, caption)

            new_caption = caption_parts[0]

            this_caption_objects = []
            for idx, (ann_id, caption_part) in enumerate(zip(ann_ids, caption_parts[1:])):
                try:
                    ann = coco.loadAnns([ann_id])[0]
                    assert ann['image_id'] == img_id
                    
                    mask = coco.annToMask(ann)
                    ori_h, ori_w = mask.shape[:2]

                    resized_h, resized_w = int(round(ori_h / 28) * 28), int(round(ori_w / 28) * 28)
                    resized_mask = cv2.resize(mask * 255, (resized_w, resized_h))
                    patch_mask = resized_mask.reshape(resized_h // 28, 28, resized_w//28, 28).transpose(0, 2, 1, 3).mean(axis=-1).mean(axis=-1) >= 255 / 28

                    if patch_mask.sum() < 1:
                        this_turn_pass_number += 1
                        # delete this box
                        if new_caption[-2:] == ', ':
                            new_caption = new_caption[:-2] + caption_part
                        elif new_caption[-1] == '(':
                            if caption_part[0] == ')':
                                new_caption = new_caption[:-2] + caption_part[1:]
                            else:
                                new_caption += caption_part[2:]
                        continue
                    else:
                        # caption = caption.replace('%s' % pattern_str, '<|Obj_%d|>' % idx)
                        new_caption += "<|Obj_%d|>" % len(this_caption_objects) + caption_part

                    save_rle = cocomask.encode(mask.astype(np.uint8))
                    save_rle['counts'] = save_rle['counts'].decode()

                    this_caption_objects.append({
                        'patches': np.where(patch_mask.reshape(-1))[0].tolist(),
                        'bbox': [
                            ann['bbox'][0] * (resized_w / ori_w) / resized_w, 
                            ann['bbox'][1] * (resized_h / ori_h) / resized_h, 
                            (ann['bbox'][0] + ann['bbox'][2]) * (resized_w / ori_w) / resized_w, 
                            (ann['bbox'][1] + ann['bbox'][3]) * (resized_h / ori_h) / resized_h
                        ],
                        'iscrowd': ann['iscrowd'],
                        'area': ann['area'],
                        'rle': save_rle,
                        'label': ''
                    })
                    total_number += 1
                except:
                    # delete this box
                    if new_caption[-2:] == ', ':
                        new_caption = new_caption[:-2] + caption_part
                    elif new_caption[-1] == '(':
                        if caption_part[0] == ')':
                            new_caption = new_caption[:-2] + caption_part[1:]
                        else:
                            new_caption += caption_part[2:]
                    # caption = caption.replace('(%s)' % pattern_str, '').replace('%s, ' % pattern_str, '').replace(', %s' % pattern_str, '')
                    print('missing ann_id:', ann_id, 'in img_id:', img_id)

            # delete other tags into caption
            pre_sub_len = len(new_caption)
            pre_caption = new_caption
            new_caption = re.sub(r"\s*\(<?box_id:[^>\),<]+>?(, <?box_id:[^>\),<]+>?)*\)", "", new_caption)
            if len(new_caption) != pre_sub_len:
                print('Img Id:', img_id, 'Caption:', pre_caption, '-> AutoFix Caption:', new_caption)
        
            data_item = {
                'id': img_id,
                'image': coco.loadImgs([img_id])[0]['file_name'],
                'conversations': [
                    {
                        "from": "human",
                        "value": "Please describe this image."
                    }
                ],
                "answer_template": new_caption,
                "objects": this_caption_objects,
                "task": 'ric',
            }
            
            with open(os.path.join(output_filtered_json), "a+") as f:
                f.write(json.dumps(data_item) + "\n")
        
    print(this_turn_pass_number)
    print(total_number)


if __name__ == '__main__':
    input_coco_json = '../../dataset/ReferringImageCaptioning/annotations/ric_instances_val2017.json'
    output_filtered_json = '../../dataset/ReferringImageCaptioning/processed/ric_instances_val2017.json'
    prepare_ric_annotations_with_api(input_coco_json, output_filtered_json)

    input_coco_json = '../../dataset/ReferringImageCaptioning/annotations/ric_instances_train2017.json'
    output_filtered_json = '../../dataset/ReferringImageCaptioning/processed/ric_instances_train2017.json'
    prepare_ric_annotations_with_api(input_coco_json, output_filtered_json)