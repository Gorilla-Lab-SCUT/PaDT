import os
import numpy as np
from PIL import Image
import random
from pycocotools import mask
from pycocotools.coco import COCO
from refer import REFER
import matplotlib.pyplot as plt
from copy import deepcopy
import cv2
import json

os.makedirs('../../dataset/RefCOCO/processed', exist_ok=True)
data_path = "../../dataset/RefCOCO"
output_json = '../../dataset/RefCOCO/processed/{ds}_{split}.json'

for split in ['train', 'val', 'testA', 'testB']:
    for ds in ["refcoco", "refcoco+", "refcocog"]:
        if ds == 'refcocog' and split == 'testA':
            split = 'test'
        elif ds == 'refcocog' and split == 'testB':
            continue

        with open(output_json.format(ds=ds, split=split), 'w') as f:
            f.write("")
        
        print('Processing:', ds)

        if ds == "refcocog":
            splitBy = "umd"
        else:
            splitBy = "unc"

        refer_api = REFER(data_path, ds, splitBy)

        ref_ids_train = refer_api.getRefIds(split=split)
        images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)   
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train) 

        for num, image_info in zip(range(len(loaded_images)), loaded_images):
            refs = refer_api.imgToRefs[image_info["id"]]
            
            image_w, image_h = image_info['width'], image_info['height']

            resized_h, resized_w = int(round(image_h / 28) * 28), int(round(image_w / 28) * 28)
            
            for kk in range(len(refs)):
                item = {}
                item["id"] = image_info["id"]

                item["image"] = image_info["file_name"]

                ref = refs[kk]
                sentences = ref['sentences']

                ann = refer_api.refToAnn[ref['ref_id']]

                if type(ann["segmentation"][0]) == list:  # polygon
                    rle = mask.frPyObjects(
                        ann["segmentation"],
                        image_info["height"],
                        image_info["width"],
                    )
                else:
                    rle = ann["segmentation"]
                    for i in range(len(rle)):
                        if not isinstance(rle[i]["counts"], bytes):
                            rle[i]["counts"] = rle[i]["counts"].encode()

                m = mask.decode(rle)
                m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
                m[m >= 1] = 1
                m = m.astype(np.float32)

                resized_m = cv2.resize(m * 255, (resized_w, resized_h))
                patch_mask = resized_m.reshape(resized_h // 28, 28, resized_w // 28, 28).transpose(0, 2, 1, 3).mean(axis=-1).mean(axis=-1) > 255 / 28
                if patch_mask.sum() < 1:
                    print('skip one sample since the mask area is too small:', image_info)
                    continue

                bbox_x, bbox_y, bbox_w, bbox_h = refer_api.getRefBox(ref['ref_id'])

                for sentence in sentences:
                    item['conversations'] = [
                        {
                            'from': 'human',
                            'value': "Please carefully check the image and detect the object this sentence describes: \"" + sentence['sent'] + "\"."
                        }
                    ]
                    save_rle = mask.encode(m.astype(np.uint8))
                    save_rle['counts'] = save_rle['counts'].decode()

                    item['task'] = 'refering'
                    item['answer_template'] = "The \"%s\" refers to <|Obj_0|> in this image." % sentence['sent']
                    item['objects'] = [
                        {
                            'patches': np.where(patch_mask.reshape(-1))[0].tolist(),
                            'bbox': [
                                bbox_x / image_w,
                                bbox_y / image_h,
                                (bbox_x + bbox_w) / image_w,
                                (bbox_y + bbox_h) / image_h
                            ],
                            'iscrowd': ann['iscrowd'],
                            'area': ann['area'],
                            'rle': save_rle,
                            'label': sentence['sent'],
                        }
                    ]
                
                    with open(output_json.format(ds=ds, split=split), 'a+') as f:
                        f.write(json.dumps(item) + '\n')