from PaDT import PaDTForConditionalGeneration, VisonTextProcessingClass, parseVRTintoCompletion
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import os
import torch
import torch.distributed as dist
from PIL import Image
import cv2
import numpy as np
import re
import json


if __name__ == "__main__":

    MODEL_PATH="PaDT-MLLM/PaDT_Pro_3B"
    TEST_IMG_PATH="./imgs/000000368335.jpg"


    model = PaDTForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0}
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH
    )
    processor = VisonTextProcessingClass(processor, model.config.vision_config.spatial_merge_size)
    processor.prepare(model.model.embed_tokens.weight.shape[0])

    # Open Vocabulary Detection Prompt
    # prompt = """Please carefully check the image and detect the following objects: ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]."""

    # Referring Expression Comprehension Prompt
    # prompt = """Please carefully check the image and detect the object this sentence describes: "A red car"."""
    # prompt = """Please carefully check the image and detect the object this sentence describes: "A black car"."""
    # prompt = """Please carefully check the image and detect the object this sentence describes: "A white car"."""
    # prompt = """Please carefully check the image and detect the object this sentence describes: "The horse between two cars"."""
    prompt = """Please carefully check the image and detect the object this sentence describes: "The car is on the left side of the horse"."""

    # Referring Image Captioning Prompt
    # prompt = "Please describe this image."

    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": TEST_IMG_PATH
                }, {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(message)
    
    # Inference Tip: As COCO training images typically have a maximum side length of 640 (~644 = 28 * 23) pixels, 
    # the visual reference tokens in PaDT MLLM remain unaffected, but the decoder may be sensitive to larger scales. 
    # We suggest resizing inputs to a maximum side length of 644 pixels for optimal inference results.
    MAX_SIDE = 644
    new_image_inputs = []
    for image in image_inputs:
        im_w, im_h = image.size
        scale = MAX_SIDE / max(im_w, im_h)
        new_w, new_h = int(im_w * scale), int(im_h * scale)
        new_image_inputs.append(image.resize((new_w, new_h), Image.Resampling.LANCZOS))

    prompt_inputs = processor(
        text=[text],
        images=new_image_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        add_special_tokens=False
    ).to("cuda:0")

    with torch.inference_mode():
        # !important
        prompt_inputs["input_ids"] = processor.assign_to_global_vrt_id(prompt_inputs["input_ids"], prompt_inputs['image_grid_thw'])
        generate_returned_result = model.generate(**prompt_inputs, use_cache=True, max_new_tokens=1024, do_sample=False,
                                        output_hidden_states=True, return_dict_in_generate=True)
        # !important
        prompt_completion_ids = processor.assign_to_local_vrt_id(generate_returned_result['sequences'], prompt_inputs['image_grid_thw'])

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # extract Visual Reference Tokens within the sequence and their corresponding hidden features
        completions, feats, labels, vrts, vrts_feats = parseVRTintoCompletion(processor, completion_ids, generate_returned_result['hidden_states'], torch.Tensor([False]))

        with open('./outputs/demo/completion.txt', 'w') as f:
            f.write('Prompt: ' + text + '\n')
            f.write('Completion: ' + completions[0] + '\n')
        
        low_res_image_embeds = generate_returned_result.past_image_embeds
        high_res_image_embeds = generate_returned_result.past_high_res_image_embeds
        visual_pe = generate_returned_result.past_visual_pe
        # decode the visual task output
        decoded_list = model.vl_decode(feats, low_res_image_embeds, high_res_image_embeds, prompt_inputs['image_grid_thw'], visual_pe)
        pred_boxes = decoded_list['pred_boxes']
        pred_scores = decoded_list['pred_score'].sigmoid()
        pred_labels = sum(labels, [])
        pred_vrts = sum(vrts, [])
        pred_masks = decoded_list['pred_mask']
        pred_mask_valid_hws = torch.stack([decoded_list['pred_mask_valid_hw'][0], decoded_list['pred_mask_valid_hw'][1]], dim=-1)
        box_2_sample_idx = decoded_list['sample_idx']

    # draw figure
    image = cv2.imread(TEST_IMG_PATH)
    im_h, im_w = image.shape[:2]
    MAX_SIDE = 644
    scale = MAX_SIDE / max(im_w, im_h)
    im_w, im_h = int(im_w * scale), int(im_h * scale)
    image = cv2.resize(image, (im_w, im_h))

    vrt_seg = np.zeros_like(image)
    mask_seg = np.zeros_like(image)

    resized_h, resized_w = round(im_h / 28) * 28, round(im_w / 28) * 28
    patch_h, patch_w = round(im_h / 28), round(im_w / 28)
    vrt_seg = cv2.resize(vrt_seg, (resized_w, resized_h))
    mask_seg = cv2.resize(mask_seg, (resized_w, resized_h))
    image = cv2.resize(image, (resized_w, resized_h))
    
    colors = np.array([
        [255, 0, 0],
        [255, 165, 0],
        [255, 215, 0],
        [127, 255, 0],
        [0, 0, 255]
    ])
    r, g, b = np.split(colors, 3, axis=-1)
    colors = np.concatenate([b, g, r], axis=-1)

    for idx, (box, score, label, mask, mask_hw, sample_idx, vrt) in enumerate(zip(pred_boxes, pred_scores, pred_labels, pred_masks, pred_mask_valid_hws, box_2_sample_idx, pred_vrts)):
        ## box
        # predbox: [cx, cy, w, h] -> [x, y, w, h], value: [0, 1]
        eval_box = (max(box[0].item() - box[2].item() / 2, 0), max(box[1].item() - box[3].item() / 2, 0), min(box[2].item(), 1), min(box[3].item(), 1))
        # scale 0~1 to 0~H/W
        eval_box = (round(eval_box[0] * resized_w), round(eval_box[1] * resized_h), round(eval_box[2] * resized_w), round(eval_box[3] * resized_h))
        cv2.rectangle(image, (eval_box[0], eval_box[1]), (eval_box[0] + eval_box[2], eval_box[1] + eval_box[3]), (0, 0, 255), 2)
        # draw box label
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if eval_box[1] - text_height - 5 < 0:
            text_start_point = eval_box[1] + text_height + 5 
            end_point =  eval_box[1] + text_height + 5
        else:
            text_start_point = eval_box[1]
            end_point = eval_box[1]
        image = cv2.rectangle(image, (round(max(eval_box[0], 0)), round(min(text_start_point - text_height - 5, resized_h))), (round(max(eval_box[0] + text_width + 5, 0)), round(min(end_point, resized_h))), (64, 64, 255), -1)
        image = cv2.putText(image, label, (round(eval_box[0]), round(text_start_point - baseline)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        ## mask
        mask = (torch.nn.functional.interpolate(mask[None, None, :mask_hw[0] * 4, :mask_hw[1] * 4], size=(resized_h, resized_w), mode='bilinear')[0, 0].sigmoid() > 0.5).cpu().numpy()
        color = colors[idx % 5]
        mask_seg[mask] = color
        mask = mask.astype(np.uint8)
        # rle = cocomask.encode(np.asfortranarray(mask))
        
        ## VRT
        vrt_idxs = re.findall(r'<\|VRT_(\d+)\|>', vrt)
        for idx, vrt_idx in enumerate(vrt_idxs):
            vrt_x, vrt_y = int(vrt_idx) % patch_w, int(vrt_idx) // patch_w
            vrt_seg[int(vrt_y * 28): int((vrt_y + 1) * 28), int(vrt_x * 28): int((vrt_x + 1) * 28), :] = colors[idx % 5]
            cv2.putText(vrt_seg, '%s' % (vrt_idx), (vrt_x * 28 + 0, vrt_y * 28 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imwrite('./outputs/demo/mask_seg.png', mask_seg)
    cv2.imwrite('./outputs/demo/pred_box.png', image)
    cv2.imwrite('./outputs/demo/vrt_seg.png', vrt_seg * 0.6 + image * 0.4)