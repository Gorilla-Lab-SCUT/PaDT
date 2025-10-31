import re
import os
import PIL
import math
import json
import torch
import torch.distributed as dist
import random
import deepspeed
import numpy as np
from copy import deepcopy
from datasets import load_dataset, Dataset
from utils import load_model, unwrap_model_for_generation
from PaDT import VisonTextProcessingClass, parseVRTintoCompletion


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")


def eval_od_r1(
    model, dataset, processor, accelerator, output_dir, batch_size=1, suffix='', sample_num=500, seed=42
):
    random.seed(seed)

    all_number = math.ceil(len(dataset) / (accelerator.num_processes * batch_size)) * accelerator.num_processes * batch_size
    sub_dataset_idx = range(accelerator.process_index * batch_size, all_number, accelerator.num_processes * batch_size)

    with open(os.path.join(output_dir, f'coco_{accelerator.process_index}_pred_results_{suffix}.json'), 'w+') as f:
        f.write('')

    with open(os.path.join(output_dir, f'coco_{accelerator.process_index}_pred_comp_{suffix}.json'), 'w+') as f:
        f.write('')

    for idx in sub_dataset_idx:
        if int(os.getenv('LOCAL_RANK')) == 0:
            print(f"Processing {idx}... | Total: {len(dataset)}")
        
        with torch.no_grad():
            with unwrap_model_for_generation(model, accelerator) as unwrapped_model:

                if idx < len(dataset):
                    input_ = dataset[idx: idx+batch_size]

                    # prompts_text = [maybe_apply_chat_template({'prompt': prompt}, processor)['prompt'] for prompt in input_['prompt']]
                    prompts_text = processor.apply_chat_template(input_['prompt'], tokenize=False, add_generation_prompt=True)

                    imgs = [PIL.Image.open(p) for sample in input_['image_path'] for p in sample]
                    
                    images = []
                    for img in imgs:
                        try:
                            # Ensure minimum dimensions of 28 pixels
                            w, h = img.size
                            if w < 28 or h < 28:
                            # Calculate new dimensions maintaining aspect ratio
                                if w < h:
                                    new_w = 28
                                    new_h = int(h * (28/w))
                                else:
                                    new_h = 28
                                    new_w = int(w * (28/h))
                            img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                        except:
                            pass
                        images.append(img)
                    
                    prompt_inputs = processor(
                        text=prompts_text,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                        add_special_tokens=False,
                    )
                    prompt_inputs = prompt_inputs.to(accelerator.device)
                    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

                    prompt_inputs["input_ids"] = processor.assign_to_global_vrt_id(prompt_inputs["input_ids"], prompt_inputs['image_grid_thw'])
                    generate_returned_result = unwrapped_model.generate(
                        **prompt_inputs, 
                        use_cache=True, max_new_tokens=1024, do_sample=False, output_hidden_states=True, return_dict_in_generate=True, synced_gpus=False
                    )
                    prompt_length = prompt_ids.size(1)
                    prompt_completion_ids = processor.assign_to_local_vrt_id(generate_returned_result['sequences'], prompt_inputs['image_grid_thw'])
                    completion_ids = prompt_completion_ids[:, prompt_length:]
                    completions, feats, labels, vrts, vrts_feats = parseVRTintoCompletion(processor, completion_ids, generate_returned_result['hidden_states'], torch.Tensor([False] * batch_size))

                    low_res_image_embeds = generate_returned_result.past_image_embeds
                    high_res_image_embeds = generate_returned_result.past_high_res_image_embeds
                    visual_pe = generate_returned_result.past_visual_pe
                    
                    decoded_list = unwrapped_model.vl_decode(feats, low_res_image_embeds, high_res_image_embeds, prompt_inputs['image_grid_thw'], visual_pe)
        
        if idx < len(dataset):
            with open(os.path.join(output_dir, f'coco_{accelerator.process_index}_pred_comp_{suffix}.json'), 'a+') as f:
                for idx, completion in enumerate(completions):
                    f.write(json.dumps({'image_id': input_['id'][idx], 'completion': completion.replace('<|endoftext|>', '').replace('<|im_end|>', '')}) + '\n')

            if decoded_list['pred_boxes'].shape[0] == 0:
                continue
            pred_bboxes = []
            with open(os.path.join(output_dir, f'coco_{accelerator.process_index}_pred_results_{suffix}.json'), 'a+') as f:
                for box, score, label, mask, mask_hw, sample_idx in zip(decoded_list['pred_boxes'], decoded_list['pred_score'].sigmoid(), sum(labels, []), decoded_list['pred_mask'], torch.stack([decoded_list['pred_mask_valid_hw'][0], decoded_list['pred_mask_valid_hw'][1]], dim=-1), decoded_list['sample_idx']):
                    eval_box = (max(box[0].item() - box[2].item() / 2, 0), max(box[1].item() - box[3].item() / 2, 0), min(box[2].item(), 1), min(box[3].item(), 1))
                    w, h = images[sample_idx].size
                    eval_box = (round(eval_box[0] * w), round(eval_box[1] * h), round(eval_box[2] * w), round(eval_box[3] * h))
                    f.write(json.dumps({'image_id': input_['id'][sample_idx], 'score': score.item(), 'category': label, 'bbox': eval_box}) + '\n')
            
        

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
        suffix = sys.argv[2]
    else:
        checkpoint = 'PaDT-MLLM/PaDT_Pro_3B'
        suffix = 'padt_pro_3b'


    model_path = f'{checkpoint}'  # Add the path to the model
    data_files = ['PaDT-MLLM/COCO/instances_val2017.json']  # Add the data root
    image_folders = ['/home/yongyi/data/dataset/coco/val2017']  # Add the image root
    assert len(data_files) == len(image_folders), "Number of data files must match number of image folders"
    output_dir = '../logs'  # Add the output directory, default is logs
    
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        if os.path.exists(data_file) is False:
            data = load_dataset("/".join(data_file.split("/")[:-1]), data_files=data_file.split("/")[-1])['train'].to_list()
        else:
            data = [json.loads(i) for i in open(data_file, 'r').readlines()]
        
        for item in data:
            if 'image' in item:
                if isinstance(item['image'], str):
                    # Store image path instead of loading the image
                    item['image_path'] = [os.path.join(image_folder, item['image'])]
                    del item['image'] # remove the image column so that it can be loaded later
                elif isinstance(item['image'], list):
                    # if the image is a list, then it is a list of images (for multi-image input)
                    item['image_path'] = [os.path.join(image_folder, image) for image in item['image']]
                    del item['image'] # remove the image column so that it can be loaded later
                else:
                    raise ValueError(f"Unsupported image type: {type(item['image'])}")
            
                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                
                del item['answer_template'], item['objects']
                del item['conversations']
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        assert all(os.path.exists(p) for p in example['image_path']), f"Image paths do not exist: {example['image_path']}"
        # Don't load image here, just store the path

        return {
            'image_path': [p for p in example['image_path']],  # Store path instead of loaded image
            'problem': example['problem'],
            'solution': None,
            'prompt': [{
                'role': 'user',
                'content': [
                    *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                    {'type': 'text', 'text': example['problem']}
                ]
            }]
        }

    dataset = dataset.map(make_conversation_from_jsonl, num_proc=16)
    
    model, processor, accelerator = load_model(model_path, local_rank)
    processor = VisonTextProcessingClass(processor)

    # align processing_class.tokenizer.vocab_size with self.model.model.embed_tokens vocab_size 
    with deepspeed.zero.GatheredParameters([model.model.embed_tokens.weight], enabled=True):
        # model.model.embed_tokens.shape
        model_embed_token_size = model.model.embed_tokens.weight.shape[0]
    processor.prepare(model_embed_token_size)

    eval_od_r1(
        model=model,
        dataset=dataset,
        processor=processor,
        accelerator=accelerator,
        output_dir=output_dir,
        batch_size=16,
        suffix=suffix
    )
