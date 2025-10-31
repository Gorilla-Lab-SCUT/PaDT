import re
import os
import json
import torch
import torch.distributed as dist
import deepspeed
from datasets import load_dataset, Dataset
from utils import load_model, infer_dataset
from PaDT import VisonTextProcessingClass


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
    image_folders = ['../../dataset/coco/val2017']  # Add the image root
    assert len(data_files) == len(image_folders), "Number of data files must match number of image folders"
    output_dir = '../outputs/coco'  # Add the output directory, default is logs
    
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

    infer_dataset(
        model=model,
        dataset=dataset,
        processor=processor,
        accelerator=accelerator,
        output_dir=output_dir,
        batch_size=16,
        datasetname='coco',
        suffix=suffix
    )
