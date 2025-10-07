import os
import re
import PIL
import json
import math
import torch
import pathlib
from typing import Tuple
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset

from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer, HfArgumentParser

from PaDT import PaDTForConditionalGeneration
from PaDT.trainer import PaDTSFTConfig, PaDTScriptArguments, PaDTModelConfig
from PaDT.trainer import PaDTSFTTrainer
from PaDT.utils.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_torch_load
monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_torch_load()



def main(script_args, training_args, model_args):
    # Load the JSONL datasets
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    assert len(data_files) == len(image_folders), "Number of data files must match number of image folders"

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
            item['solution'] = {
                'text': item['answer_template'],
                'objects': item['objects']
            }
            del item['answer_template'], item['objects']
            del item['conversations']
            all_data.append(item)

    dataset = Dataset.from_list(all_data)
    
    def make_conversation(example):
        assert all(os.path.exists(p) for p in example['image_path']), f"Image paths do not exist: {example['image_path']}"
        # Don't load image here, just store the path

        return {
            'image_path': [p for p in example['image_path']],  # Store path instead of loaded image
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': [{
                'role': 'user',
                'content': [
                    *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                    {'type': 'text', 'text': example['problem']}
                ]
            }]
        }

    # Map the conversations
    dataset = dataset.map(make_conversation, num_proc=16)

    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    trainer_cls = PaDTSFTTrainer
    print("using trainer:", trainer_cls.__name__)

    # Initialize the SFT trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        script_args=script_args,
        training_args=training_args,
        model_args=model_args,
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and (training_args.resume_from_checkpoint == 'true' or training_args.resume_from_checkpoint is True):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((PaDTScriptArguments, PaDTSFTConfig, PaDTModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    main(script_args, training_args, model_args)
