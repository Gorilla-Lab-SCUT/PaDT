import os
import PIL
import math
import json
import torch
import random
import itertools
import deepspeed
import numpy as np
from copy import deepcopy
from packaging import version
from contextlib import contextmanager
from pycocotools import mask as cocomask
from transformers import AutoConfig, AutoProcessor
from accelerate import Accelerator, DeepSpeedPlugin
from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
from PaDT import PaDTForConditionalGeneration, VisonTextProcessingClass, parseVRTintoCompletion


def prepare_deepspeed(model: "Module", accelerator: "Accelerator"):
    """Prepares the model for DeepSpeed inference or evaluation by initializing it with the appropriate configuration.

    Adapted from accelerate:
    https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    """
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    stage = config_kwargs["zero_optimization"]["stage"]

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if hidden_size is not None and stage == 3:
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            # @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                }
            )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO
    # disabled (stage 0)
    if stage != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model

def load_model(model_path, local_rank):
    config = AutoConfig.from_pretrained(model_path)
    config.vl_decoder['use_mask_loss'] = True
      #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = PaDTForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank},
        config=config
    )

    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=HfTrainerDeepSpeedConfig(os.path.join(os.path.dirname(__file__), '../../', 'src/PaDT/local_scripts/zero3.json')))
    deepspeed_plugin.hf_ds_config.config['bf16']['enable']=True
    deepspeed_plugin.hf_ds_config.config['fp16']['enable']=False
    deepspeed_plugin.hf_ds_config.config['train_batch_size'] = int(os.getenv('WORLD_SIZE'))
    deepspeed_plugin.hf_ds_config.config['train_micro_batch_size_per_gpu'] = 1
    deepspeed_plugin.hf_ds_config.config['gradient_clipping'] = 1.0
    deepspeed_plugin.hf_ds_config.config['gradient_accumulation_steps'] = 1

    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)

    model_engine = prepare_deepspeed(model, accelerator)

    # default processer
    processor = AutoProcessor.from_pretrained(model_path)

    return model_engine, processor, accelerator

def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())

def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]

def remove_hooks(model) -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []

def add_hooks(model) -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")
    if version.parse(deepspeed.__version__) >= version.parse("0.16.4"):
        # Account for renaming in https://github.com/deepspeedai/DeepSpeed/pull/6847
        optimizer_offload._register_deepspeed_module(optimizer_offload.module)
    else:
        optimizer_offload._register_hooks_recursively(optimizer_offload.module)


@contextmanager
def unwrap_model_for_generation(
    model,
    accelerator,
    gather_deepspeed3_params=True,
):
    """
    Context manager to unwrap distributed or accelerated models for generation tasks.

    Args:
        model (`Union[DistributedDataParallel, DeepSpeedEngine]`):
            Model to be unwrapped.
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator instance managing the model.
        gather_deepspeed3_params (`bool`, *optional*, defaults to `True`):
            Whether to gather weights for DeepSpeed ZeRO Stage 3 models. If `False`, skips parameter gathering, which
            can be more memory-efficient but may lead to slower generation times.

    Yields:
        Unwrapped model.

    Example:
    ```python
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        generated_outputs = unwrapped_model.generate(input_ids)
    ```
    """
    unwrapped_model = accelerator.unwrap_model(model)
    is_gradient_checkpointing = unwrapped_model.is_gradient_checkpointing
    if is_gradient_checkpointing:
        unwrapped_model.gradient_checkpointing_disable()
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        yield unwrapped_model
    if is_gradient_checkpointing:
        unwrapped_model.gradient_checkpointing_enable()


def inference_dataset(
    model, dataset, processor, accelerator, output_dir, batch_size=1, datasetname='coco', suffix='', sample_num=500, seed=42
):
    random.seed(seed)

    all_number = math.ceil(len(dataset) / (accelerator.num_processes * batch_size)) * accelerator.num_processes * batch_size
    sub_dataset_idx = range(accelerator.process_index * batch_size, all_number, accelerator.num_processes * batch_size)

    with open(os.path.join(output_dir, f'{datasetname}_{accelerator.process_index}_pred_results_{suffix}.json'), 'w+') as f:
        f.write('')

    with open(os.path.join(output_dir, f'{datasetname}_{accelerator.process_index}_pred_comp_{suffix}.json'), 'w+') as f:
        f.write('')

    for idx in sub_dataset_idx:
        if int(os.getenv('LOCAL_RANK')) == 0:
            print(f"Processing {idx}... | Total: {len(dataset)}")
        
        with torch.no_grad():
            with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
                if idx < len(dataset):
                    input_ = dataset[idx: idx+batch_size]

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
            with open(os.path.join(output_dir, f'{datasetname}_{accelerator.process_index}_pred_comp_{suffix}.json'), 'a+') as f:
                for idx, completion in enumerate(completions):
                    f.write(json.dumps({'image_id': input_['id'][idx], 'completion': completion.replace('<|endoftext|>', '').replace('<|im_end|>', '')}) + '\n')

            if decoded_list['pred_boxes'].shape[0] == 0:
                continue
            pred_bboxes = []
            with open(os.path.join(output_dir, f'{datasetname}_{accelerator.process_index}_pred_results_{suffix}.json'), 'a+') as f:
                for box, score, label, mask, mask_hw, sample_idx in zip(decoded_list['pred_boxes'], decoded_list['pred_score'].sigmoid(), sum(labels, []), decoded_list['pred_mask'], torch.stack([decoded_list['pred_mask_valid_hw'][0], decoded_list['pred_mask_valid_hw'][1]], dim=-1), decoded_list['sample_idx']):
                    eval_box = (max(box[0].item() - box[2].item() / 2, 0), max(box[1].item() - box[3].item() / 2, 0), min(box[2].item(), 1), min(box[3].item(), 1))
                    w, h = images[sample_idx].size
                    eval_box = (round(eval_box[0] * w), round(eval_box[1] * h), round(eval_box[2] * w), round(eval_box[3] * h))

                    mask = torch.nn.functional.interpolate(mask[None, None, :mask_hw[0] * 4, :mask_hw[1] * 4], size=(h, w), mode='bilinear')[0, 0].sigmoid() > 0.5
                    mask = mask.cpu().numpy().astype(np.uint8)
                    rle = cocomask.encode(np.asfortranarray(mask))
                    rle['counts'] = rle['counts'].decode()
                    f.write(json.dumps({'image_id': input_['id'][sample_idx], 'score': score.item(), 'category': label, 'bbox': eval_box, 'mask': rle}) + '\n')
        