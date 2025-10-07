# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import os
import cv2
import json
import torch
import PIL.Image
import numpy as np
from packaging import version
from collections import defaultdict
from datasets import Dataset, IterableDataset
from typing import Any, Callable, Optional, Union, Sized

import deepspeed
import transformers
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from pycocotools import mask
from torch.utils.data import Sampler
from torchvision.ops.boxes import box_area
from torch.utils.data.dataloader import DataLoader
from accelerate.utils import is_peft_model, set_seed
from PaDT import PaDTForConditionalGeneration, VisonTextProcessingClass, parseVRTintoCompletion
from PaDT.trainer import PaDTSFTConfig, PaDTScriptArguments, PaDTModelConfig

if is_wandb_available():
    import wandb


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
        num_processes: int = 1,
        gradient_accumulation_steps: int = 1
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.num_processes = num_processes
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes) // self.batch_size * self.batch_size, self.batch_size)]
        for chunk in indexes:
            for _ in range(self.repeat_count):
                for accumultaion_step_idx in range(self.gradient_accumulation_steps):
                    accumulation_chunk = chunk[accumultaion_step_idx::self.gradient_accumulation_steps]
                    chunk_idx = torch.arange(0, len(accumulation_chunk), dtype=torch.long)[:, None].repeat_interleave(self.mini_repeat_count, -1).view(-1)
                    for index in chunk_idx:
                        yield accumulation_chunk[index]

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class PaDTSFTTrainer(Trainer):

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        script_args: PaDTScriptArguments = None,
        training_args: PaDTSFTConfig = None,
        model_args: PaDTModelConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        **kwargs
    ):
        # Args
        assert script_args is not None and training_args is not None and model_args is not None
        self.script_args = script_args
        args = training_args
        self.model_args = model_args

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = model_args.attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = 'bfloat16'
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )

        model_cls = PaDTForConditionalGeneration
        model_config, model_init_kwargs = model_cls.config_class.from_pretrained(model_id, return_unused_kwargs=True, **model_init_kwargs)
        model_config.update({
            # vl_decoder config
            "vl_decoder": {
                "name": "PaDTDecoder",
                "attn_implementation": "flash_attention_2",
                "hidden_size": 1280,
                "intermediate_size": 3420,
                "llm_hidden_state": 2048,
                "num_heads": 16,
                "spatial_merge_size": 2,
                "use_mask_loss": args.use_mask_loss,
            },
            "use_visual_prototype_projection": args.use_visual_prototype_projection
        })
        model = model_cls.from_pretrained(model_id, config=model_config, **model_init_kwargs)

        # Freeze vision modules
        if model_args.freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in ['visual']):
                    p.requires_grad = False
        
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        processing_class = AutoProcessor.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
        for component, processing_keyword in [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]:
            processing_component = getattr(processing_class, component, processing_class)
            setattr(processing_component, processing_keyword, getattr(script_args, processing_keyword))
        if getattr(processing_class, "tokenizer",  None) is not None:
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        else:
            assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
            pad_token_id = processing_class.pad_token_id
        processing_class = VisonTextProcessingClass(processing_class)
        
        # Data collator
        def data_collator(features):
            return features

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # align processing_class.tokenizer.vocab_size with self.model.model.embed_tokens vocab_size 
        with deepspeed.zero.GatheredParameters([model.model.embed_tokens.weight], enabled=True):
            # model.model.embed_tokens.shape
            self.model_embed_token_size = model.model.embed_tokens.weight.shape[0]
        processing_class.prepare(self.model_embed_token_size)

        set_seed(args.seed, device_specific=True)


    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            if getattr(model, "language_model", None) is not None:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False
            else:
                model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    @staticmethod
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)
    
    @staticmethod
    def box_iou(boxes1, boxes2):
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union

    def generalized_box_iou(self, boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/

        The boxes should be in [x0, y0, x1, y1] format

        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union = self.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area, iou

    def dice_loss(self, inputs, targets, loss_mask):
        # inputs: preds [N, H, W]
        # targets: gts [N, H, W]
        # loss_mask: whether the value is valid. [N, H, W]
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        loss_mask = loss_mask.flatten(1)
        numerator = 2 * (inputs * targets * loss_mask).sum(dim=-1)
        denominator = (inputs * loss_mask).sum(dim=-1) + (targets * loss_mask).sum(dim=-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / ((loss_mask.sum(dim=-1) > 0) + 1e-5).sum()
    
    def sigmoid_focal_loss(self, inputs, targets, loss_mask, alpha = 0.25, gamma = 2):
        # inputs: preds [N, H, W]
        # targets: gts [N, H, W]
        # loss_mask: whether the value is valid. [N, H, W]
        prob = inputs.sigmoid()
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
        return ((loss * loss_mask).sum(dim=[1, 2]) / (loss_mask.sum(dim=[1, 2]) + 1e-5)).sum() / ((loss_mask.sum(dim=[1, 2]) > 0) + 1e-5).sum()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The PaDTTrainer does not support returning outputs")

        prompt_text = [self.processing_class.apply_chat_template(i['prompt'], tokenize=False, add_generation_prompt=True) for i in inputs]
        
        images = []
        completions = []
        solutions = []
        for idx, x in enumerate(inputs):
            # input_images
            assert len(x['image_path']) == 1, "current support only an image per sample"

            for img in x['image_path']:
                image = PIL.Image.open(img)
                try:
                    w, h = image.size
                    if w < 28 or h < 28:
                        if w < h:
                            new_w = 28
                            new_h = int(h * (28 / w))
                        else:
                            new_h = 28
                            new_w = int(w * (28 / h))
                    image = image.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                except:
                    pass
                images.append(image)
            
            # completion
            im_w, im_h = image.size
            patch_w, patch_h = round(im_w / 28), round(im_h / 28)

            solution = x['solution']
            completion = solution['text']
            pattern = r'(<\|Obj_(\d+)\|>)'
            obj_in_completion = re.findall(pattern, completion)
            obj_strs = [i[0] for i in obj_in_completion]
            objs = [solution['objects'][int(i[1])] for i in obj_in_completion]
            pattern_without_matching = r'<\|Obj_\d+\|>'
            completion_parts = re.split(pattern_without_matching, completion)
            
            completion_with_vrt = completion_parts[0]
            new_objs = []
            for obj_str, completion_part, obj in zip(obj_strs, completion_parts[1:], objs):
                obj_ = obj.copy()
                selected_patches = np.array(obj_['patches'])
                if self.args.random_select_patch_num < 0:
                    pick_patch = selected_patches.copy()
                elif self.args.random_select_patch is False:
                    selected_patches_x, selected_patches_y = selected_patches % patch_w, selected_patches // patch_w
                    
                    left_patches_m = selected_patches_x == selected_patches_x.min()
                    right_patches_m = selected_patches_x == selected_patches_x.max()
                    top_patches_m = selected_patches_y == selected_patches_y.min()
                    bottom_patches_m = selected_patches_y == selected_patches_y.max()
                    centre_patches_m = (left_patches_m + right_patches_m + top_patches_m + bottom_patches_m) == 0                        
                    
                    left_patches, right_patches, top_patches, bottom_patches, centre_patches = \
                        selected_patches[left_patches_m], \
                        selected_patches[right_patches_m], \
                        selected_patches[top_patches_m], \
                        selected_patches[bottom_patches_m], \
                        selected_patches[centre_patches_m]
                    if centre_patches_m.sum() == 0:
                        centre_patches = selected_patches
                        
                    pick_patch = np.array([np.random.choice(centre_patches), np.random.choice(left_patches), np.random.choice(top_patches), np.random.choice(right_patches), np.random.choice(bottom_patches)])
                else:
                    if selected_patches.shape[0] < self.args.random_select_patch_num:
                        pick_patch = np.random.choice(selected_patches, self.args.random_select_patch_num, replace=True)
                    else:
                        pick_patch = np.random.choice(selected_patches, self.args.random_select_patch_num, replace=False)
            
                obj_['picked'] = pick_patch
                new_objs.append(obj_)
                completion_with_vrt += self.processing_class.pid2vrt(pick_patch) + completion_part

            solutions.append({
                'text': solution['text'],
                'objects': new_objs
            })
            completions.append(completion_with_vrt + self.processing_class.tokenizer.eos_token)

        # tokenizing
        prompt_inputs = self.processing_class(
            text=prompt_text,
            images=images,
            return_tensors='pt',
            padding=True,
            padding_side='left',
            add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        batch_size = prompt_ids.size(0)
        prompt_length = prompt_ids.size(1)
        multimodal_inputs = {
            'image_grid_thw': prompt_inputs['image_grid_thw'],
            'pixel_values': prompt_inputs['pixel_values']
        }
        
        completion_inputs = self.processing_class(
            text=completions,
            return_tensors='pt',
            padding=True,
            padding_side='right',
            add_special_tokens=False
        )
        completion_inputs = super()._prepare_inputs(completion_inputs)
        completion_ids, completion_mask = completion_inputs["input_ids"], completion_inputs["attention_mask"]

        # prepare for Robust Per-token Cross-Entropy Loss.
        loss_masks = []
        gt_bboxes = []
        vision_patch_nums = torch.nn.functional.pad((multimodal_inputs['image_grid_thw'].cumprod(-1)[:, -1] // (self.model.config.vision_config.spatial_merge_size ** 2)).cumsum(-1), (1, 0), 'constant', 0)
        all_vision_patch_nums = vision_patch_nums[-1]
        for sol, vpn in zip(solutions, vision_patch_nums[:-1]):
            for obj in sol['objects']:
                this_object_loss_mask = torch.zeros((obj['picked'].shape[0], all_vision_patch_nums), device=self.accelerator.device, dtype=torch.bool)
                this_object_loss_mask[:, vpn.item() + np.array(obj['patches'])] = True
                this_object_loss_mask[np.arange(obj['picked'].shape[0]), vpn.item() + obj['picked']] = False
                loss_masks.append(this_object_loss_mask)
                # obj['bbox']: x1, y1, x2, y2. Value in [0, 1].
                gt_bboxes.append(obj['bbox'])

        loss_masks = torch.cat(loss_masks, dim=0)
        loss_masks = torch.nn.functional.pad(loss_masks, (self.model_embed_token_size, 0), 'constant', False)
        gt_bboxes = torch.Tensor(gt_bboxes).to(self.accelerator.device).to(torch.bfloat16)
        if len(gt_bboxes.shape) == 1:
            gt_bboxes = gt_bboxes.unsqueeze(dim=-1).repeat_interleave(4, dim=-1)

        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        input_ids = self.processing_class.assign_to_global_vrt_id(input_ids, multimodal_inputs['image_grid_thw'])

        # Get the current policy's log probabilities
        model_output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **multimodal_inputs)

        logits = model_output.logits[:, prompt_length-1:-1, :]  # (B, L, V)
        input_ids = input_ids[:, prompt_length:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        if self.args.use_sft_vp_mask:
            visual_patch_mask = input_ids >= self.model_embed_token_size
            logits[visual_patch_mask] = logits[visual_patch_mask].masked_fill(loss_masks, float('-inf'))
        
        # decode to bbox
        hidden_states = torch.stack(model_output.hidden_states, dim=1)[:, -1:, prompt_length-1:-1].permute(2, 1, 0, 3).unsqueeze(dim=-2).contiguous() # [BS, Layers, N, Dim] -> [N, Layers, BS, 1, D]
        completions, feats, labels, vps, vps_feats = parseVRTintoCompletion(self.processing_class, completion_ids, hidden_states, torch.tensor([False] * batch_size), model_output.past_image_embeds, multimodal_inputs['image_grid_thw']) # hidden_states: [N, Layers, BS, D]
        low_res_image_embeds = model_output.past_image_embeds
        high_res_image_embeds = model_output.past_high_res_image_embeds
        visual_pe = model_output.past_visual_pe
        
        # warm up stage: using visual prototype rather than hidden features to feed into decoder.
        if self.state.epoch < (self.state.num_train_epochs / 4) and self.state.global_step < 300 and self.args.use_warm_up:
            feats = vps_feats
        decoded_list = model(feats, low_res_image_embeds, high_res_image_embeds, multimodal_inputs['image_grid_thw'], visual_pe, is_main=False)
        del model_output

        if self.args.use_mask_loss:
            gt_mask = torch.zeros_like(decoded_list['pred_mask'])
            loss_mask = torch.zeros_like(decoded_list['pred_mask'])

            obj_idx = 0
            for sol in solutions:
                for obj in sol['objects']:
                    if 'rle' in obj:
                        gt_m = mask.decode(obj['rle'])
                        mask_h, mask_w = decoded_list['pred_mask_valid_hw'][0][obj_idx].item(), decoded_list['pred_mask_valid_hw'][1][obj_idx].item()
                        resized_gt_m = torch.from_numpy(cv2.resize(gt_m.astype(np.float32()), (mask_w * 4, mask_h * 4)) > 0.5).to(gt_mask.dtype).to(gt_mask.device)
                        gt_mask[obj_idx, :mask_h * 4, :mask_w * 4] = resized_gt_m
                        loss_mask[obj_idx, :mask_h * 4, :mask_w * 4] = 1.0
                    obj_idx += 1
            mask_loss = self.dice_loss(decoded_list['pred_mask'], gt_mask, loss_mask) + self.sigmoid_focal_loss(decoded_list['pred_mask'], gt_mask, loss_mask)
            self._metrics['mask_loss'].append(self.accelerator.gather_for_metrics(mask_loss).mean().item())
        else:
            mask_loss = 0.

        # token loss
        logit_log_probs = logits.log_softmax(dim=-1)
        token_log_prob = torch.gather(logit_log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        per_token_loss = -token_log_prob
        sft_loss = ((per_token_loss * completion_mask).sum(dim=-1) / (completion_mask.sum(dim=-1) + 1e-4)).to(logits.dtype)
        self._metrics['sft_loss'].append(self.accelerator.gather_for_metrics(sft_loss).mean().item())
        
        if self.args.use_bbox_loss:
            # bbox loss
            pred_bboxes = decoded_list['pred_boxes']  # num_bbox, 4 [cx, cy, w, h]
            # gt_bboxes # num_bbox, 4 [x1, y1, x2, y2]
            num_bboxes = gt_bboxes.shape[0]
            giou, iou = self.generalized_box_iou(self.box_cxcywh_to_xyxy(pred_bboxes), gt_bboxes)
            giou = torch.diag(giou).to(pred_bboxes.dtype)
            bbox_loss = 1. - giou.sum() / (num_bboxes + 1e-4)
            bbox_loss += torch.nn.functional.l1_loss(pred_bboxes, self.box_xyxy_to_cxcywh(gt_bboxes), reduction='none').sum() / (num_bboxes + 1e-4)
            self._metrics['bbox_loss'].append(self.accelerator.gather_for_metrics(bbox_loss).mean().item())
            self._metrics['iou'].append(self.accelerator.gather_for_metrics(torch.diag(iou).sum() / (num_bboxes + 1e-4)).mean().item())
            self._metrics['giou'].append(self.accelerator.gather_for_metrics(giou.sum() / (num_bboxes + 1e-4)).mean().item())
        else:
            bbox_loss = 0.

        if self.args.use_bbox_loss and self.args.use_score_loss:
            # score loss
            pred_score = decoded_list['pred_score'].sigmoid() * 2. - 1.  # [-1, 1]
            score_loss = torch.nn.functional.mse_loss(pred_score, giou.unsqueeze(1).detach(), reduction='sum') / (num_bboxes + 1e-4)
            self._metrics['score_loss'].append(self.accelerator.gather_for_metrics(score_loss).mean().item())
        else:
            score_loss = 0.

        loss = sft_loss.mean() + bbox_loss + score_loss + mask_loss
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    
    def _get_train_sampler(self) -> Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=1,
            batch_size=effective_batch_size,
            repeat_count=1,
            seed=self.args.seed,
            num_processes=self.accelerator.num_processes,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=1,
            seed=self.args.seed,
            num_processes=self.accelerator.num_processes,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps
        )
