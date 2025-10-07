import os
import copy 
import deepspeed
from typing import Optional, Union, Callable, List, Tuple

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_emb, Qwen2RMSNorm, flash_attn_varlen_func


class QwenDecoderFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_cu_seqlens: torch.Tensor,
        key_cu_seqlens: torch.Tensor,
        query_pos_emb: torch.Tensor,
        key_pos_emb: torch.Tensor,
        is_rotary_pos_emb: Tuple[bool] = (True, True)
    ) -> torch.Tensor:
        q = self.q_proj(query) if is_rotary_pos_emb[0] else self.q_proj(query + query_pos_emb)
        k = self.k_proj(key) if is_rotary_pos_emb[1] else self.k_proj(key + key_pos_emb)
        v = self.v_proj(key)

        q = q.reshape(query.shape[0], self.num_heads, -1)
        k = k.reshape(key.shape[0], self.num_heads, -1)
        v = v.reshape(key.shape[0], self.num_heads, -1)

        if is_rotary_pos_emb[0]:
            cos, sin = query_pos_emb
            cos = cos.chunk(2, dim=-1)[0].contiguous()
            sin = sin.chunk(2, dim=-1)[0].contiguous()
            q = q.unsqueeze(0)
            q = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
            q = q.squeeze(0)
        if is_rotary_pos_emb[1]:
            cos, sin = key_pos_emb
            cos = cos.chunk(2, dim=-1)[0].contiguous()
            sin = sin.chunk(2, dim=-1)[0].contiguous()
            k = k.unsqueeze(0)
            k = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
            k = k.squeeze(0)

        max_seqlen_q = (query_cu_seqlens[1:] - query_cu_seqlens[:-1]).max().item()
        max_seqlen_k = (key_cu_seqlens[1:] - key_cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, query_cu_seqlens, key_cu_seqlens, max_seqlen_q, max_seqlen_k).reshape(
            query.shape[0], -1
        )

        attn_output = self.proj(attn_output)
        return attn_output


QwenDecoder_VISION_ATTENTION_CLASSES = {
    "flash_attention_2": QwenDecoderFlashAttention2,
}


class QwenDecoderBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa", update_memory: bool = False) -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config['hidden_size'], eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config['hidden_size'], eps=1e-6)
        self.norm3 = Qwen2RMSNorm(config['hidden_size'], eps=1e-6)
        self.norm4 = Qwen2RMSNorm(config['hidden_size'], eps=1e-6)

        self.self_attn = QwenDecoder_VISION_ATTENTION_CLASSES[attn_implementation](
            config['hidden_size'], num_heads=config['num_heads']
        )
        self.cross_attn_query_to_image = QwenDecoder_VISION_ATTENTION_CLASSES[attn_implementation](
            config['hidden_size'], num_heads=config['num_heads']
        )
        self.mlp = nn.Sequential(
            nn.Linear(config['hidden_size'], config['intermediate_size']),
            nn.GELU(),
            nn.Linear(config['intermediate_size'], config['hidden_size'])
        )
        self.update_memory = update_memory
        if update_memory:
            self.cross_attn_image_to_query = QwenDecoder_VISION_ATTENTION_CLASSES[attn_implementation](
                config['hidden_size'], num_heads=config['num_heads']
            )
            self.norm5 = Qwen2RMSNorm(config['hidden_size'], eps=1e-6)
            self.norm6 = Qwen2RMSNorm(config['hidden_size'], eps=1e-6)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_cu_seqlens: torch.Tensor,
        memory_cu_seqlens: torch.Tensor,
        query_pos: torch.Tensor,
        memory_pos: torch.Tensor,
    ) -> torch.Tensor:
        # query: [B * seqlens, D]
        # memory: [B * H * W, D]
        # query_cu_seqlens: [B+1]
        # memory_cu_seqlens: [B+1]
        # query_pos: [B * seqlens, D]
        # memory_pos: [B * seqlens, D // num_heads]

        # query self_attn, pre-norm, follow up Qwen2.5-VL
        query_norm = self.norm1(query)
        query = query + self.self_attn(query_norm, query_norm, query_cu_seqlens, query_cu_seqlens, query_pos, query_pos, (False, False))

        # query cross attention with image embedding
        query_norm = self.norm2(query)
        mem_norm = self.norm3(memory)
        query = query + self.cross_attn_query_to_image(query_norm, mem_norm, query_cu_seqlens, memory_cu_seqlens, query_pos, memory_pos, (False, True))

        # query MLP
        query = query + self.mlp(self.norm4(query))

        if self.update_memory:
            query_norm = self.norm5(query)
            memory_norm = self.norm6(memory)
            memory = memory + self.cross_attn_image_to_query(memory_norm, query_norm, memory_cu_seqlens, query_cu_seqlens, memory_pos, query_pos, (True, False))

        return query, memory


class PaDTDecoder(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        
        self.config = config
        self.dtype = dtype
        self.use_mask_loss = config['use_mask_loss'] if 'use_mask_loss' in config else True

        self.vp_embedding = nn.Embedding(1, config['hidden_size'])
        self.bbox_score_mask_tokens = nn.Embedding(3, config['hidden_size'])

        self.input_projection = nn.Sequential(
            Qwen2RMSNorm(config['llm_hidden_state']),
            nn.Linear(config['llm_hidden_state'], config['hidden_size']),
            nn.GELU(),
            nn.Linear(config['hidden_size'], config['hidden_size']),
        )

        self.spatial_merge_size = config['spatial_merge_size']
        
        self.low_res_transformer = QwenDecoderBlock(config, config['attn_implementation'], update_memory=True)
        self.high_res_transformer1 = QwenDecoderBlock(config, config['attn_implementation'], update_memory=True)
        self.high_res_transformer2 = QwenDecoderBlock(config, config['attn_implementation'], update_memory=True)


        self.high_res_norm = Qwen2RMSNorm(config['hidden_size'])

        self.bbox_prediction = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.GELU(),
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.GELU(),
            nn.Linear(config['hidden_size'], 4),
            nn.Sigmoid()
        )
        self.score_prediction = nn.Linear(config['hidden_size'], 1)

        self.mask_output_upscaling1 = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size'] // 4 * 4),
            Qwen2RMSNorm(config['hidden_size'] // 4 * 4),
            nn.GELU(),
        )
        self.mask_output_upscaling2 = nn.Sequential(
            nn.Linear(config['hidden_size'] // 4, config['hidden_size'] // 16 * 4),
            nn.GELU(),
        )

        self.mask_output_mlp = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.GELU(),
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.GELU(),
            nn.Linear(config['hidden_size'], config['hidden_size'] // 16),
        )
        pass
    
    def forward(self, object_vp_feat, cu_low_res_feats, cu_high_res_feats, visual_pe, cu_patch, obj_image_grid_thws, device):
        vp_num_into_obj = [i.shape[0] for i in object_vp_feat]
        max_vp_num_into_obj = max(vp_num_into_obj)
        num_object = len(object_vp_feat)

        cu_query = []
        num_additional_token = 3  # box token, score token, mask token
        query_attn_mask = torch.zeros((num_object, max_vp_num_into_obj + num_additional_token), dtype=torch.bool).to(device)

        all_object_vp_feat = torch.cat(object_vp_feat)
        all_object_vp_feat = self.input_projection(all_object_vp_feat)

        acc_obj_num = 0 
        for obj_idx, obj_num in enumerate(vp_num_into_obj):
            cu_query.append(self.bbox_score_mask_tokens.weight)
            cu_query.append(all_object_vp_feat[acc_obj_num: acc_obj_num+obj_num] + self.vp_embedding.weight)
            query_attn_mask[obj_idx, :num_additional_token + obj_num] = 1
            acc_obj_num += obj_num

        cu_query = torch.cat(cu_query, dim=0)
        query_cu_seqlens = torch.nn.functional.pad(query_attn_mask.sum(dim=-1).cumsum(dim=-1), (1, 0), 'constant', 0).to(torch.int32)
        
        low_res_feats_cu_seqlens = (cu_patch // self.spatial_merge_size**2).to(torch.int32)
        cu_low_res_feats = self.input_projection(cu_low_res_feats)
        visual_pe_d = visual_pe[0].shape[-1]
        low_res_visual_pe = (visual_pe[0].reshape(-1, self.spatial_merge_size**2, visual_pe_d)[:, 0, :], visual_pe[1].reshape(-1, self.spatial_merge_size**2, visual_pe_d)[:, 0, :])

        # finetune query from image embedding and update image embedding focusing on query points.
        output = cu_query
        output, cu_low_res_feats = self.low_res_transformer(output, cu_low_res_feats, query_cu_seqlens, low_res_feats_cu_seqlens, cu_query, low_res_visual_pe)
        
        # using high resolution image embedding
        high_res_feats_cu_seqlens = cu_patch
        cu_high_res_feats = self.high_res_norm(cu_low_res_feats.unsqueeze(1).repeat_interleave(self.spatial_merge_size ** 2, dim=1).flatten(0, 1) + cu_high_res_feats)
        high_res_visual_pe = visual_pe

        # finetune query again using high res feat
        output, cu_high_res_feats = self.high_res_transformer1(output, cu_high_res_feats, query_cu_seqlens, high_res_feats_cu_seqlens, cu_query, high_res_visual_pe)
        output, cu_high_res_feats = self.high_res_transformer2(output, cu_high_res_feats, query_cu_seqlens, high_res_feats_cu_seqlens, cu_query, high_res_visual_pe)

        output_in_batch = torch.zeros((num_object, max_vp_num_into_obj + num_additional_token, self.config['hidden_size']), dtype=self.dtype).to(device)
        output_in_batch[query_attn_mask] = output

        bbox_score_mask_tokens = output_in_batch[:, :3]
        
        bbox_output = self.bbox_prediction(bbox_score_mask_tokens[:, 0])
        score_output = self.score_prediction(bbox_score_mask_tokens[:, 1])

        if self.use_mask_loss is False:
            return bbox_output, score_output, None, ()

        mask_output = self.mask_output_mlp(bbox_score_mask_tokens[:, 2])

        embedding_N, embedding_D = cu_high_res_feats.shape
        mask_embeddings = self.mask_output_upscaling1(cu_high_res_feats).reshape(embedding_N, 2, 2, embedding_D // 4).permute(1, 2, 0, 3) # 2, 2, N, D // 4
        mask_embeddings = self.mask_output_upscaling2(mask_embeddings).reshape(2, 2, embedding_N, 2, 2, embedding_D // 16).permute(0, 3, 1, 4, 2, 5).flatten(0, 1).flatten(1, 2) # 4, 4, N, D // 16

        mask_embeddings_per_patch = mask_embeddings.permute(2, 0, 1, 3).contiguous() # N, 4, 4, D // 16

        obj_in_image_patch_num = cu_patch[1:] - cu_patch[:-1]
        num_objects = obj_in_image_patch_num.shape[0]
        sum_pn = cu_patch[-1].item()

        object_ids_per_patch = torch.repeat_interleave(torch.arange(num_objects, device=device), obj_in_image_patch_num)
        expanded_mask_output = mask_output.index_select(0, object_ids_per_patch)  # (sum_pn, emb_dim_last)
        
        offsets = cu_patch[:-1]
        patch_indices = torch.arange(sum_pn, device=device, dtype=torch.int64)
        pos_in_obj = patch_indices - offsets[object_ids_per_patch]  # (sum_pn,)
            
        Hs = torch.tensor([thw[1] for thw in obj_image_grid_thws], device=device, dtype=torch.int64)
        Ws = torch.tensor([thw[2] for thw in obj_image_grid_thws], device=device, dtype=torch.int64)

        Ws_per_patch = Ws[object_ids_per_patch]
        row_pos = pos_in_obj // Ws_per_patch
        col_pos = pos_in_obj % Ws_per_patch  # each in its local object's grid

        mask_logit_per_patch = (mask_embeddings_per_patch * expanded_mask_output[:, None, None, :]).sum(dim=-1)  # (sum_pn,4,4)

        H_max = int(Hs.max().item())
        W_max = int(Ws.max().item())

        masks_padded = torch.zeros((num_objects, 4, 4, H_max, W_max), device=device, dtype=mask_logit_per_patch.dtype)

        masks_padded[object_ids_per_patch, :, :, row_pos, col_pos] = mask_logit_per_patch
        
        masks_per_object = masks_padded.permute(0, 3, 1, 4, 2).contiguous()  # (num_objects, H_max, 4, W_max, 4)
        masks_per_object = masks_per_object.reshape(num_objects, H_max * 4, W_max * 4)  # (num_objects, 4H_max, 4W_max)

        return bbox_output, score_output, masks_per_object, (Hs, Ws)