#!/usr/bin/env python3
'''
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import Attention
from timm.models.vision_transformer import Block

from ...utils import logging
logger = logging.get_logger("visual_prompt")

class LoRA_Layer(nn.Module):
    # according to the repo:https://github.com/JamesQFreeman/LoRA-ViT
    def __init__(self, dim, rank):
        super().__init__()
        self.lora_w_a = nn.Linear(dim, rank, bias=False)
        self.lora_w_b = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_w_b.weight)

    def forward(self, x):
        x = self.lora_w_b(self.lora_w_a(x))
        return x


class LORA_Attention(nn.Module):
    def __init__(self, attn, lora_config):
        super().__init__()
        self.attn = attn
        self.lora_config = lora_config
        self.rank = lora_config.RANK
        self.tune_query = lora_config.TUNE_QUERY
        self.tune_key = lora_config.TUNE_KEY
        self.tune_value = lora_config.TUNE_VALUE
        self.tune_out = lora_config.TUNE_OUT
        self.hidden_size = attn.qkv.in_features
        if self.tune_query:
            self.lora_query = LoRA_Layer(self.hidden_size, self.rank)
        if self.tune_key:
            self.lora_key = LoRA_Layer(self.hidden_size, self.rank)
        if self.tune_value:
            self.lora_value = LoRA_Layer(self.hidden_size, self.rank)
        if self.tune_out:
            self.lora_out = LoRA_Layer(self.hidden_size, self.rank)

    def forward(self, x): 
        # timm.__version__ == 0.54
        B, N, C = x.shape
        qkv = self.attn.qkv(x)
        if self.tune_query:
            qkv[:, :, : self.hidden_size] += self.lora_query(x)
        if self.tune_key:
            qkv[:, :, self.hidden_size:-self.hidden_size] += self.lora_key(x)
        if self.tune_value:
            qkv[:, :, -self.hidden_size :] += self.lora_value(x)
        qkv = qkv.reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.attn.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.tune_out:
            x = self.attn.proj(x) + self.lora_out(x)
        else:
            x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x


class LORA_Block(Block):
    def __init__(self, lora_cfg, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super(LORA_Block, self).__init__(
            dim=dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            drop=drop, 
            attn_drop=attn_drop,
            drop_path=drop_path, 
            act_layer=act_layer, 
            norm_layer=norm_layer)
        
        self.lora_cfg = lora_cfg
        self.attn = LORA_Attention(self.attn,lora_cfg)

    def forward(self, x):
        # same as reguluar ViT block
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.mlp(x)

        x = self.drop_path(x)
        x = x + h 
        
        return x
