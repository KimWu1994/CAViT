import logging
import math
import copy
from functools import partial
from os import path

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fastreid.layers import DropPath, trunc_normal_, to_2tuple
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from fastreid.modeling.backbones.build import BACKBONE_REGISTRY
from .vit import Mlp, Attention, Block, PatchEmbed, HybridEmbed, resize_pos_embed, PatchEmbed_overlap
# from .vit_msf import build_position

logger = logging.getLogger('fastreid.' + __name__)


def build_position(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # pe.requires_grad = False
    # pe = pe.unsqueeze(0)
    pe = nn.Parameter(pe, requires_grad=False)
    return pe   # [len, d_model]

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # []
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, memory):

        # ipdb.set_trace()
        N1, (B, N2, C) = query.shape[1], memory.shape
    
        # q_fc = self.qkv.weight[0:C,      :]
        # k_fc = self.qkv.weight[C:2*C,    :]
        # v_fc = self.qkv.weight[2*C:3*C, : ]

        q = F.linear(query, self.qkv.weight[0:C,   :], self.qkv.bias[0:C])
        k = F.linear(memory, self.qkv.weight[C:2*C, :], self.qkv.bias[C:2*C])
        v = F.linear(memory, self.qkv.weight[2*C:3*C, :], self.qkv.bias[2*C:3*C])

        q = q.reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        k = k.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        v = v.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class CrossBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, memory):
        atten = self.attn(self.norm1(query), self.norm1(memory))
        query = query + self.drop_path(atten)
        query = query + self.drop_path(self.mlp(self.norm2(query)))
        return query



class ViTMultiScaleFlow(nn.Module):
    """ Vision Transformer
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
        """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, padding_size=0, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., camera=0, drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=1.0, inference_depth=12, layer_num=12, norm_out=False, 
                 dilation=2, patch2=(32, 16), stride2=(32, 16), patch3=(16, 32), stride3=(16, 32), 
                 tpe='sin', max_len=1000):

        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                    img_size=img_size, patch_size=patch_size, padding_size=padding_size, stride_size=stride_size, 
                    in_chans=in_chans, 
                    embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

            if patch2 != (0, 0):
                self.patch_embed2 = PatchEmbed_overlap(
                    img_size=img_size, patch_size=patch2, stride_size=stride2, padding_size=padding_size, 
                    in_chans=in_chans,
                    embed_dim=embed_dim)
                num_patches2 = self.patch_embed2.num_patches
                self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches2, embed_dim))
                
                trunc_normal_(self.pos_embed2, std=.02)
            else:
                self.patch_embed2 = None
            
            if patch3 != (0, 0):
                self.patch_embed3 = PatchEmbed_overlap(
                    img_size=img_size, patch_size=patch3, stride_size=stride3, padding_size=padding_size, 
                    in_chans=in_chans,
                    embed_dim=embed_dim)

                num_patches3 = self.patch_embed3.num_patches 
                self.pos_embed3 = nn.Parameter(torch.zeros(1, num_patches3, embed_dim))
                trunc_normal_(self.pos_embed3, std=.02)
            else:
                self.patch_embed3 = None
        
        self.cam_num = camera
        self.sie_xishu = sie_xishu
        self.max_len = max_len
        # Initialize SIE Embedding
        if camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = nn.ModuleList([
            CrossBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.inference_depth = inference_depth
        self.layer_num = layer_num
        self.tpe = tpe
        self.blocks = blocks[: layer_num]

        self._build_scale_branch()
        self._build_flow_emb()

        self.norm_out = norm_out
        if self.norm_out:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     token = {'pos_embed', 'pos_embed2', 'pos_embed3', 'cls_token'}
    #     return token

    def _build_scale_branch(self):
        if self.patch_embed2 != None:
            self.blocks2 = copy.deepcopy(self.blocks)        
        
        if self.patch_embed3 != None:
            self.blocks3 = copy.deepcopy(self.blocks)
    
    def _build_flow_emb(self):
        
        if self.tpe == 'flow':
            self.pos_embed_flow  = copy.deepcopy(self.patch_embed)
            if self.patch_embed2 != None:
                self.pos_embed2_flow = copy.deepcopy(self.patch_embed2)
            
            if self.patch_embed3 != None:
                self.pos_embed3_flow = copy.deepcopy(self.patch_embed3)
        
        elif self.tpe == 'sin':
            self.pos_embed_sin  = build_position(max_len=self.max_len, d_model=768)
            if self.patch_embed2 != None:
                self.pos_embed2_sin = build_position(max_len=self.max_len, d_model=768)
            
            if self.patch_embed3 != None:
                self.pos_embed3_sin = build_position(max_len=self.max_len, d_model=768)
        
        elif self.tpe == 'all':
            self.pos_embed_flow  = copy.deepcopy(self.patch_embed)
            self.pos_embed_sin  = build_position(max_len=self.max_len, d_model=768)

            if self.patch_embed2 != None:
                self.pos_embed2_flow = copy.deepcopy(self.patch_embed2)
                self.pos_embed2_sin = build_position(max_len=self.max_len, d_model=768)
            
            if self.patch_embed3 != None:
                self.pos_embed3_flow = copy.deepcopy(self.patch_embed3)
                self.pos_embed3_sin = build_position(max_len=self.max_len, d_model=768)
        
        else:
            pass
    

    def forward(self, x, camera_id=None, frame_id=None):
        
       # ipdb.set_trace()
        B, T, C, H, W = x.shape

        ### flow ##############################################
        x = x.view(B*T, C, H, W)
        x1 = self.patch_embed(x)
        x1 = x1 + self.pos_embed

        if self.patch_embed2 != None:
            x2 = self.patch_embed2(x)
            x2 = x2 + self.pos_embed2 
        
        if self.patch_embed3 != None:
            x3 = self.patch_embed3(x)
            x3 = x3 + self.pos_embed3            

        ##########################################################
        if self.tpe == 'flow':
            x = x.view(B, T, C, H, W)
            x_roll = torch.roll(x, 1, dims=1)
            x_flow = x_roll - x
            x_flow[:, 0, :, :, :] = x_flow[:, 0, :, :, :] * 0.0
            tpe = x_flow.view(B*T, C, H, W)
            x = x.view(B*T, C, H, W)
            
            tpe = rearrange(tpe, "b c h w -> b c (h w)", h=H, w=W)
            tpe = F.softmax(tpe, dim=-1)
            tpe = rearrange(tpe, "b c (h w) -> b c h w", h=H, w=W)

            x1_tpe = self.pos_embed_flow(tpe)
            x1_tpe = F.softmax(x1_tpe, dim=-1)
            x1 = x1 + x1_tpe
            

            if self.patch_embed2 != None:
                x2_tpe = self.pos_embed2_flow(tpe)
                x2_tpe = F.softmax(x2_tpe, dim=-1)
                x2 = x2 + x2_tpe
            
            if self.patch_embed3 != None:
                x3_tpe = self.pos_embed3_flow(tpe)
                x3_tpe = F.softmax(x3_tpe, dim=-1)
                x3 = x3 + x3_tpe

        elif self.tpe == 'sin':
            tpe = self.pos_embed_sin[frame_id.view(B*T), :].unsqueeze(1)  # [B*T, 1, C]
            x1 = x1 + tpe.expand(B*T, x1.shape[1], -1)
            
            if self.patch_embed2 != None:
                tpe = self.pos_embed2_sin[frame_id.view(B*T), :].unsqueeze(1)  # [B*T, 1, C]
                x2 = x2 + tpe.expand(B*T, x2.shape[1], -1)
            
            if self.patch_embed3 != None:
                tpe = self.pos_embed3_sin[frame_id.view(B*T), :].unsqueeze(1)  # [B*T, 1, C]
                x3 = x3 + tpe.expand(B*T, x3.shape[1], -1)
        
        elif self.tpe == 'all':
            x = x.view(B, T, C, H, W)
            x_roll = torch.roll(x, 1, dims=1)
            x_flow = x_roll - x
            x_flow[:, 0, :, :, :] = x_flow[:, 0, :, :, :] * 0.0
            tpe = x_flow.view(B*T, C, H, W)
            x = x.view(B*T, C, H, W)
            
            tpe = rearrange(tpe, "b c h w -> b c (h w)", h=H, w=W)
            tpe = F.softmax(tpe, dim=-1)
            tpe = rearrange(tpe, "b c (h w) -> b c h w", h=H, w=W)

            x1_flow_tpe = self.pos_embed_flow(tpe)
            # x1_flow_tpe = F.softmax(x1_flow_tpe, dim=-1)

            x1_sin_tpe = self.pos_embed_sin[frame_id.view(B*T), :].unsqueeze(1)  # [B*T, 1, C]
            x1_sin_tpe = x1_sin_tpe.expand(B*T, x1.shape[1], -1)

            x1 = x1 + 0.5 * x1_sin_tpe + 0.5 * x1_flow_tpe
            
            if self.patch_embed2 != None:
                x2_flow_tpe = self.pos_embed2_flow(tpe)
                # x2_flow_tpe = F.softmax(x2_flow_tpe, dim=-1)
                
                x2_sin_tpe = self.pos_embed2_sin[frame_id.view(B*T), :].unsqueeze(1)  # [B*T, 1, C]
                x2_sin_tpe = x2_sin_tpe.expand(B*T, x2.shape[1], -1)

                x2 = x2 + 0.5 * x2_sin_tpe + 0.5 * x2_flow_tpe
            
            if self.patch_embed3 != None:
                x3_flow_tpe = self.pos_embed3_flow(tpe)
                # x3_flow_tpe = F.softmax(x3_flow_tpe, dim=-1)

                x3_sin_tpe = self.pos_embed3_sin[frame_id.view(B*T), :].unsqueeze(1)  # [B*T, 1, C]
                x3_sin_tpe = x3_sin_tpe.expand(B*T, x3.shape[1], -1)

                x3 = x3 + 0.5 * x3_sin_tpe + 0.5 * x3_flow_tpe
        
        else:
            pass


        #################################################################
        x1 = self.pos_drop(x1)  # [batch, 129, 768]
        
        if self.patch_embed2 != None:
            x2 = self.pos_drop(x2)  # [batch, 129, 768]

        if self.patch_embed3 != None: 
            x3 = self.pos_drop(x3)  # [batch, 129, 768]
        
        ##############################################################################################################
        out = []

        for i in range(self.inference_depth):
            
            x1 = rearrange(x1, "(b t) p c -> b t p c", t=T)
            x1_shift = torch.roll(x1, shifts=-1, dims=1)
            
            x1_memory = torch.cat([x1, x1_shift], dim=2)
            x1_memory = rearrange(x1_memory, "b t p c -> (b t) p c", t=T)
            x1 = rearrange(x1, 'b t p c -> (b t) p c', t=T)

            x1 = self.blocks[i](x1, x1_memory)
            
            if self.patch_embed2 != None:
                x2 = rearrange(x2, "(b t) p c -> b t p c", t=T)
                x2_shift = torch.roll(x2, shifts=-1, dims=1)
                
                x2_memory = torch.cat([x2, x2_shift], dim=2)
                x2_memory = rearrange(x2_memory, "b t p c -> (b t) p c", t=T)
                x2 = rearrange(x2, 'b t p c -> (b t) p c', t=T)

                x2 = self.blocks2[i](x2, x2_memory)

            if self.patch_embed3 != None:
                x3 = rearrange(x3, "(b t) p c -> b t p c", t=T)
                x3_shift = torch.roll(x3, shifts=-1, dims=1)
                
                x3_memory = torch.cat([x3, x3_shift], dim=2)
                x3_memory = rearrange(x3_memory, "b t p c -> (b t) p c", t=T)
                x3 = rearrange(x3, 'b t p c -> (b t) p c', t=T)
                x3 = self.blocks3[i](x3, x3_memory)
        
                
        if self.norm_out:
            x1 = self.norm(x1)
            x1 = rearrange(x1, "(b t) p c -> b t p c", t=T)
            out.append(x1)

            if self.patch_embed2 != None:
                x2 = self.norm(x2)
                x2 = rearrange(x2, "(b t) p c -> b t p c", t=T)
                out.append(x2)

            if self.patch_embed3 != None:
                x3 = self.norm(x3)
                x3 = rearrange(x3, "(b t) p c -> b t p c", t=T)
                out.append(x3)

        x = torch.cat(out, dim=2)
        x = torch.mean(x, dim=2)
        return x

def resize_pos_embed2(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    logger.info('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                      posemb_new.shape,
                                                                                                      hight,
                                                                                                      width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    return posemb_grid


@BACKBONE_REGISTRY.register()
def build_cavit_backbone(cfg):
    """
    Create a Vision Transformer instance from config.
    Returns:
        SwinTransformer: a :class:`SwinTransformer` instance.
    """
    # fmt: off
    input_size      = cfg.INPUT.SIZE_TRAIN
    pretrain        = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path   = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    depth           = cfg.MODEL.BACKBONE.DEPTH
    sie_xishu       = cfg.MODEL.BACKBONE.SIE_COE
    
    drop_ratio      = cfg.MODEL.BACKBONE.DROP_RATIO
    drop_path_ratio = cfg.MODEL.BACKBONE.DROP_PATH_RATIO
    attn_drop_rate  = cfg.MODEL.BACKBONE.ATT_DROP_RATE
    inference_depth = cfg.MODEL.BACKBONE.INFERENCE_DEPTH
    layer_num       = cfg.MODEL.BACKBONE.LAYER_NUM
    norm_out        = cfg.MODEL.BACKBONE.NORM_OUT
    padding_size    = cfg.MODEL.BACKBONE.PADDING_SIZE
    num_camera      = cfg.MODEL.BACKBONE.NUM_CAMERA
    dilation        = cfg.MODEL.BACKBONE.DILATION
    
    patch_size      = cfg.CASCADE.PATCH1
    stride_size     = cfg.CASCADE.STRIDE1

    patch2          = cfg.CASCADE.PATCH2
    stride2         = cfg.CASCADE.STRIDE2

    patch3          = cfg.CASCADE.PATCH3
    stride3         = cfg.CASCADE.STRIDE3

    TPE             = cfg.CASCADE.TPE
    max_len         = cfg.CASCADE.MAX_LEN

    # fmt: on

    num_depth = {
        'small': 8,
        'base': 12,
    }[depth]

    assert inference_depth <= layer_num
    assert layer_num <= num_depth

    num_heads = {
        'small': 8,
        'base': 12,
    }[depth]

    mlp_ratio = {
        'small': 3.,
        'base': 4.
    }[depth]

    qkv_bias = {
        'small': False,
        'base': True
    }[depth]

    qk_scale = {
        'small': 768 ** -0.5,
        'base': None,
    }[depth]

    model = ViTMultiScaleFlow(img_size=input_size, sie_xishu=sie_xishu, stride_size=stride_size, padding_size=padding_size, depth=num_depth,
                                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, camera=num_camera,
                                        drop_path_rate=drop_path_ratio, drop_rate=drop_ratio, attn_drop_rate=attn_drop_rate,
                                        inference_depth=inference_depth, layer_num=layer_num, norm_out=norm_out, dilation=dilation, 
                                        patch_size=patch_size, patch2=patch2, stride2=stride2, patch3=patch3, stride3=stride3, 
                                        tpe=TPE, max_len=max_len)

    if patch2 != (0, 0) : blocks2_dict = {}
    else: blocks2_dict = None

    
    if patch3 != (0, 0): blocks3_dict = {}
    else: blocks3_dict = None

    if pretrain:
        try:
            state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
            logger.info(f"Loading pretrained model from {pretrain_path}")

            if 'model' in state_dict:
                state_dict = state_dict.pop('model')
            if 'state_dict' in state_dict:
                state_dict = state_dict.pop('state_dict')

            if TPE == 'flow' or TPE == 'all':
                O, I, H, W = model.patch_embed.proj.weight.shape
                value = F.interpolate(state_dict['patch_embed.proj.weight'], size=[H, W], mode='bilinear')
                state_dict["pos_embed_flow.proj.weight"] = value
                state_dict["pos_embed_flow.proj.bias"]   = state_dict['patch_embed.proj.bias']


            if patch2 != (0, 0):
                O, I, H, W = model.patch_embed2.proj.weight.shape
                blocks2_dict["patch_embed2.proj.weight"] = F.interpolate(state_dict['patch_embed.proj.weight'], size=[H, W], mode='bilinear')
                blocks2_dict["patch_embed2.proj.bias"] = state_dict['patch_embed.proj.bias']
                blocks2_dict['pos_embed2'] = resize_pos_embed2(state_dict['pos_embed'], model.pos_embed2.data, model.patch_embed2.num_y, model.patch_embed2.num_x)

                if TPE == 'flow'  or TPE == 'all':
                    blocks2_dict["pos_embed2_flow.proj.weight"] = blocks2_dict["patch_embed2.proj.weight"]
                    blocks2_dict["pos_embed2_flow.proj.bias"]   = blocks2_dict['patch_embed2.proj.bias']


            if patch3 != (0, 0):
                O, I, H, W = model.patch_embed3.proj.weight.shape
                blocks3_dict["patch_embed3.proj.weight"] = F.interpolate(state_dict['patch_embed.proj.weight'], size=[H, W], mode='bilinear')
                blocks3_dict["patch_embed3.proj.bias"] = state_dict['patch_embed.proj.bias']
                blocks3_dict['pos_embed3'] = resize_pos_embed2(state_dict['pos_embed'], model.pos_embed3.data, model.patch_embed3.num_y, model.patch_embed3.num_x)

                if TPE == 'flow'  or TPE == 'all':
                    blocks2_dict["pos_embed3_flow.proj.weight"] = blocks3_dict["patch_embed3.proj.weight"]
                    blocks2_dict["pos_embed3_flow.proj.bias"] = blocks3_dict['patch_embed3.proj.bias']
            
            
            for k, v in state_dict.items():
                if 'head' in k or 'dist' in k:
                    continue

                # if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                #     # For old models that I trained prior to conv based patchification
                #     O, I, H, W = model.patch_embed.proj.weight.shape
                #
                #     v = v.reshape(O, -1, H, W)

                if 'blocks' in k:
                    if blocks2_dict != None:
                        k2 = k.split('.')
                        k2[0] = 'blocks2'
                        k2 = '.'.join(i for i in k2)
                        blocks2_dict[k2] = v
                                        
                    if blocks3_dict != None:
                        k3 = k.split('.')
                        k3[0] = 'blocks3'
                        k3 = '.'.join(i for i in k3)
                        blocks3_dict[k3] = v


                if 'patch_embed.proj.weight' in k:
                    O1, I1, H1, W1 = v.shape
                    O, I, H, W = model.patch_embed.proj.weight.shape
                    v = F.interpolate(v, size=[H, W], mode='bilinear')

                elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
                    # To resize pos embedding when using model at different size from pretrained weights
                    if 'distilled' in pretrain_path:
                        logger.info("distill need to choose right cls token in the pth.")
                        v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                    v = resize_pos_embed2(v, model.pos_embed.data, model.patch_embed.num_y, model.patch_embed.num_x)

                state_dict[k] = v
            
            # ipdb.set_trace()
            if blocks2_dict != None:
                state_dict.update(blocks2_dict)
            
            if blocks3_dict != None:
                state_dict.update(blocks3_dict)

        except FileNotFoundError as e:
            logger.info(f'{pretrain_path} is not found! Please check this path.')
            raise e
        except KeyError as e:
            logger.info("State dict keys error! Please check the state dict.")
            raise e

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )
        
        
        # org_state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))

        # if 'model' in org_state_dict:
        #         org_state_dict = org_state_dict.pop('model')
        # if 'state_dict' in org_state_dict:
        #         org_state_dict = org_state_dict.pop('state_dict')

        # v1 = org_state_dict['patch_embed.proj.weight']
        # v2 = org_state_dict['pos_embed']

        # if patch2 != (0, 0):
        #     O, I, H, W = model.patch_embed2.proj.weight.shape
        #     model.patch_embed2.proj.weight.data = F.interpolate(org_state_dict['patch_embed.proj.weight'], size=[H, W], mode='bilinear')
        #     model.pos_embed2.data = resize_pos_embed2(org_state_dict['pos_embed'], model.pos_embed2.data, model.patch_embed2.num_y, model.patch_embed2.num_x)
        
        # if patch3 != (0, 0):
        #     O, I, H, W = model.patch_embed3.proj.weight.shape
        #     model.patch_embed3.proj.weight.data = F.interpolate(org_state_dict['patch_embed.proj.weight'], size=[H, W], mode='bilinear')
        #     model.pos_embed3.data = resize_pos_embed2(org_state_dict['pos_embed'], model.pos_embed3.data, model.patch_embed3.num_y, model.patch_embed3.num_x)

        # model._build_scale_branch()
        # model._build_flow_emb()

    return model
