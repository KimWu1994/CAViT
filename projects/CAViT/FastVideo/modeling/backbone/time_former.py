import logging

import ipdb
import torch
import torch.nn as nn
from einops import  rearrange, repeat
from .vit import Attention, DropPath, Mlp, trunc_normal_
import torch.nn.functional as F
from functools import partial
from fastreid.layers import DropPath, trunc_normal_, to_2tuple

import os
import math
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from fastreid.modeling.backbones.build import BACKBONE_REGISTRY

logger = logging.getLogger("fastreid." + __name__)


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, padding_size=0, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        padding_size_tuple = to_2tuple(padding_size)
        self.num_x = (2 * padding_size_tuple[1] + img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (2 * padding_size_tuple[0] + img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size,
                              padding=padding_size_tuple)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, T, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.proj(x)

        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x, T, W

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only', 'joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class ViT3D(nn.Module):
    """ Vision Transformere
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, padding_size=0, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8,
                 attention_type='divided_space_time', dropout=0.):

        super().__init__()

        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size,
                                              padding_size=padding_size, in_chans=in_chans, embed_dim=embed_dim)

        self.num_patches = self.patch_embed.num_patches
        # print(self.num_patches)

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                  attention_type=self.attention_type) for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def forward(self, x, track_id=None, camera_id=None):
        # ipdb.set_trace()
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size()[1] != self.pos_embed.size()[1]:
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size()[1] // W
            other_pos_embed = other_pos_embed.reshape(1, x.size()[2], P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)    # [b*p, t, c]
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size()[1]:
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m', b=B, t=T)
            x = torch.mean(x, 1)  # averaging predictions for every frame

        x = self.norm(x)

        ###  temporal max pooling
        # cls_token = x[:, 0, :].unsqueeze(1)
        # patchs = x[:, 1:, :]
        # patchs = rearrange(patchs, 'b (h w t) m -> (b h w) t m', b=B, w=W, t=T)
        # patchs = torch.max(patchs, dim=1)[0]
        # patchs = rearrange(patchs, '(b h w) m -> b (h w) m', b=B, w=W)
        # x = torch.cat([cls_token, patchs], dim=1)

        # x = torch.mean(x, dim=1)
        x = x[:, 0, :]
        x = rearrange(x, "b (m h w) -> b m h w", h=1, w=1)
        return x



def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif 'model_state' in checkpoint:
            state_dict_key = 'model_state'
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `model.` prefix
                name = k[6:] if k.startswith('model') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained(model, cfg=None,  in_chans=3, filter_fn=None, num_frames=8,
                    num_patches=196, attention_type='divided_space_time', pretrained_model=""):

    if len(pretrained_model) == 0:
        state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')
    else:
        try:
            state_dict = load_state_dict(pretrained_model)['model']
        except:
            state_dict = load_state_dict(pretrained_model)

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            logger.warning('Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            logger.info('Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    ## Resizing the positional embeddings in case they don't match
    if num_patches + 1 != state_dict['pos_embed'].size(1):
        logger.info('Resizing position embedding from statict {} to {}'.format(state_dict['pos_embed'].size(1),
                                                                               num_patches + 1))
        pos_embed = state_dict['pos_embed']
        cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
        # print(other_pos_embed.size(), num_patches)
        new_pos_embed = F.interpolate(other_pos_embed, size=num_patches, mode='nearest')
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict['pos_embed'] = new_pos_embed

    ## Resizing time embeddings in case they don't match
    if 'time_embed' in state_dict and num_frames != state_dict['time_embed'].size(1):
        logger.info(
            'Resizing time embedding from statict {} to {}'.format(state_dict['time_embed'].size(1), num_frames))
        time_embed = state_dict['time_embed'].transpose(1, 2)
        new_time_embed = F.interpolate(time_embed, size=(num_frames), mode='nearest')
        state_dict['time_embed'] = new_time_embed.transpose(1, 2)

    ## Initializing temporal attention
    if attention_type == 'divided_space_time':
        logger.info("Initializing temporal attention")
        new_state_dict = state_dict.copy()
        for key in state_dict:
            if 'blocks' in key and 'attn' in key:
                new_key = key.replace('attn', 'temporal_attn')
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
            if 'blocks' in key and 'norm1' in key:
                new_key = key.replace('norm1', 'temporal_norm1')
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
        state_dict = new_state_dict

    ## Loading the weights
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logger.info(
            get_missing_parameters_message(incompatible.missing_keys)
        )
    if incompatible.unexpected_keys:
        logger.info(
            get_unexpected_parameters_message(incompatible.unexpected_keys)
        )


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@BACKBONE_REGISTRY.register()
def build_vit3d_backbone(cfg):
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
    stride_size     = cfg.MODEL.BACKBONE.STRIDE_SIZE
    drop_ratio      = cfg.MODEL.BACKBONE.DROP_RATIO
    drop_path_ratio = cfg.MODEL.BACKBONE.DROP_PATH_RATIO
    attn_drop_rate  = cfg.MODEL.BACKBONE.ATT_DROP_RATE
    inference_depth = cfg.MODEL.BACKBONE.INFERENCE_DEPTH
    layer_num       = cfg.MODEL.BACKBONE.LAYER_NUM
    norm_out        = cfg.MODEL.BACKBONE.NORM_OUT
    padding_size    = cfg.MODEL.BACKBONE.PADDING_SIZE
    num_camera      = cfg.MODEL.BACKBONE.NUM_CAMERA
    seq_max         = cfg.MODEL.BACKBONE.SEQ_MAX

    attention_type = cfg.MODEL.BACKBONE.ATT_TYPE
    num_frames = cfg.MODEL.BACKBONE.NUM_FRAMES

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



    model = ViT3D(img_size=input_size, patch_size=16, stride_size=stride_size, padding_size=padding_size, embed_dim=768, depth=layer_num,
                  num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=drop_ratio,
                  attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_ratio, num_frames=num_frames, attention_type='divided_space_time')



    if pretrain:
        load_pretrained(model,  in_chans=3, filter_fn=_conv_filter, num_frames=num_frames, num_patches=model.num_patches,
                        attention_type=attention_type, pretrained_model=pretrain_path)


    return model





