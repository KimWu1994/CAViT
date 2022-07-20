# encoding: utf-8
import ipdb
import torch
import logging
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fastreid.config import configurable
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from .ViTBaseline import ViTBaseline

logger = logging.getLogger('fastreid.' + __name__)

@META_ARCH_REGISTRY.register()
class CAViT(ViTBaseline):

    @configurable
    def __init__(self, *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None,
            temp_kwargs=None,
            token='all',
            with_camera=None):

        super(CAViT, self).__init__(backbone=backbone,
                                          heads=heads,
                                          pixel_mean=pixel_mean,
                                          pixel_std=pixel_std,
                                          loss_kwargs=loss_kwargs,
                                          temp_kwargs=temp_kwargs)

        self.token = token
        self.with_camera = with_camera

    # @classmethod
    # def from_config(cls, cfg):
    #     base_res = ViTSlowFast.from_config(cfg)
    #     base_res['token'] = cfg.MODEL.BACKBONE.TOKEN
    #     return base_res

    def forward(self, batched_inputs):

        # ipdb.set_trace()
        if isinstance(batched_inputs['images'], list) or isinstance(batched_inputs, list):
            outputs = self.flexible_forward(batched_inputs)
        else:
            outputs = self.fix_forward(batched_inputs)

        return outputs

    def fix_forward(self, batched_inputs):

        # b, t, _, _, _ = batched_inputs['images'].size()
        images = self.preprocess_image(batched_inputs)

        cameras = batched_inputs['camids']
        frames = batched_inputs['frame_ids']
        features = self.backbone(images, cameras, frames)  # [b*t, 129, 768]

        features = features.unsqueeze(3).unsqueeze(4)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            # ipdb.set_trace()
            outputs = self.heads(features)
            return outputs


    def flexible_forward(self, batched_inputs):
        # ipdb.set_trace()
        outputs = []
        batch_num = len(batched_inputs['images'])  # b * [t, c, h, w]

        # ipdb.set_trace()
        for ix in range(batch_num):

            image = batched_inputs['images'][ix].to(self.device)
            camid = batched_inputs['camids'][ix].unsqueeze(0)
            frame = batched_inputs['frame_ids'][ix].unsqueeze(0)
            
            image = self.preprocess_image(image)  # [1, t, c, h, w]
            image = image.unsqueeze(0)
            track_split = self.temp_kwargs['test']['track_split']
            
            
            images = torch.split(image, track_split, dim=1)
            frames = torch.split(frame, track_split, dim=1)


            # T = image.shape[1]
            features = []
            for ix, (image_t, frame_t) in enumerate(zip(images, frames)):

                # if ix > 0 and image_t.shape[1] < T/2:
                #     continue

                # t_num, c, h, w = image_t.size()
                feature = self.backbone(image_t, camid, frame_t)  # [b, t, c]
                feat_dim = feature.shape[-1]
                feature = feature.unsqueeze(3).unsqueeze(4) # [b, t, c, 1, 1]
                
                # num, p, c = feature.size()
                # # feature = feature.view(1, num, p, c)
                # feature = self._token_pool(feature).view(1, num, c, 1, 1)
                feature = self.heads(feature, reduce=False).view(-1, feat_dim)  # [1 * num， c]
                # feature = self.heads(feature).view(-1, feat_dim)  # [1 * num， c]
                features.append(feature)

            output = torch.mean(torch.cat(features, dim=0), dim=0, keepdim=True)
            outputs.append(output)

        return torch.cat(outputs, dim=0)


    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        if len(images.shape) > 4:
            b, t, c, h, w = images.size()
            images = images.view(b * t, c, h, w)
            images.sub_(self.pixel_mean).div_(self.pixel_std)
            images = images.view(b, t, c, h, w)
        
        else:
            images.sub_(self.pixel_mean).div_(self.pixel_std)

        return images







