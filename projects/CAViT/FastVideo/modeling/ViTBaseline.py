# encoding: utf-8
import ipdb
import torch
import logging
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fastreid.config import configurable
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from .VideoBaseline import VideoBaseline

logger = logging.getLogger('fastreid.' + __name__)

@META_ARCH_REGISTRY.register()
class ViTBaseline(VideoBaseline):

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

        super(ViTBaseline, self).__init__(backbone=backbone,
                                          heads=heads,
                                          pixel_mean=pixel_mean,
                                          pixel_std=pixel_std,
                                          loss_kwargs=loss_kwargs,
                                          temp_kwargs=temp_kwargs)

        self.token = token
        self.with_camera = with_camera

    @classmethod
    def from_config(cls, cfg):
        base_res = VideoBaseline.from_config(cfg)
        base_res['token'] = cfg.MODEL.BACKBONE.TOKEN
        return base_res

    def forward(self, batched_inputs):

        # ipdb.set_trace()
        if isinstance(batched_inputs['images'], list) or isinstance(batched_inputs, list):
            outputs = self.flexible_forward(batched_inputs)
        else:
            outputs = self.fix_forward(batched_inputs)

        return outputs

    def fix_forward(self, batched_inputs):

        b, t, _, _, _ = batched_inputs['images'].size()
        images = self.preprocess_image(batched_inputs)

        # cameras = self.preprocess_cameras(batched_inputs, b, t)
        cameras = batched_inputs['camids']
        frames = batched_inputs['frame_ids']
        features = self.backbone(images, cameras, frames)  # [b*t, 129, 768]

        _, p, c = features.size()

        # ipdb.set_trace()
        features = features.view(b, t, p, c)
        features = self._token_pool(features).view(b, t, c, 1, 1)

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

    # def flexible_forward(self, batched_inputs):
    #     track_num = [item.size()[0] for item in batched_inputs['images']]
    #
    #     images = torch.cat(batched_inputs['images'], dim=0)  # [1, b*t, h, w]
    #     images = images.unsqueeze(0)
    #
    #     track_split = self.temp_kwargs['test']['track_split']
    #     images = torch.split(images, track_split, dim=1)
    #     features = []
    #
    #     # ipdb.set_trace()
    #     for image in images:
    #         image = self.preprocess_image(image.to(self.device))
    #         feature = self.backbone(image)  # [1*b*t, p, c]
    #         num, p, c = feature.size()
    #         feature = feature.view(1, num, p, c)
    #         feature = self._token_pool(feature).view(1, num, c, 1, 1)
    #         feature = self.heads(feature, reduce=False).view(num, -1)  # [1 * num， c]
    #         features.append(feature)  # [num, c]
    #
    #     features = torch.cat(features, dim=0)  # [num, c]
    #     features = torch.split(features, split_size_or_sections=track_num)  # [t, p, c]
    #     features = [torch.mean(feat, dim=0, keepdim=True) for feat in features]
    #     features = torch.cat(features, dim=0)  # [track, c]
    #     return features

    def flexible_forward(self, batched_inputs):
        # ipdb.set_trace()
        outputs = []
        batch_num = len(batched_inputs['images'])  # b * [t, c, h, w]


        for ix in range(batch_num):

            image = batched_inputs['images'][ix].to(self.device)
            camids = batched_inputs['camids'][ix].unsqueeze(0)
            image = self.preprocess_image(image)  # [1, t, c, h, w]
            track_split = self.temp_kwargs['test']['track_split']
            images = torch.split(image, track_split, dim=0)

            features = []
            for image_t in images:
                # t_num, c, h, w = image_t.size()
                feature = self.backbone(image_t, camids)  # [1*t, c, h, w]
                num, p, c = feature.size()
                feature = feature.view(1, num, p, c)
                feature = self._token_pool(feature).view(1, num, c, 1, 1)
                feature = self.heads(feature, reduce=False).view(num, -1)  # [1 * num， c]

                features.append(feature)

            output = torch.mean(torch.cat(features, dim=0), dim=0, keepdim=True)
            outputs.append(output)

        return torch.cat(outputs, dim=0)

    def _token_pool(self, features):

        if self.token == 'all':
            # g_feat = features[:, :, 0, :]  # [b, t, c]
            # local_feat = torch.mean(features[:, :, 1:, :], dim=2)  # [b, t, c]
            # features = (g_feat + local_feat) / 2.0
            features = torch.mean(features, dim=2)  # [b ,t, p, c] --> [b, t, c]

        elif self.token == 'cls':
            features = features[:, :, 0, :]  # [b, t, c]

        elif self.token == 'patch':
            features = features[:, :, 1:, :]  # [b, t, p, c]
            features = torch.mean(features, dim=2)

        else:  raise KeyError(f"Expect token in [all, cls, patch]")

        return features

    def preprocess_cameras(self, batched_inputs, b, t):

        if isinstance(batched_inputs, dict):
            camids = batched_inputs['camids']
        elif isinstance(batched_inputs, torch.Tensor):
            camids = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        camids = repeat(camids, 'b -> b t ', b=b, t=t)
        camids = rearrange(camids, 'b t -> (b t)')

        return camids






