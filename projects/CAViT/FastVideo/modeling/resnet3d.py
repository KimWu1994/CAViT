# encoding: utf-8
import ipdb
import torch
from torch import nn
import torch.nn.functional as F

from fastreid.config import configurable
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from fastreid.modeling.meta_arch import Baseline
from einops import rearrange, repeat

@META_ARCH_REGISTRY.register()
class Res3D(Baseline):

    @configurable
    def __init__(self, *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None,
            temp_kwargs=None):

        super(Res3D, self).__init__(backbone=backbone,
                                            heads=heads,
                                            pixel_mean=pixel_mean,
                                            pixel_std=pixel_std,
                                            loss_kwargs=loss_kwargs)

        self.temp_kwargs = temp_kwargs


    @classmethod
    def from_config(cls, cfg):
        base_res = Baseline.from_config(cfg)
        base_res["temp_kwargs"] = {
            "reduce":cfg.TEMP.REDUCE,
            'test': {
                'all_track' : cfg.TEMP.TEST.ALL,
                'track_split': cfg.TEMP.TEST.TRACK_SPLIT,
            }
        }
        return base_res

    def forward(self, batched_inputs):

        # ipdb.set_trace()
        if not self.training and self.temp_kwargs['test']['all_track']:
            outputs = self.test_all_track_forward(batched_inputs)
            return outputs

        outputs = self.train_forward(batched_inputs)
        return outputs

    def train_forward(self, batched_inputs):
        # ipdb.set_trace()

        b, t, _, _, _ = batched_inputs['images'].size()

        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images)  # [b*t, c, h, w]
        b2, c, h, w = features.size()
        # ipdb.set_trace()
        if b2 == b*t:
            features = features.view(b, t, c, h, w)
        else:
            features = features.view(b, 1, c, h, w)

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

    def test_all_track_forward(self, batched_inputs):

        # ipdb.set_trace()
        outputs = []
        batch_num = len(batched_inputs['images'])  # b * [t, c, h, w]

        for ix in range(batch_num):

            image = batched_inputs['images'][ix].to(self.device)
            image = self.preprocess_image(image)  # [1*t, c, h, w]
            track_split = self.temp_kwargs['test']['track_split']
            images = torch.split(image, track_split, dim=0)

            features = []
            for image_t in images:
                # t_num, c, h, w = image_t.size()
                feature = self.backbone(image_t)  # [1*t, c, h, w]
                t_num, c, h, w = feature.size()
                feature = feature.view(1, t_num, c, h, w)

                feature = self.heads(feature, reduce=False).view(t_num, -1)  # [1 * t， 2048]
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
        images = rearrange(images, '(b t) c h w -> b c t h w', b=b, t=t, c=c,h=h, w=w)
        return images











