# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import ipdb
import torch
import torch.nn.functional as F

from fastreid.modeling.heads import REID_HEADS_REGISTRY, EmbeddingHead


@REID_HEADS_REGISTRY.register()
class TempHead(EmbeddingHead):


    def __init__(self, cfg):
        super().__init__(cfg)
        self.temp_reduce = cfg.TEMP.REDUCE

    def forward(self, features, targets=None, reduce=True):
        """
        See :class:`ClsHeads.forward`.
        features: [b * t, c, h, w]
        """
        # ipdb.set_trace()
        b, t, c, h, w = features.size()
        features = features.view(b*t, c, h, w)
        # b = bt // t_size

        pool_feat = self.pool_layer(features)  # [b*t, 2048, 1, 1]
        neck_feat = self.bottleneck(pool_feat) # [b*t, 2048, 1, 1]
        neck_feat = neck_feat[..., 0, 0]

        # temp funsion
        pool_feat = pool_feat.view(b, t, -1)   # [b, t, 2048]
        neck_feat = neck_feat.view(b, t, -1)   # [b, t, 2048]

        if reduce and self.temp_reduce != None:
            if self.temp_reduce == 'avg':
                pool_feat = torch.mean(pool_feat, dim=1)
                neck_feat = torch.mean(neck_feat, dim=1)
            elif self.temp_reduce == 'max':
                pool_feat = torch.max(pool_feat, 1)[0]
                neck_feat = torch.max(neck_feat, 1)[0]
            else:
                raise KeyError(f"{self.temp_reduce} is invalid in ['avg', 'max', 'None'] ")
        else:
            pool_feat = pool_feat.view(b, t, c)
            neck_feat = neck_feat.view(b, t, c)



        # Evaluation
        # fmt: off
        if not self.training: return neck_feat
        # fmt: on

        # Training
        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        # Pass logits.clone() into cls_layer, because there is in-place operations
        cls_outputs = self.cls_layer(logits.clone(), targets)

        # fmt: off
        if self.neck_feat == 'before':
            feat = pool_feat
        elif self.neck_feat == 'after':
            feat = neck_feat
        else:
            raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits.mul(self.cls_layer.s),
            "features": feat,
        }



