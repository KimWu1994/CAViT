#!/usr/bin/env python
# encoding: utf-8

import ipdb
import torch
import logging
import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

from FastVideo import *

class VideoTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        train_loader = build_video_reid_train_loader(cfg)
        return train_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        test_loader, num_query = build_video_reid_test_loader(cfg, dataset_name=dataset_name)

        return test_loader, num_query


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):

        data_loader, num_query = build_video_reid_test_loader(cfg, dataset_name)

        return data_loader, NewVideoReidEvaluator(cfg, num_query, output_dir)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_temp_config(cfg)
    add_swin_config(cfg)
    add_vit_config(cfg)
    add_cascade_config(cfg)
    add_slowfast_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = VideoTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = VideoTrainer.test(cfg, model)
        return res

    trainer = VideoTrainer(cfg)

    # trainer.test(trainer.cfg, trainer.model)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

