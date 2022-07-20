import logging
import torch
from torch._six import container_abcs, string_classes, int_classes

from fastreid.utils import comm
# from fastreid.data import build_reid_train_loader, build_reid_test_loader, samplers
from fastreid.data import build_reid_train_loader, samplers
from fastreid.data.data_utils import DataLoaderX

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.transforms import build_transforms
from fastreid.data.build import _root, fast_batch_collator, CommDataset
from fastreid.config import configurable

from .temporal_transforms import TemporalBeginCrop, TemporalRandomCrop
from .temporal_transforms2 import build_temporal_transforms, TemporalRestrictedTest

from .video_dataset import VideoCommonDataset
from .sampler import BalancedIdentitySamplerV2, WeightedTrackSampler

logger = logging.getLogger("fastreid.build_dataset" + __name__)

__all__ = [
    "build_video_reid_train_loader",
    "build_video_reid_test_loader",
]


def flexible_video_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    # print(type(batched_inputs))
    if isinstance(elem, torch.Tensor):
        out = []
        for i, tensor in enumerate(batched_inputs):
            out.append(tensor)
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: flexible_video_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    # elif isinstance(elem, list):
    #     return torch.tensor(batched_inputs)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs


def build_video_reid_train_loader(cfg):

    logger = logging.getLogger("fastreid."+__name__)
    logger.info("Prepare training set")
    train_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL,
                                              dense=cfg.TEMP.DATA.DENSE, sampling_step=cfg.TEMP.DATA.SAMPLING_STEP)

        if comm.is_main_process():
            dataset.show_train()
        train_items.extend(dataset.train)

    spatial_transforms = build_transforms(cfg, is_train=True)
    # temporal_transforms = TemporalRandomCrop(num=cfg.TEMP.TRAIN.SEQ_SIZE)
    temporal_transforms = build_temporal_transforms(cfg.TEMP.TRAIN.SAMPLER, cfg.TEMP.TRAIN.SEQ_SIZE, cfg.TEMP.TRAIN.STRIDE)


    train_set = VideoCommonDataset(train_items,
                                 spatial_transform=spatial_transforms,
                                 temporal_transform=temporal_transforms)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(train_set))
    elif sampler_name == "NaiveIdentitySampler":
        sampler = samplers.NaiveIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
    elif sampler_name == "BalancedIdentitySampler":
        sampler = samplers.BalancedIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
    elif sampler_name == "SetReWeightSampler":
        set_weight = cfg.DATALOADER.SET_WEIGHT
        sampler = samplers.SetReWeightSampler(train_set.img_items, mini_batch_size, num_instance, set_weight)
    elif sampler_name == "ImbalancedDatasetSampler":
        sampler = samplers.ImbalancedDatasetSampler(train_set.img_items)
    elif sampler_name == "ImbalancedDatasetSampler":
        sampler = samplers.ImbalancedDatasetSampler(train_set.img_items)
    elif sampler_name == "WeightedTrackSampler":
        sampler = WeightedTrackSampler(train_set.img_items, mini_batch_size, num_instance)
    elif sampler_name == "BalancedIdentitySamplerV2":
        sampler = BalancedIdentitySamplerV2(train_set.img_items, mini_batch_size, num_instance)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    data_loader = build_reid_train_loader(cfg, train_set=train_set, sampler=sampler)
    return data_loader



def _build_flexible_video_loader(test_set, test_batch_size, num_query, collate_fn, num_workers=4):



    mini_batch_size = test_batch_size // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoaderX(
        comm.get_local_rank(),
        dataset=test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return test_loader, num_query


def build_video_reid_test_loader(cfg, dataset_name):

    if cfg.TEMP.TEST.ALL:
        logger.info("Test with all frames:")
        # temporal_transforms = TemporalRestrictedTest(cfg.TEMP.TEST.SEQ_SIZE, cfg.TEMP.TEST.STRIDE) # None
        temporal_transforms = None
        collate_fn = flexible_video_batch_collator
    else:
        logger.info("Testing with {} frames:".format(cfg.TEMP.TEST.SEQ_SIZE))
        # temporal_transform = TemporalBeginCrop(num=cfg.TEMP.TEST.SEQ_SIZE)
        temporal_transforms = build_temporal_transforms(cfg.TEMP.TEST.SAMPLER, cfg.TEMP.TEST.SEQ_SIZE, cfg.TEMP.TEST.STRIDE)
        collate_fn = fast_batch_collator

    spatial_transforms = build_transforms(cfg, is_train=False)

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
    if comm.is_main_process():
        dataset.show_test()

    test_items = dataset.query + dataset.gallery
    test_set = VideoCommonDataset(test_items,
                                  spatial_transform=spatial_transforms,
                                  temporal_transform=temporal_transforms,
                                  relabel=False)

    num_query = len(dataset.query)

    test_loader, num_query = _build_flexible_video_loader(test_set,
                                                          cfg.TEST.IMS_PER_BATCH,
                                                          num_query,
                                                          collate_fn,
                                                          cfg.DATALOADER.NUM_WORKERS)
    return test_loader, num_query

# def build_temporal_transforms():
    

