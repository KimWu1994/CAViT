import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import h5py
import math
import os.path as osp
import numpy as np
from scipy.io import loadmat
from fastreid.data.datasets import DATASET_REGISTRY

# from tools.utils import mkdir_if_missing, write_json, read_json
from .dukev_dl import BaseVideoDataset
from ..data_utils import mkdir_if_missing, write_json, read_json


def recombination_for_testset(dataset, seq_len=16, stride=4):
    ''' Split all videos in test set into lots of equilong clips.

    Args:
        dataset (list): input dataset, each video is organized as (img_paths, pid, camid)
        seq_len (int): sequence length of each output clip
        stride (int): temporal sampling stride

    Returns:
        new_dataset (list): output dataset with lots of equilong clips
        vid2clip_index (list): a list contains the start and end clip index of each original video
    '''
    new_dataset = []
    vid2clip_index = np.zeros((len(dataset), 2), dtype=int)
    for idx, (img_paths, pid, camid, trackid, frame, new_ambi) in enumerate(dataset):
        # start index
        vid2clip_index[idx, 0] = len(new_dataset)
        # process the sequence that can be divisible by seq_len*stride
        for i in range(len(img_paths)//(seq_len*stride)):
            for j in range(stride):
                begin_idx = i * (seq_len * stride) + j
                end_idx = (i + 1) * (seq_len * stride)
                clip_paths = img_paths[begin_idx : end_idx : stride]
                clip_frame = frame[begin_idx : end_idx : stride]
                assert(len(clip_paths) == seq_len)
                assert(len(clip_frame) == seq_len)
                new_dataset.append((clip_paths, pid, camid, trackid, clip_frame, new_ambi))
        # process the remaining sequence that can't be divisible by seq_len*stride        
        if len(img_paths)%(seq_len*stride) != 0:
            # reducing stride
            new_stride = (len(img_paths)%(seq_len*stride)) // seq_len
            for i in range(new_stride):
                begin_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + i
                end_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + seq_len * new_stride
                clip_paths = img_paths[begin_idx : end_idx : new_stride]
                clip_frame = frame[begin_idx : end_idx : stride]
                assert(len(clip_paths) == seq_len)
                new_dataset.append((clip_paths, pid, camid, trackid, clip_frame, new_ambi))
            # process the remaining sequence that can't be divisible by seq_len
            # if len(img_paths) % seq_len != 0:
            if len(img_paths) % seq_len != 0:
                clip_paths = img_paths[len(img_paths)//seq_len*seq_len:]
                clip_frame = frame[len(img_paths)//seq_len*seq_len:]
                # loop padding
                while len(clip_paths) < seq_len:
                    for index in clip_paths:
                        if len(clip_paths) >= seq_len:
                            break
                        clip_paths.append(index)
                assert(len(clip_paths) == seq_len)
                new_dataset.append((clip_paths, pid, camid, trackid, clip_frame, new_ambi))
        # end index
        vid2clip_index[idx, 1] = len(new_dataset)
        assert((vid2clip_index[idx, 1]-vid2clip_index[idx, 0]) == math.ceil(len(img_paths)/seq_len))

    return new_dataset, vid2clip_index.tolist()


def densesampling_for_trainingset(dataset, sampling_step=64):
    ''' Split all videos in training set into lots of clips for dense sampling.

    Args:
        dataset (list): input dataset, each video is organized as (img_paths, pid, camid)
        sampling_step (int): sampling step for dense sampling

    Returns:
        new_dataset (list): output dataset
    '''
    new_dataset = []
    for (img_paths, pid, camid, trackid, frame, new_ambi) in dataset:
        if sampling_step != 0:
            num_sampling = len(img_paths)//sampling_step
            if num_sampling == 0:
                new_dataset.append((img_paths, pid, camid, trackid, frame, new_ambi))
            else:
                for idx in range(num_sampling):
                    if idx == num_sampling - 1:
                        new_dataset.append((img_paths[idx*sampling_step:], pid, camid, trackid, frame[idx*sampling_step:], new_ambi))
                    else:
                        new_dataset.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], pid, camid, trackid, frame[idx*sampling_step:], new_ambi))
        else:
            new_dataset.append((img_paths, pid, camid, trackid, frame, new_ambi))

    return new_dataset

@DATASET_REGISTRY.register()


class DukeV2(object):
    """
    Reference:
    Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning. CVPR 2018.

    """
    def __init__(self, root='/data/datasets/', sampling_step=64, seq_len=16, stride=4, **kwargs):
        self.dataset_dir = osp.join(root, 'DukeMTMC-VideoReID')
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.split_train_json_path = osp.join(self.dataset_dir, 'split_train.json')
        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery.json')

        self._check_before_run()
        print("Note: if root path is changed, the previously generated json files need to be re-generated (so delete them first)")

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_dir(self.train_dir, self.split_train_json_path, relabel=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_dir(self.query_dir, self.split_query_json_path, relabel=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_dir(self.gallery_dir, self.split_gallery_json_path, relabel=False)

        print("the number of tracklets under dense sampling for train set: {}".format(num_train_tracklets_dense))

        train_dense = densesampling_for_trainingset(train, sampling_step)
        recombined_query, query_vid2clip_index = recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = recombination_for_testset(gallery, seq_len=seq_len, stride=stride)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets
        
        print("=> DukeMTMC-VideoReID loaded")
        print("Dataset statistics:")
        print("  ---------------------------------")
        print("  subset      | # ids | # tracklets")
        print("  ---------------------------------")
        print("  train       | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  train_dense | {:5d} | {:8d}".format(num_train_pids, len(train_dense)))
        print("  query       | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery     | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ---------------------------------")
        print("  total       | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ---------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.train_dense = train_dense
        self.recombined_query = recombined_query
        self.recombined_gallery = recombined_gallery
        self.query_vid2clip_index = query_vid2clip_index
        self.gallery_vid2clip_index = gallery_vid2clip_index

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            # if relabel: pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)
                num_imgs_per_tracklet.append(num_imgs)
                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx+1).zfill(4)
                    res = glob.glob(osp.join(tdir, '*' + img_idx_name + '*.jpg'))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))

        num_pids = len(pid_container)
        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {
            'tracklets': tracklets,
            'num_tracklets': num_tracklets,
            'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet,
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet



# lsvid = LSVID( root='/disk1/jinlin/DATA')
