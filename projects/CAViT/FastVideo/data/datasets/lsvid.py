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
class LSVID(object):
    """
    Reference:
    Li J, Wang J, Tian Q, Gao W and Zhang S Global-Local Temporal Representations for Video Person Re-Identification[J]. ICCV, 2019

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 11310 (gallery)
    # cameras: 15

    Note:
    # gallery set must contain query set

    """
    def __init__(self, root='/data/datasets/', sampling_step=64, dense=True, seq_len=16, stride=4, **kwargs):
        self.root = osp.join(root, 'LS-VID')
        self.train_name_path = osp.join(self.root, 'list_sequence/list_seq_train.txt')
        # self.val_name_path = osp.join(self.root, 'list_sequence/list_seq_val.txt')
        self.test_name_path = osp.join(self.root, 'list_sequence/list_seq_test.txt')
        # self.val_query_IDX_path = osp.join(self.root, 'test/data/info_val.mat')
        self.test_query_IDX_path = osp.join(self.root, 'test/data/info_test.mat')
        self.split_json_path = osp.join(self.root, 'data_path')
        self._check_before_run()

        if not osp.exists(self.split_json_path):
            # prepare meta data
            tracklet_train = self._get_names(self.train_name_path)
            # tracklet_val = self._get_names(self.val_name_path)
            tracklet_test = self._get_names(self.test_name_path)
            # val_query_IDX = h5py.File(self.val_query_IDX_path, mode='r')['query'][0,:]
            test_query_IDX = h5py.File(self.test_query_IDX_path, mode='r')['query'][0,:]
            # val_query_IDX = np.array(val_query_IDX, dtype=int)
            test_query_IDX = np.array(test_query_IDX, dtype=int)
            # val_query_IDX -= 1  # index from 0
            test_query_IDX -= 1  # index from 0
            # tracklet_val_query = tracklet_val[val_query_IDX, :]
            tracklet_test_query = tracklet_test[test_query_IDX, :]
            # val_gallery_IDX = [i for i in range(tracklet_val.shape[0]) if i not in val_query_IDX]
            test_gallery_IDX = [i for i in range(tracklet_test.shape[0]) if i not in test_query_IDX]
            # tracklet_val_gallery = tracklet_val[val_gallery_IDX, :]
            tracklet_test_gallery = tracklet_test[test_gallery_IDX, :]
            
            train, num_train_tracklets, num_train_pids, num_train_imgs = \
                self._process_data(tracklet_train, home_dir='tracklet_train', relabel=True)
            # val_query, num_val_query_tracklets, num_val_query_pids, num_val_query_imgs = \
            #     self._process_data(tracklet_val_query, home_dir='tracklet_val', relabel=False)
            # val_gallery, num_val_gallery_tracklets, num_val_gallery_pids, num_val_gallery_imgs = \
            #     self._process_data(tracklet_val_gallery, home_dir='tracklet_val', relabel=False)
            test_query, num_test_query_tracklets, num_test_query_pids, num_test_query_imgs = \
                self._process_data(tracklet_test_query, home_dir='tracklet_test', relabel=False)
            test_gallery, num_test_gallery_tracklets, num_test_gallery_pids, num_test_gallery_imgs = \
                self._process_data(tracklet_test_gallery, home_dir='tracklet_test', relabel=False)

            print("Saving dataset to {}".format(self.split_json_path))
            dataset_dict = {
                'train': train,
                'num_train_tracklets': num_train_tracklets,
                'num_train_pids': num_train_pids,
                'num_train_imgs': num_train_imgs,
                'test_query': test_query,
                'num_test_query_tracklets': num_test_query_tracklets,
                'num_test_query_pids': num_test_query_pids,
                'num_test_query_imgs': num_test_query_imgs,
                'test_gallery': test_gallery,
                'num_test_gallery_tracklets': num_test_gallery_tracklets,
                'num_test_gallery_pids': num_test_gallery_pids,
                'num_test_gallery_imgs': num_test_gallery_imgs,
            }
            write_json(dataset_dict, self.split_json_path)
        else:
            dataset = read_json(self.split_json_path)
            train = dataset['train']
            num_train_tracklets = dataset['num_train_tracklets']
            num_train_pids = dataset['num_train_pids']
            num_train_imgs = dataset['num_train_imgs']
            test_query = dataset['test_query']
            num_test_query_tracklets = dataset['num_test_query_tracklets']
            num_test_query_pids = dataset['num_test_query_pids']
            num_test_query_imgs = dataset['num_test_query_imgs']
            test_gallery = dataset['test_gallery']
            num_test_gallery_tracklets = dataset['num_test_gallery_tracklets']
            num_test_gallery_pids = dataset['num_test_gallery_pids']
            num_test_gallery_imgs = dataset['num_test_gallery_imgs']

        train_dense = densesampling_for_trainingset(train, sampling_step)
        recombined_test_query, query_vid2clip_index = recombination_for_testset(test_query, seq_len=seq_len, stride=stride)
        recombined_test_gallery, gallery_vid2clip_index = recombination_for_testset(test_gallery, seq_len=seq_len, stride=stride)

        num_imgs_per_tracklet = num_train_imgs + num_test_gallery_imgs + num_test_query_imgs #+ num_val_query_imgs + num_val_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_test_gallery_pids # + num_val_gallery_pids 
        num_total_tracklets = num_train_tracklets + num_test_gallery_tracklets + num_test_query_tracklets #+ num_val_query_tracklets + num_val_gallery_tracklets
        
        
        self.num_train_pids = num_train_pids
        self.num_train_tracklets = num_train_tracklets

       
        self.num_test_query_tracklets = num_test_query_tracklets
        self.num_test_gallery_tracklets = num_test_gallery_tracklets
        
        self.num_total_pids = num_total_pids
        self.num_total_tracklets = num_total_tracklets

        self.min_num = min_num
        self.max_num = max_num
        self.avg_num = avg_num


        
        self.train = train
        # self.val_query = val_query
        # self.val_gallery = val_gallery
        self.query = test_query
        self.gallery = test_gallery

        self.train_dense = train_dense
        self.recombined_query = recombined_test_query
        self.recombined_gallery = recombined_test_gallery
        self.query_vid2clip_index = query_vid2clip_index
        self.gallery_vid2clip_index = gallery_vid2clip_index

        self.num_train_pids = num_train_pids
        # self.num_val_query_pids = num_val_query_pids
        # self.num_val_gallery_pids = num_val_gallery_pids
        self.num_query_pids = num_test_query_pids
        self.num_gallery_pids = num_test_gallery_pids

        if dense:
            self.train = self.train_dense

        # super(LSVID, self).__init__(self.train, self.query, self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        # if not osp.exists(self.val_name_path):
        #     raise RuntimeError("'{}' is not available".format(self.val_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        # if not osp.exists(self.val_query_IDX_path):
        #     raise RuntimeError("'{}' is not available".format(self.val_query_IDX_path))
        if not osp.exists(self.test_query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.test_query_IDX_path))
    
    def show_train(self):
            print("=> LS-VID loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset       | # ids | # tracklets")
            print("  ------------------------------")
            print("  train        | {:5d} | {:8d}".format(self.num_train_pids, self.num_train_tracklets))
            print("  train_dense  | {:5d} | {:8d}".format(self.num_train_pids, len(self.train_dense)))
            # print("  val_query    | {:5d} | {:8d}".format(num_val_query_pids, num_val_query_tracklets))
            # print("  val_gallery  | {:5d} | {:8d}".format(num_val_gallery_pids, num_val_gallery_tracklets))
            print("  test_query   | {:5d} | {:8d}".format(self.num_query_pids, self.num_test_query_tracklets))
            print("  test_gallery | {:5d} | {:8d}".format(self.num_gallery_pids, self.num_test_gallery_tracklets))
            print("  ------------------------------")
            print("  total        | {:5d} | {:8d}".format(self.num_total_pids, self.num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(self.min_num, self.max_num, self.avg_num))
            print("  ------------------------------")
    
    def show_test(self):
            print("=> LS-VID loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset       | # ids | # tracklets")
            print("  ------------------------------")
            print("  train        | {:5d} | {:8d}".format(self.num_train_pids, self.num_train_tracklets))
            print("  train_dense  | {:5d} | {:8d}".format(self.num_train_pids, len(self.train_dense)))
            # print("  val_query    | {:5d} | {:8d}".format(num_val_query_pids, num_val_query_tracklets))
            # print("  val_gallery  | {:5d} | {:8d}".format(num_val_gallery_pids, num_val_gallery_tracklets))
            print("  test_query   | {:5d} | {:8d}".format(self.num_query_pids, self.num_test_query_tracklets))
            print("  test_gallery | {:5d} | {:8d}".format(self.num_gallery_pids, self.num_test_gallery_tracklets))
            print("  ------------------------------")
            print("  total        | {:5d} | {:8d}".format(self.num_total_pids, self.num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(self.min_num, self.max_num, self.avg_num))
            print("  ------------------------------")


    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                basepath, pid = new_line.split(' ')
                names.append([basepath, int(pid)])
        return np.array(names)

    def _process_data(self, meta_data, home_dir=None, relabel=False):
        assert home_dir in ['tracklet_train', 'tracklet_val', 'tracklet_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 1].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {int(pid): label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            tracklet_path = osp.join(self.root, meta_data[tracklet_idx, 0]) + '*'
            img_paths = glob.glob(tracklet_path)  # avoid .DS_Store
            img_paths.sort()
            pid = int(meta_data[tracklet_idx, 1])
            trackid, _, camid, _ = osp.basename(img_paths[0]).split('_')
            camid = int(camid)
            frame = [int(osp.basename(img_path).split('_')[-3]) for img_path in img_paths]

            # if relabel:
            #     pid = pid2label[pid]
            camid -= 1  # index starts from 0
            new_ambi = pid

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, trackid, frame, new_ambi))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet



# lsvid = LSVID( root='/disk1/jinlin/DATA')
