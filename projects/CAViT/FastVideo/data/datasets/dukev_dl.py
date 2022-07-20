import os.path as osp

import ipdb
from scipy.io import loadmat
import glob
import warnings

from fastreid.data.datasets import DATASET_REGISTRY

from .video_bases import VideoPersonDataset
from ..data_utils import read_json, write_json



# encoding: utf-8
import glob
import re
import json
import pickle
import os
import os.path as osp
from scipy.io import loadmat
# from .bases import BaseVideoDataset
import pandas as pd
import numpy as np



class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def get_videodata_info(self, data, return_tracklet_stats=False):
        pids, cams, tracklet_stats = [], [], []

        for item in data:
            img_paths, pid, camid = item[0], item[1], item[2]
            pids += [pid]
            cams += [camid]
            tracklet_stats += [len(img_paths)]

 
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_tracklets = len(data)
        if return_tracklet_stats:
            return num_pids, num_tracklets, num_cams, tracklet_stats
        return num_pids, num_tracklets, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class BaseVideoDataset(BaseDataset):
    def __init__(self, train, query, gallery):
        super().__init__()
        self.train = train
        self.query = query 
        self.gallery = gallery

    def show_train(self):
        num_train_pids, num_train_tracklets, num_train_cams, train_tracklet_stats = \
            self.get_videodata_info(self.train, return_tracklet_stats=True)

        num_query_pids, num_query_tracklets, num_query_cams, query_tracklet_stats = \
            self.get_videodata_info(self.query, return_tracklet_stats=True)

        num_gallery_pids, num_gallery_tracklets, num_gallery_cams, gallery_tracklet_stats = \
            self.get_videodata_info(self.gallery, return_tracklet_stats=True)

        tracklet_stats = train_tracklet_stats + query_tracklet_stats + gallery_tracklet_stats
        min_num = np.min(tracklet_stats)
        max_num = np.max(tracklet_stats)
        avg_num = np.mean(tracklet_stats)

        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset   | # ids | # tracklets | # cameras")
        print("  -------------------------------------------")
        print("  train    | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_cams))
        print("  query    | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_cams))
        print("  gallery  | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_cams))
        print("  -------------------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.2f}".format(min_num, max_num, avg_num))
        print("  -------------------------------------------")
    
    def show_test(self):
        num_train_pids, num_train_tracklets, num_train_cams, train_tracklet_stats = \
            self.get_videodata_info(self.train, return_tracklet_stats=True)

        num_query_pids, num_query_tracklets, num_query_cams, query_tracklet_stats = \
            self.get_videodata_info(self.query, return_tracklet_stats=True)

        num_gallery_pids, num_gallery_tracklets, num_gallery_cams, gallery_tracklet_stats = \
            self.get_videodata_info(self.gallery, return_tracklet_stats=True)

        tracklet_stats = train_tracklet_stats + query_tracklet_stats + gallery_tracklet_stats
        min_num = np.min(tracklet_stats)
        max_num = np.max(tracklet_stats)
        avg_num = np.mean(tracklet_stats)

        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset   | # ids | # tracklets | # cameras")
        print("  -------------------------------------------")
        print("  train    | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_cams))
        print("  query    | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_cams))
        print("  gallery  | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_cams))
        print("  -------------------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.2f}".format(min_num, max_num, avg_num))
        print("  -------------------------------------------")


@DATASET_REGISTRY.register()
class DukeV_DL(BaseVideoDataset):
    # dataset_dir = 'DukeMTMC-VideoReID'
    

    def __init__(self, root='/disk1/jinlin/DATA', verbose=True, min_seq_len =0, sampling_step=32, dense=False, new_eval=False, **kwargs):
        
        self.dataset_dir = 'DukeMTMC-VideoReID-DL'
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir,'train')
        self.gallery_dir = osp.join(self.dataset_dir,'gallery')
        self.query_dir = osp.join(self.dataset_dir,'query')
        self.min_seq_len = min_seq_len
        self.sampling_step = sampling_step
        self.dense = dense

        info_dir='DukeV_info'
        #for self-created duke info
        if 'DL' in self.dataset_dir:
            info_dir = osp.join(self.dataset_dir, info_dir)
        
        if dense:
            self.train_pkl = osp.join(info_dir,'train_dense.pkl')
        else:
            self.train_pkl = osp.join(info_dir,'train.pkl')

        self.gallery_pkl = osp.join(info_dir,'gallery.pkl')
        self.query_pkl = osp.join(info_dir,'query.pkl')
        self.info_dir = info_dir
        self._check_before_run()

        if 'DL' in self.dataset_dir:
            train_mask_csv = pd.read_csv(osp.join(self.dataset_dir,'duke_mask_info.csv'),sep=',',header=None).values
            query_mask_csv = pd.read_csv(osp.join(self.dataset_dir,'duke_mask_info_query.csv'),sep=',',header=None).values
            gallery_mask_csv = pd.read_csv(osp.join(self.dataset_dir,'duke_mask_info_test.csv'),sep=',',header=None).values
        else:
            train_mask_csv,query_mask_csv, gallery_mask_csv = None,None,None
        
        if self.dense:
            train = self._process_dir(self.train_dir,self.train_pkl, dense=True, mask_info=None)
        else:
            train = self._process_dir(self.train_dir,self.train_pkl, dense=False, mask_info=None)
            
        gallery = self._process_dir(self.gallery_dir,self.gallery_pkl, dense=False, mask_info=None)
        query = self._process_dir(self.query_dir,self.query_pkl, dense=False, mask_info=None)
        if verbose:
            print("=> DukeV loaded")
            # self.print_dataset_statistics(train, query, gallery)
        self.train = train # list of tuple--(paths,id,cams)
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_tracklets, self.num_train_cams = self.get_videodata_info(self.train)
        self.num_query_pids, self.num_query_tracklets, self.num_query_cams = self.get_videodata_info(self.query)
        self.num_gallery_pids, self.num_gallery_tracklets, self.num_gallery_cams = self.get_videodata_info(self.gallery)

        super(DukeV_DL, self).__init__(train, query, gallery)


    def _process_dir(self, dir_path, pkl_path, dense=False, mask_info=None):

        if osp.exists(pkl_path):
            print('==> %s exisit. Load...'%(pkl_path))
            with open(pkl_path,'rb') as f:
                pkl_file = pickle.load(f)
            
            if mask_info is None:
                return pkl_file

            tracklets = []
            start = 0
            for info in pkl_file:
                end = start + len(info[0])
                tracklets.append((info[0],info[1],info[2],mask_info[start:end,1:].astype('int16')//16))
                start = end
            return tracklets

        pdirs = sorted(glob.glob(osp.join(dir_path, '*')))
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))
        pids = sorted(list(set([int(osp.basename(pdir)) for pdir in pdirs])))
        # pid2label = {pid : label for label,pid in enumerate(pids)}

        tracklets = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            # if relabel : pid = pid2label[pid]
            track_dirs = sorted(glob.glob(osp.join(pdir,'*')))
            for track_dir in track_dirs:
                img_paths = sorted(glob.glob(osp.join(track_dir,'*.jpg')))
                num_imgs = len(img_paths)
                if num_imgs < self.min_seq_len :
                    continue
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1 :
                    camid = int(img_name[5])-1
                else:
                    camid = int(img_name[6])-1
                img_paths = tuple(img_paths)
                
                frame_ids = [int(img_path.split('_')[-2][1:]) for img_path in img_paths]
                track_id = [img_path.split('/')[-2] for img_path in img_paths][0]

                if track_id == '10-2': track_id=10
                else: track_id = int(track_id)
                
                new_ambi = pid

                if dense:
                    ## dense sampling
                    num_sampling = len(img_paths) // self.sampling_step
                    if num_sampling == 0:
                        if len(img_paths) >= self.min_seq_len:
                            img_paths = tuple(img_paths)
                            tracklets.append((img_paths, pid, camid, track_id, frame_ids, new_ambi))
                    else:
                        for idx in range(num_sampling):
                            if idx == num_sampling - 1:
                                tracklets.append(
                                    (img_paths[idx * self.sampling_step:], pid, camid, track_id, frame_ids[idx * self.sampling_step:], new_ambi))
                            else:
                                tracklets.append((img_paths[idx * self.sampling_step: (idx + 1) * self.sampling_step], pid, camid,
                                                track_id, frame_ids[idx * self.sampling_step: (idx + 1) * self.sampling_step], new_ambi))

                else:
                    if len(img_paths) >= self.min_seq_len:
                        img_paths = tuple(img_paths)
                        tracklets.append((img_paths, pid, camid, track_id, frame_ids, new_ambi))

                # tracklets.append((img_paths, pid, camid, track_id, frame_ids, new_ambi))  
                # (img_paths, pid, camid, track_id, frame_ids, new_ambi)


        # save to pickle
        if not osp.isdir(self.info_dir):
            os.mkdir(self.info_dir)
        with open(pkl_path,'wb') as f:
            pickle.dump(tracklets,f)
        return tracklets

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))


# dukev = DukeV(root='/disk1/jinlin/DATA', verbose=True, min_seq_len =0, new_eval=True)

# if __name__ == '__main__':
#     data = DukeMTMCVidReID_DL()

