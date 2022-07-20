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
from tools.utils import mkdir_if_missing, write_json, read_json


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
    for idx, (img_paths, pid, camid) in enumerate(dataset):
        # start index
        vid2clip_index[idx, 0] = len(new_dataset)
        # process the sequence that can be divisible by seq_len*stride
        for i in range(len(img_paths)//(seq_len*stride)):
            for j in range(stride):
                begin_idx = i * (seq_len * stride) + j
                end_idx = (i + 1) * (seq_len * stride)
                clip_paths = img_paths[begin_idx : end_idx : stride]
                assert(len(clip_paths) == seq_len)
                new_dataset.append((clip_paths, pid, camid))
        # process the remaining sequence that can't be divisible by seq_len*stride        
        if len(img_paths)%(seq_len*stride) != 0:
            # reducing stride
            new_stride = (len(img_paths)%(seq_len*stride)) // seq_len
            for i in range(new_stride):
                begin_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + i
                end_idx = len(img_paths) // (seq_len*stride) * (seq_len*stride) + seq_len * new_stride
                clip_paths = img_paths[begin_idx : end_idx : new_stride]
                assert(len(clip_paths) == seq_len)
                new_dataset.append((clip_paths, pid, camid))
            # process the remaining sequence that can't be divisible by seq_len
            # if len(img_paths) % seq_len != 0:
            if len(img_paths) % seq_len != 0:
                clip_paths = img_paths[len(img_paths)//seq_len*seq_len:]
                # loop padding
                while len(clip_paths) < seq_len:
                    for index in clip_paths:
                        if len(clip_paths) >= seq_len:
                            break
                        clip_paths.append(index)
                assert(len(clip_paths) == seq_len)
                new_dataset.append((clip_paths, pid, camid))
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
    for (img_paths, pid, camid) in dataset:
        if sampling_step != 0:
            num_sampling = len(img_paths)//sampling_step
            if num_sampling == 0:
                new_dataset.append((img_paths, pid, camid))
            else:
                for idx in range(num_sampling):
                    if idx == num_sampling - 1:
                        new_dataset.append((img_paths[idx*sampling_step:], pid, camid))
                    else:
                        new_dataset.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], pid, camid))
        else:
            new_dataset.append((img_paths, pid, camid))

    return new_dataset


class MARS(object):
    """
    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    
    Note: 
    # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
    # gallery imgs with label=-1 can be remove, which do not influence on final performance.

    """
    def __init__(self, root='/data/datasets/', sampling_step=64, seq_len=16, stride=4, DL=False, **kwargs):
        self.DL = DL
        if DL:
            self.root = osp.join(root, 'MARS-DL')
        else:
            self.root = osp.join(root, 'MARS')
        self.train_name_path = osp.join(self.root, 'info/train_name.txt')
        self.test_name_path = osp.join(self.root, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self.root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self.root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self.root, 'info/query_IDX.mat')
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True)
        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False)

        train_dense = densesampling_for_trainingset(train, sampling_step)
        recombined_query, query_vid2clip_index = recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = recombination_for_testset(gallery, seq_len=seq_len, stride=stride)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  -----------------------------------")
        print("  subset      | # ids | # tracklets")
        print("  -----------------------------------")
        print("  train       | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  train_dense | {:5d} | {:8d}".format(num_train_pids, len(train_dense)))
        print("  query       | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery     | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  -----------------------------------")
        print("  total       | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  -----------------------------------")

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
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if self.DL and pid == 0: continue
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            
            # img_paths = tuple(img_paths)
            tracklets.append((img_paths, int(pid), int(camid)))
            num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class DukeMTMCVidReID(object):
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
            if relabel: pid = pid2label[pid]
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


class iLIDSVID(object):
    """
    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
    
    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
    """
    def __init__(self, root='/data/datasets/', split_id=9, sampling_step=32, seq_len=16, stride=4, **kwargs):
        self.root = osp.join(root, 'iLIDS-VID')
        self.data_dir = osp.join(self.root, 'i-LIDS-VID')
        self.split_dir = osp.join(self.root, 'train-test people splits')
        self.split_mat_path = osp.join(self.split_dir, 'train_test_splits_ilidsvid.mat')
        self.split_path = osp.join(self.root, 'splits.json')
        self.cam_1_path = osp.join(self.root, 'i-LIDS-VID/sequences/cam1')
        self.cam_2_path = osp.join(self.root, 'i-LIDS-VID/sequences/cam2')
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_data(test_dirs, cam1=False, cam2=True)

        train_dense = densesampling_for_trainingset(train, sampling_step)
        recombined_query, query_vid2clip_index = recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = recombination_for_testset(gallery, seq_len=seq_len, stride=stride)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded")
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
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids/2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = list(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = list(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class PRID(object):
    """
    PRID

    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
    
    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """

    def __init__(self, root='/data/datasets/', split_id=9, sampling_step=32, seq_len=16, stride=4, min_seq_len=0, **kwargs):
        self.root = osp.join(root, 'prid2011')
        self.split_path = osp.join(self.root, 'splits_prid2011.json')
        self.cam_a_path = osp.join(self.root, 'prid_2011', 'multi_shot', 'cam_a')
        self.cam_b_path = osp.join(self.root, 'prid_2011', 'multi_shot', 'cam_b')

        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_data(train_dirs, cam1=True, cam2=True, min_seq_len=min_seq_len)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_data(test_dirs, cam1=True, cam2=False, min_seq_len=min_seq_len)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_data(test_dirs, cam1=False, cam2=True, min_seq_len=min_seq_len)

        train_dense = densesampling_for_trainingset(train, sampling_step)
        recombined_query, query_vid2clip_index = recombination_for_testset(query, seq_len=seq_len, stride=stride)
        recombined_gallery, gallery_vid2clip_index = recombination_for_testset(gallery, seq_len=seq_len, stride=stride)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
        print("Dataset statistics:")
        print("  ---------------------------------")
        print("  subset   | # ids    | # tracklets")
        print("  ---------------------------------")
        print("  train    | {:5d}    | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  train_dense | {:5d} | {:8d}".format(num_train_pids, len(train_dense)))
        print("  query    | {:5d}    | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d}    | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ---------------------------------")
        print("  total    | {:5d}    | {:8d}".format(num_total_pids, num_total_tracklets))
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
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_data(self, dirnames, cam1=True, cam2=True, min_seq_len=21):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            person_dir_a = osp.join(self.cam_a_path, dirname)
            person_dir_b = osp.join(self.cam_b_path, dirname)
            img_names_a = glob.glob(osp.join(person_dir_a, '*.png'))
            img_names_b = glob.glob(osp.join(person_dir_b, '*.png'))

            if len(img_names_a)<min_seq_len or len(img_names_b)<min_seq_len:
                continue

            if cam1:
                img_names_a = list(img_names_a)
                pid = dirname2pid[dirname]
                tracklets.append((img_names_a, pid, 0))
                num_imgs_per_tracklet.append(len(img_names_a))

            if cam2:
                img_names_b = list(img_names_b)
                pid = dirname2pid[dirname]
                tracklets.append((img_names_b, pid, 1))
                num_imgs_per_tracklet.append(len(img_names_b))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


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
    def __init__(self, root='/data/datasets/', sampling_step=64, seq_len=16, stride=4, **kwargs):
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

        print("=> LS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset       | # ids | # tracklets")
        print("  ------------------------------")
        print("  train        | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  train_dense  | {:5d} | {:8d}".format(num_train_pids, len(train_dense)))
        # print("  val_query    | {:5d} | {:8d}".format(num_val_query_pids, num_val_query_tracklets))
        # print("  val_gallery  | {:5d} | {:8d}".format(num_val_gallery_pids, num_val_gallery_tracklets))
        print("  test_query   | {:5d} | {:8d}".format(num_test_query_pids, num_test_query_tracklets))
        print("  test_gallery | {:5d} | {:8d}".format(num_test_gallery_pids, num_test_gallery_tracklets))
        print("  ------------------------------")
        print("  total        | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

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
            _, _, camid, _ = osp.basename(img_paths[0]).split('_')
            camid = int(camid)

            if relabel:
                pid = pid2label[pid]
            camid -= 1  # index starts from 0

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

