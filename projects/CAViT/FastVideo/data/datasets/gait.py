import os
import os.path as osp

import numpy as np

# from .data_set import DataSet


def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()

    if dataset == 'OU-LP':
        pid_num = int(pid_num*0.6)
    for _label in sorted(list(os.listdir(dataset_path.replace("./", "../../")))):
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if dataset == 'CASIA-B' and _label == '005':
            continue
        label_path = osp.join(dataset_path.replace("./", "../../"), _label)
        for _seq_type in sorted(list(os.listdir(label_path))):
            seq_type_path = osp.join(label_path, _seq_type)
            for _view in sorted(list(os.listdir(seq_type_path))):
                _seq_dir = osp.join(seq_type_path, _view)
                seqs = os.listdir(_seq_dir)
                if len(seqs) > 0:
                    seq_dir.append([_seq_dir])
                    label.append(_label)
                    seq_type.append(_seq_type)
                    view.append(_view)

    pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
        dataset, pid_num, pid_shuffle))
    if not osp.exists(pid_fname):
        pid_list = sorted(list(set(label)))
        if pid_shuffle:
            np.random.shuffle(pid_list)
        pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
        os.makedirs('partition', exist_ok=True)
        np.save(pid_fname, pid_list)

    # modify the default parameters of np.load
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    pid_list = np.load(pid_fname)
    np.load = np_load_old

    train_list = pid_list[0]
    test_list = pid_list[1]
    
    train_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        [label[i] for i, l in enumerate(label) if l in train_list],
        [seq_type[i] for i, l in enumerate(label) if l in train_list],
        [view[i] for i, l in enumerate(label)
         if l in train_list],
        cache, resolution)

    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label)
         if l in test_list],
        cache, resolution)

    return train_source, test_source



import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import cv2
import xarray as xr

from fastreid.data.datasets import DATASET_REGISTRY

from .video_bases import VideoPersonDataset
from ..data_utils import read_json, write_json

@DATASET_REGISTRY.register()
class CASIA_B(VideoPersonDataset):
    """CASIA_B.

    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and
        Discriminative Classification. SCIA 2011.

    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_
    
    Dataset statistics:
        - identities: 200.
        - tracklets: 400.
        - cameras: 2.
    """

    def __init__(self, root='', split_id=73, dense=False, sampling_step=32, **kwargs):
        
        self.root = root
        self.dataset_dir = osp.join(root, 'CASIA-B', 'GaitDatasetB-pretreatmented')
        self.split_id = split_id

        if dense:
            train = self.process_dense_dir(self.dataset_dir, sampling_step=sampling_step, train=True)
        else:
            train = self.process_dir(self.dataset_dir, train=True)
        
        test = self.process_dir(self.dataset_dir, train=False)

        # self.train = train
        # self.test = test

        super(CASIA_B, self).__init__(train, test, test, **kwargs)


    def process_dir(self, dirnames, train=True):
        
        tracklets = []
        dirnames = sorted(list(os.listdir(self.dataset_dir)))
        
        if train: dirnames = dirnames[:self.split_id+1]
        else: dirnames = dirnames[self.split_id+1:]

        for dirname in dirnames:
            if dirname == '005':
                continue

            # person_dir = osp.join(self.root, dirname)
            # img_names = glob.glob(osp.join(person_dir, '*.png'))
            # assert len(img_names) > 0

            label_path = osp.join(self.dataset_dir,  dirname)

            for _seq_type in sorted(list(os.listdir(label_path))):
                seq_type_path = osp.join(label_path, _seq_type)
                for _view in sorted(list(os.listdir(seq_type_path))):
                    _seq_dir = osp.join(seq_type_path, _view)
                    seqs = os.listdir(_seq_dir)
                    if len(seqs) > 0:
                        # seq_dir.append([_seq_dir])
                        # label.append(_label)
                        # seq_type.append(_seq_type)
                        # view.append(_view)
                        # 
                        # ipdb.set_trace()
                        img_paths = [ os.path.join(_seq_dir, seq) for seq in seqs]
                        frames = [int(seq.split('-')[-1].split('.')[0]) for seq in seqs]
                        pid = int(dirname)
                        # cid = self.view_dict[_view]
                        # data, frame_set, self.view[index], self.seq_type[index], self.label[index], index
                        track = (img_paths, pid, _view, _seq_type, frames)
                        tracklets.append(track)
        
        return tracklets


    def process_dense_dir(self, dirnames, sampling_step=32, train=True):
 
        tracklets = []
        dirnames = sorted(list(os.listdir(self.dataset_dir)))
        
        if train: dirnames = dirnames[:self.split_id+1]
        else: dirnames = dirnames[self.split_id+1:]


        for dirname in dirnames:

            # person_dir = osp.join(self.root, dirname)
            # img_names = glob.glob(osp.join(person_dir, '*.png'))
            # assert len(img_names) > 0

            if dirname == '005':
                continue
            
            label_path = osp.join(self.dataset_dir,  dirname)

            for _seq_type in sorted(list(os.listdir(label_path))):
                seq_type_path = osp.join(label_path, _seq_type)
                for _view in sorted(list(os.listdir(seq_type_path))):
                    _seq_dir = osp.join(seq_type_path, _view)
                    seqs = os.listdir(_seq_dir)
                    if len(seqs) > 0:
                        # seq_dir.append([_seq_dir])
                        # label.append(_label)
                        # seq_type.append(_seq_type)
                        # view.append(_view)
                        img_paths = [os.path.join(_seq_dir, seq) for seq in seqs]
                        frames = [int(seq.split('-')[-1].split('.')[0]) for seq in seqs]
                        pid = int(dirname)
                        # cid = self.view_dict[_view]

                        num_sampling = len(img_paths) // sampling_step
                        if num_sampling == 0:
                            tracklets.append((img_paths, pid, _view, _seq_type, frames))
                        else:
                            for idx in range(num_sampling):
                                if idx == num_sampling - 1:
                                    track = (img_paths[idx * sampling_step:], pid, _view, _seq_type, frames[idx * sampling_step:])
                                else:
                                    track = (img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid,_view, _seq_type, frames[idx * sampling_step:])
                                
                                tracklets.append(track)

        return tracklets
  


        