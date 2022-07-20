# encoding: utf-8
import torch
from torch.utils.data import Dataset

from fastreid.data.data_utils import read_image


class VideoCommonDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items,
                 spatial_transform=None,
                 temporal_transform=None,
                 relabel=True):

        self.img_items = img_items
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_paths = img_item[0]
        pid       = int(img_item[1])
        camid     = int(img_item[2])
        trackids  = img_item[3]
        frameids  = img_item[4]
        ambi = int(img_item[5])

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)
            # trackids   = self.temporal_transform(trackids)
            frameids = self.temporal_transform(frameids)
            # print(trackids)

        imgs = [read_image(img_path) for img_path in img_paths]
        if self.spatial_transform is not None:
            imgs = [self.spatial_transform(img) for img in imgs]

        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
            ambi = self.pid_dict[ambi]

        imgs = torch.stack(imgs, 0) # [t, c, h, w]
        # trackids = torch.tensor(trackids)
        # print(frameids)
        frameids = torch.tensor(frameids)

        return {
            "images": imgs,
            "targets": pid,
            "camids": camid,
            "track_ids": trackids,
            "frame_ids": frameids,
            'ambis': ambi
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


class GaitDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items,
                 spatial_transform=None,
                 temporal_transform=None,
                 relabel=True):

        self.img_items = img_items
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.relabel = relabel

        pid_set = set()
        # cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            # cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        # self.cams = sorted(list(cam_set))
        
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            # self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        # img_paths, pid, _view, _seq_type, frames
        img_paths = img_item[0]
        pid       = int(img_item[1])
        view      = img_item[2]
        seq_type  = img_item[3]
        frames    = img_item[4]
        

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)
            # trackids   = self.temporal_transform(trackids)
            frames = self.temporal_transform(frames)
            # print(trackids)

        imgs = [read_image(img_path) for img_path in img_paths]
        if self.spatial_transform is not None:
            imgs = [self.spatial_transform(img) for img in imgs]

        if self.relabel:
            pid = self.pid_dict[pid]
            # camid = self.cam_dict[camid]
            # ambi = self.pid_dict[ambi]

        imgs = torch.stack(imgs, 0) # [t, c, h, w]
        # trackids = torch.tensor(trackids)
        # print(frameids)
        frames = torch.tensor(frames)

        return {
            "images": imgs,
            "targets": pid,
            "view": view,
            "seq_type": seq_type,
            "frame_ids": frames
        }

    @property
    def num_classes(self):
        return len(self.pids)

    # @property
    # def num_cameras(self):
    #     return len(self.cams)


