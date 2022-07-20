import os.path as osp

import ipdb
from scipy.io import loadmat
import glob
import warnings

from fastreid.data.datasets import DATASET_REGISTRY

from .video_bases import VideoPersonDataset
from ..data_utils import read_json, write_json


@DATASET_REGISTRY.register()
class DukeV(VideoPersonDataset):

    def __init__(self, root, min_seq_len=0, dense=False, sampling_step=32, **kwargs):

        self.root = root
        self.dataset_dir = osp.join(root, 'DukeMTMC-VideoReID')

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        if dense == True:
            self.split_train_json_path = osp.join(self.dataset_dir, 'dense_split_train_with-track-id.json')
        else:
            self.split_train_json_path = osp.join(self.dataset_dir, 'split_train_with-track-id.json')

        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query_with-track-id.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery_with-track-id.json')

        self.min_seq_len = min_seq_len
        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        self.check_before_run(required_files)

        if dense:
            train = self._process_dir_dense(self.train_dir, self.split_train_json_path, sampling_step)
        else:
            train = self.process_dir(self.train_dir, self.split_train_json_path)

        query = self.process_dir(self.query_dir, self.split_query_json_path)
        gallery = self.process_dir(self.gallery_dir, self.split_gallery_json_path)


        super(DukeV, self).__init__(train, query, gallery, **kwargs)


    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def process_dir(self, dir_path, json_path):
        if osp.exists(json_path):
            split = read_json(json_path)
            return split['tracklets']

        print('=> Generating split json file (** this might take a while **)')
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print(
            'Processing "{}" with {} person identities'.format(
                dir_path, len(pdirs)
            )
        )

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}

        tracklets = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            new_ambi = pid

            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:

                if '-' in tdir.split('/')[-1]: # 110/158-2
                    # ipdb.set_trace()
                    continue

                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx + 1).zfill(4)
                    res = glob.glob(
                        osp.join(tdir, '*' + img_idx_name + '*.jpg')
                    )
                    if len(res) == 0:
                        warnings.warn(
                            'Index name {} in {} is missing, skip'.format(
                                img_idx_name, tdir
                            )
                        )
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1

                frame_ids = [int(img_path.split('_')[-2][1:]) for img_path in img_paths]
                track_id = [img_path.split('/')[-2] for img_path in img_paths][0]
                
                if track_id == '10-2': track_id=10
                else: track_id = int(track_id)
                
                img_paths = tuple(img_paths)
                frame_ids = tuple(frame_ids)

                tracklets.append((img_paths, pid, camid, track_id, frame_ids, new_ambi))

        print('Saving split to {}'.format(json_path))
        split_dict = {'tracklets': tracklets}
        write_json(split_dict, json_path)

        return tracklets

    def _process_dir_dense(self, dir_path, json_path, sampling_step=32):
        if osp.exists(json_path):
            print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            # return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']
            return split['tracklets']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        # pid2label = {pid:label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            new_ambi = pid

            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

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
                
                # ipdb.set_trace()

                frame_ids = [img_path.split('_')[-2][1:] for img_path in img_paths]
                track_id = [img_path.split('/')[-2] for img_path in img_paths][0]
                
                if track_id == '10-2': track_id=10
                else: track_id = int(track_id)

                img_paths = tuple(img_paths)
                # track_ids = tuple(track_ids)
                frame_ids = tuple(frame_ids)

                # dense sampling
                num_sampling = len(img_paths)//sampling_step
                if num_sampling == 0:
                    tracklets.append((img_paths, pid, camid, track_id, frame_ids, new_ambi))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_paths[idx*sampling_step:], pid, camid, track_id, frame_ids[idx*sampling_step:], new_ambi))
                        else:
                            tracklets.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], pid, camid, track_id,
                                              frame_ids[idx*sampling_step : (idx+1)*sampling_step], new_ambi))

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

        # return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
        return tracklets




if __name__ == '__main__':
    data = DukeMTMCVidReID()

