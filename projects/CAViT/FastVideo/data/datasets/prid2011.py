import glob
import os.path as osp

from fastreid.data.datasets import DATASET_REGISTRY

from ..data_utils import read_json
from .video_bases import VideoPersonDataset


@DATASET_REGISTRY.register()
class PRID2011(VideoPersonDataset):
    """PRID2011.

    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and
        Discriminative Classification. SCIA 2011.

    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_
    
    Dataset statistics:
        - identities: 200.
        - tracklets: 400.
        - cameras: 2.
    """
    dataset_dir = 'Prid2011'
    dataset_url = None

    def __init__(self, root='', split_id=0, dense=False, sampling_step=32, **kwargs):

        self.dataset_dir = osp.join(root, 'Prid2011')

        self.split_path = osp.join(self.dataset_dir, 'splits_prid2011.json')

        self.cam_a_dir = osp.join(self.dataset_dir, 'multi_shot', 'cam_a')
        self.cam_b_dir = osp.join(self.dataset_dir, 'multi_shot', 'cam_b')

        required_files = [self.dataset_dir, self.cam_a_dir, self.cam_b_dir]
        self.check_before_run(required_files)

        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                .format(split_id,
                        len(splits) - 1)
            )
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']

        if dense:
            train = self.process_dense_dir(train_dirs, cam1=True, cam2=True,
                                           sampling_step=sampling_step)
        else:
            train = self.process_dir(train_dirs, cam1=True, cam2=True)

        query = self.process_dir(test_dirs, cam1=True, cam2=False)
        gallery = self.process_dir(test_dirs, cam1=False, cam2=True)

        super(PRID2011, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        # dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)

                pid = int(dirname.split('_')[-1])
                # pid = dirname2pid[dirname]
                
                trackid = 0
                new_ambi = pid
                frames = [int(img_name.split('/')[-1].split('.')[0]) for img_name in img_names]
                tracklets.append((img_names, pid, 0, trackid, frames, new_ambi))

            if cam2:
                person_dir = osp.join(self.cam_b_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                
                pid = int(dirname.split('_')[-1])
                # pid = dirname2pid[dirname]
                
                trackid = 0
                new_ambi = pid
                frames = [int(img_name.split('/')[-1].split('.')[0]) for img_name in img_names]

                tracklets.append((img_names, pid, 1, trackid, frames, new_ambi))

        return tracklets

    def process_dense_dir(self, dirnames, cam1=True, cam2=True, sampling_step=32):
        tracklets = []
        # dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                # pid = dirname2pid[dirname]
                pid = int(dirname.split('_')[-1])
                camid = 0
                
                trackid = 0
                new_ambi = pid
                frames = [int(img_name.split('/')[-1].split('.')[0]) for img_name in img_names]

                # dense sampling
                num_sampling = len(img_names) // sampling_step
                if num_sampling == 0:
                    tracklets.append((img_names, pid, camid, trackid, frames, new_ambi))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_names[idx * sampling_step:], pid, camid, trackid, frames[idx * sampling_step:], new_ambi))
                        else:
                            tracklets.append((img_names[idx * sampling_step: (idx + 1) * sampling_step], pid, camid, trackid, frames[idx * sampling_step:], new_ambi))

            if cam2:
                person_dir = osp.join(self.cam_b_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                # pid = dirname2pid[dirname]
                pid = int(dirname.split('_')[-1])
                camid = 1
                
                trackid = 1
                new_ambi = pid
                frames = [int(img_name.split('/')[-1].split('.')[0]) for img_name in img_names]

                # dense sampling
                num_sampling = len(img_names) // sampling_step
                
                if num_sampling == 0:
                    tracklets.append((img_names, pid, camid, trackid, frames, new_ambi))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_names[idx * sampling_step:], pid, camid, trackid, frames[idx * sampling_step:], new_ambi))
                        else:
                            tracklets.append((img_names[idx * sampling_step: (idx + 1) * sampling_step], pid, camid, trackid, frames[idx * sampling_step:], new_ambi))


        return tracklets
      


