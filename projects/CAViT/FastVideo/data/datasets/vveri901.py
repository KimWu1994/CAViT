import glob
import os.path as osp
import logging

from fastreid.data.datasets import DATASET_REGISTRY

from ..data_utils import read_json
from .video_bases import VideoPersonDataset
logger = logging.getLogger('fastreid.' + __name__)

@DATASET_REGISTRY.register()
class VVERI901(VideoPersonDataset):
    """VVERI901_V1_Trial.

    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and
        Discriminative Classification. SCIA 2011.

    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_
    
    Dataset statistics:
        - identities: 200.
        - tracklets: 400.
        - cameras: 2.
    """

    def __init__(self, root='', split_id=0, dense=False, sampling_step=32, **kwargs):
        
        self.root = root
        self.dataset_dir = osp.join(root, 'VVERI901_V1_Trial')

        train_file = osp.join(self.dataset_dir, 'train.txt')
        query_file = osp.join(self.dataset_dir, 'query.txt')
        gallery_file = osp.join(self.dataset_dir, 'gallery.txt')

        with open(train_file, 'r') as f:
            lines = f.readlines()
            train_dirs = [line.strip()[2:] for line in lines]
        
        with open(query_file, 'r') as f:
            lines = f.readlines()
            query_dirs = [line.strip()[2:] for line in lines]
        
        with open(gallery_file, 'r') as f:
            lines = f.readlines()
            gallery_dirs = [line.strip()[2:] for line in lines]

        if dense:
            train = self.process_dense_dir(train_dirs, sampling_step=sampling_step)
        else:
            train = self.process_dir(train_dirs)

        query = self.process_dir(query_dirs)
        gallery = self.process_dir(gallery_dirs)

        super(VVERI901, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dirnames):
        tracklets = []
        # dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            person_dir = osp.join(self.root, dirname)
            img_names = glob.glob(osp.join(person_dir, '*.png'))
            assert len(img_names) > 0


            # img_names: /disk1/jinlin/DATA/VVERI901_V1_Trial/2120/L09_C31_V07/21553.png
            # dirnames:  /disk1/jinlin/DATA/VVERI901_V1_Trial/2120/L09_C31_V07
            img_names = tuple(img_names)

            pid = int(dirname.split('/')[-2])
            cid = int(dirname.split('_')[-2][1:])
            frames = [int(name.split('/')[-1].split('.')[0]) for name in img_names]
            # pid = dirname2pid[dirname]
            
            trackid = cid
            new_ambi = pid
            
            tracklets.append((img_names, pid, cid, trackid, frames, new_ambi))

        return tracklets

    def process_dense_dir(self, dirnames, sampling_step=32):
        tracklets = []
        # dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            person_dir = osp.join(self.root, dirname)
            img_names = glob.glob(osp.join(person_dir, '*.png'))
            assert len(img_names) > 0


            # img_names: /disk1/jinlin/DATA/VVERI901_V1_Trial/2120/L09_C31_V07/21553.png
            # dirnames:  /disk1/jinlin/DATA/VVERI901_V1_Trial/2120/L09_C31_V07
            img_names = tuple(img_names)

            pid = int(dirname.split('/')[-2])
            cid = int(dirname.split('_')[-2][1:])
            frames = [int(name.split('/')[-1].split('.')[0]) for name in img_names]
            # pid = dirname2pid[dirname]
            
            trackid = cid
            new_ambi = pid
            # dense sampling
            num_sampling = len(img_names) // sampling_step
            if num_sampling == 0:
                tracklets.append((img_names, pid, cid, frames))
            else:
                for idx in range(num_sampling):
                    if idx == num_sampling - 1:
                        tracklets.append((img_names[idx * sampling_step:], pid, cid, trackid, frames[idx * sampling_step:], new_ambi))
                    else:
                        tracklets.append((img_names[idx * sampling_step: (idx + 1) * sampling_step], pid, cid, trackid, frames[idx * sampling_step:], new_ambi))


        return tracklets
  

  