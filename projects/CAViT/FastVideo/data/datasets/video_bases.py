# encoding: utf-8

import logging
from tabulate import tabulate
from termcolor import colored
from fastreid.data.datasets.bases import Dataset

logger = logging.getLogger('fastreid.' + __name__)


class VideoPersonDataset(Dataset):
    def __init__(self, train, query, gallery, **kwargs):
        super(VideoPersonDataset, self).__init__(train, query, gallery, **kwargs)

    def show_train(self):
        num_train_pids, num_train_cams, num_imgs_per_tracklet = self.parse_data(self.train)
        num_imgs = sum(num_imgs_per_tracklet)
        min_img_num = min(num_imgs_per_tracklet)
        max_img_num = max(num_imgs_per_tracklet)
        avg_img_num = num_imgs // len(num_imgs_per_tracklet)

        headers = ['subset', '# ids', '# images', '# tracklets', '# cameras']
        csv_results = [['train', num_train_pids, num_imgs, len(self.train), num_train_cams]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )

        stat_headers = ['number of images per track', '# min', '# max ', '# average']
        stat_csv_results = [['train', min_img_num, max_img_num, avg_img_num]]

        # tabulate it
        stat_table = tabulate(
            stat_csv_results,
            tablefmt="pipe",
            headers=stat_headers,
            numalign="left",
        )

        logger = logging.getLogger("fastreid." + __name__)
        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))
        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(stat_table, "cyan"))

    def show_test(self):
        num_query_pids, num_query_cams, num_imgs_per_tracklet_query = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams, num_imgs_per_tracklet_gallery = self.parse_data(self.gallery)

        num_query_imgs = sum(num_imgs_per_tracklet_query)
        num_gallery_imgs = sum(num_imgs_per_tracklet_gallery)
        headers = ['subset', '# ids', '# images', '# tracklets', '# cameras']
        csv_results = [
            ['query', num_query_pids, num_query_imgs, len(self.query), num_query_cams],
            ['gallery', num_gallery_pids, num_gallery_imgs, len(self.gallery), num_gallery_cams],
        ]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )

        min_img_num_query = min(num_imgs_per_tracklet_query)
        max_img_num_query = max(num_imgs_per_tracklet_query)
        avg_img_num_query = num_query_imgs // len(num_imgs_per_tracklet_query)

        min_img_num_gallery = min(num_imgs_per_tracklet_gallery)
        max_img_num_gallery = max(num_imgs_per_tracklet_gallery)
        avg_img_num_gallery = num_query_imgs // len(num_imgs_per_tracklet_gallery)

        stat_headers = ['number of images per track', '# min', '# max', '# average']
        stat_csv_results = [
            ['query', min_img_num_query, max_img_num_query, avg_img_num_query],
            ['gallery', min_img_num_gallery, max_img_num_gallery, avg_img_num_gallery],
        ]

        # tabulate it
        stat_table = tabulate(
            stat_csv_results,
            tablefmt="pipe",
            headers=stat_headers,
            numalign="left",
        )

        logger = logging.getLogger("fastreid." + __name__)
        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))
        logger.info(f"=> Static information {self.__class__.__name__} in csv format: \n" + colored(stat_table, "cyan"))

    def parse_data(self, data):
        """
        Args:
            data (list): contains tuples of (img_paths, pid, camid)
        """
        pids = set()
        cams = set()
        # imgs = 0
        num_imgs_per_tracklet = []
        for info in data:
            pids.add(info[1])
            cams.add(info[2])
            # imgs = imgs + len(info[0])
            num_imgs_per_tracklet.append(len(info[0]))
        return len(pids), len(cams), num_imgs_per_tracklet



