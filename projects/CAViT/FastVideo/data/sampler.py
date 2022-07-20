import copy
import itertools
import logging

import ipdb
import numpy as np
from fastreid.utils import comm
from collections import defaultdict
from typing import Optional, List

from fastreid.data.samplers import triplet_sampler
from fastreid.data.samplers.triplet_sampler import reorder_index, no_index
logger = logging.getLogger('fastreid.' + __name__)


class BalancedIdentitySamplerV2(triplet_sampler.BalancedIdentitySampler):
    def __init__(self, data_source: List, mini_batch_size: int, num_instances: int, seed: Optional[int] = None):
        super(BalancedIdentitySamplerV2, self).__init__(data_source, mini_batch_size, num_instances, seed)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            # Shuffle identity list
            identities = np.random.permutation(self.num_identities)

            # If remaining identities cannot be enough for a batch,
            # just drop the remaining parts
            drop_indices = self.num_identities % (self.num_pids_per_batch * self._world_size)
            if drop_indices: identities = identities[:-drop_indices]

            batch_indices = []
            for kid in identities:
                i = np.random.choice(self.pid_index[self.pids[kid]])
                i_pid, i_cam = self.data_source[i][1], self.data_source[i][2]
                batch_indices.append(i)
                pid_i = self.index_pid[i]
                cams = self.pid_cam[pid_i]
                index = self.pid_index[pid_i]
                select_cams = no_index(cams, i_cam)

                if select_cams:
                    if len(select_cams) >= self.num_instances:
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                    else:
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
                    for kk in cam_indexes:
                        batch_indices.append(index[kk])
                else:
                    select_indexes = no_index(index, i)
                    if not select_indexes:
                        # Only one image for this identity
                        ind_indexes = [0] * (self.num_instances - 1)
                    elif len(select_indexes) >= self.num_instances:
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                    else:
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                    for kk in ind_indexes:
                        batch_indices.append(index[kk])

                if len(batch_indices) == self.batch_size:
                    yield from reorder_index(batch_indices, self._world_size)
                    batch_indices = []


class WeightedTrackSampler(triplet_sampler.BalancedIdentitySampler):
    def __init__(self, data_source: List, mini_batch_size: int, num_instances: int, seed: Optional[int] = None):
        super(WeightedTrackSampler, self).__init__(data_source, mini_batch_size, num_instances, seed)
        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            # trackid = info[3][0]

            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)
        self.possibility = np.zeros(len(data_source)) + 0.01

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            # Shuffle identity list
            identities = np.random.permutation(self.num_identities)

            # If remaining identities cannot be enough for a batch,
            # just drop the remaining parts
            drop_indices = self.num_identities % (self.num_pids_per_batch * self._world_size)
            if drop_indices: identities = identities[:-drop_indices]

            batch_indices = []
            for kid in identities:
                i = np.random.choice(self.pid_index[self.pids[kid]])
                # i_pid, i_cam = self.data_source[i][1], self.data_source[i][2]
                batch_indices.append(i)
                pid_i = self.index_pid[i]
                # cams = self.pid_cam[pid_i]

                index = self.pid_index[pid_i]
                select_indexes = no_index(index, i)

                if not select_indexes:
                    possibility = np.array([1 / (self.num_instances - 1)] * (self.num_instances - 1))
                else:
                    # logger.info(f"local randk {self._rank}")
                    # logger.info(select_indexes)
                    # logger.info(f"seletced {self.possibility}")
                    possibility = self.possibility[select_indexes]
                    if possibility.sum() == 0:
                        possibility = possibility + 0.01

                possibility = possibility / possibility.sum()

                # logger.info(f"index size {len(select_indexes)}, possibility size {len(possibility)}")

                if not select_indexes:
                    # Only one image for this identity
                    ind_indexes = [0] * (self.num_instances - 1)
                elif len(select_indexes) >= self.num_instances and np.nonzero(possibility)[0].shape[0] > self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1,
                                                   replace=False, p=possibility)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1,
                                                   replace=True, p=possibility)

                for kk in ind_indexes:
                    batch_indices.append(index[kk])

                if len(batch_indices) == self.batch_size:
                    yield from reorder_index(batch_indices, self._world_size)
                    batch_indices = []


    def update_weight(self, scores, upper=0.85, lower=0.6):

        logger.info(f"Updating sampling possibility...............")
        for i, score in enumerate(scores):
            if (score < lower) or (score > upper):
                self.possibility[i] = 0.0
            else:
                self.possibility[i] = score  # float(1 - score)

