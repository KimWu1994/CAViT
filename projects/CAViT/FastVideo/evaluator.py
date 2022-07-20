import copy
import logging
import itertools
from collections import OrderedDict

import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist

from fastreid.evaluation.query_expansion import aqe
from fastreid.evaluation.rank import evaluate_rank
from fastreid.evaluation.roc import evaluate_roc

from fastreid.evaluation.reid_evaluation import ReidEvaluator

logger = logging.getLogger('fastreid.'+__name__)

class VideoReidEvaluator(ReidEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        super(VideoReidEvaluator, self).__init__(cfg, num_query, output_dir=output_dir)


    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        features = []
        pids = []
        camids = []
        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])

        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]

        if ("MARS" in self.cfg.DATASETS.TESTS) or  ("MARSDL" in self.cfg.DATASETS.TESTS) or ("LSVID" in self.cfg.DATASETS.TESTS):
            logger.info('merge query')
            gallery_features = features
            gallery_pids = pids
            gallery_camids = camids
        else:
            # gallery features, person ids and camera ids
            gallery_features = features[self._num_query:]
            gallery_pids = pids[self._num_query:]
            gallery_camids = camids[self._num_query:]


        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        if self.cfg.TEST.ROC.ENABLED:
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)



class NewVideoReidEvaluator(VideoReidEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        super(NewVideoReidEvaluator, self).__init__(cfg, num_query, output_dir=output_dir)

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'pids': inputs['targets'].to(self._cpu_device),
            'camids': inputs['camids'].to(self._cpu_device),
            'ambis': inputs['ambis'].to(self._cpu_device)

        }
        self._predictions.append(prediction)

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        features = []
        pids = []
        camids = []
        ambis = []
        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])
            ambis.append(prediction['ambis'])

        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()
        ambis = torch.cat(ambis, dim=0).numpy()

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]
        query_ambis = ambis[: self._num_query]
        
        #  ('iLIDSVID' in self.cfg.DATASETS.TESTS) or
        if ("MARS" in self.cfg.DATASETS.TESTS) or  ("MARSDL" in self.cfg.DATASETS.TESTS) or ("LSVID" in self.cfg.DATASETS.TESTS):
            logger.info('merge query')
            gallery_features = features
            gallery_pids = pids
            gallery_camids = camids
            gallery_ambis = ambis
        else:
            # gallery features, person ids and camera ids
            gallery_features = features[self._num_query:]
            gallery_pids = pids[self._num_query:]
            gallery_camids = camids[self._num_query:]
            gallery_ambis = ambis[self._num_query:]

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)
        logger.info('new test eval for ID switch.......')
        new_cmc, new_all_ap = new_eval_func(dist, query_pids, gallery_pids, query_camids, gallery_camids, query_ambis, gallery_ambis)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        new_mAP = np.mean(new_all_ap)
        for r in [1, 5]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100

        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        self._results['NewRank-1'] = new_cmc[0] * 100
        self._results['New_mAP'] = new_mAP * 100


        if self.cfg.TEST.ROC.ENABLED:
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)



def new_eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_ambis, g_ambis, max_rank=50):
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis])#.astype(np.int32)

    new_eval = (q_ambis is not None) and (g_ambis is not None)
    if new_eval:
        matches_am2id = (g_ambis[indices] == q_pids[:, np.newaxis])
        matches_id2am = (g_pids[indices] == q_ambis[:, np.newaxis])
        matches_am2am = (g_ambis[indices] == q_ambis[:, np.newaxis])
        matches = matches | matches_am2am | matches_am2id | matches_id2am
    matches = matches.astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_camids[order] == q_camid)
        if not new_eval:
            remove = remove & (g_pids[order] == q_pid)
        else:
            q_amb = q_ambis[q_idx]
            remove_dis = remove & (g_pids[order] == 0) # distractor with same cam
            remove_id2id = remove & (g_pids[order] == q_pid)
            remove_am2id = remove & (g_ambis[order] == q_pid)
            remove_am2am = remove & (g_ambis[order] == q_amb)
            remove_id2am = remove & (g_pids[order] == q_amb)
            remove = remove_dis | remove_id2id | remove_am2id | remove_am2am | remove_id2am

        # remove = remove | (g_pids[order] == -1)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    # mAP = np.mean(all_AP)
    return all_cmc, all_AP



class GaitEvaluator(VideoReidEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        super(GaitEvaluator, self).__init__(cfg, num_query, output_dir=output_dir)

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'pids': inputs['targets'].to(self._cpu_device),
            'view': inputs['view'],
            'seq_type': inputs['seq_type']

        }
        self._predictions.append(prediction)
    
    def acc_vis(self, acc):
        # ipdb.set_trace()
        # include, excl?ude = [], []
        if acc.shape[0] == 3:
            # Print rank-1 accuracy of the best model
            for i in range(1):
                logger.info('===Rank-%d (Include identical-view cases)===' % (i + 1))
                logger.info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                    np.mean(acc[0, :, :, i]),
                    np.mean(acc[1, :, :, i]),
                    np.mean(acc[2, :, :, i])))
                
                include = [np.mean(acc[0, :, :, i]), np.mean(acc[1, :, :, i]), np.mean(acc[2, :, :, i])]

            # Print rank-1 accuracy of the best model，excluding identical-view cases
            for i in range(1):
                logger.info('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
                logger.info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                    de_diag(acc[0, :, :, i]),
                    de_diag(acc[1, :, :, i]),
                    de_diag(acc[2, :, :, i])))
                
                exclude = [np.mean(de_diag(acc[0, :, :, i])), np.mean(de_diag(acc[1, :, :, i])), np.mean(de_diag(acc[2, :, :, i]))]

            # Print rank-1 accuracy of the best model (Each Angle)
            np.set_printoptions(precision=2, floatmode='fixed')
            for i in range(1):
                logger.info('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
                logger.info('NM: %s', str(de_diag(acc[0, :, :, i], True)))
                logger.info('BG: %s', str(de_diag(acc[1, :, :, i], True)))
                logger.info('CL: %s', str(de_diag(acc[2, :, :, i], True)))

        elif acc.shape[0] == 1:
            # Print rank-1 accuracy of the best model
            for i in range(1):
                logger.info('===Rank-%d (Include identical-view cases)===' % (i + 1))
                logger.info('NM: %.3f' % (
                    np.mean(acc[0, :, :, i])))

            # Print rank-1 accuracy of the best model，excluding identical-view cases
            for i in range(1):
                logger.info('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
                logger.info('NM: %.3f' % (
                    de_diag(acc[0, :, :, i])))

            # Print rank-1 accuracy of the best model (Each Angle)
            np.set_printoptions(precision=2, floatmode='fixed')
            for i in range(1):
                logger.info('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
                logger.info('NM: %s', str(de_diag(acc[0, :, :, i], True)))
            
        return include, exclude
        


    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        features = []
        pids = []
        views = []
        seq_types = []
        # ipdb.set_trace()
        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            views.extend(prediction['view'])
            seq_types.extend(prediction['seq_type'])

        features = torch.cat(features, dim=0).numpy()
        pids = torch.cat(pids, dim=0).numpy()
        # views = torch.cat(views, dim=0).numpy()
        # seq_types = torch.cat(seq_types, dim=0).numpy()


        self._results = OrderedDict()
        acc = gaitevaluation(features, pids, views, seq_types, self.cfg.DATASETS.TESTS[0])
        include, exclude = self.acc_vis(acc)
        self._results['in_NM'] = include[0]
        self._results['in_BG'] = include[1]
        self._results['in_CL'] = include[2]

        self._results['ex_NM'] = exclude[0]
        self._results['ex_BG'] = exclude[1]
        self._results['ex_CL'] = exclude[2]



        # if self.cfg.TEST.AQE.ENABLED:
        #     logger.info("Test with AQE setting")
        #     qe_time = self.cfg.TEST.AQE.QE_TIME
        #     qe_k = self.cfg.TEST.AQE.QE_K
        #     alpha = self.cfg.TEST.AQE.ALPHA
        #     query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        # dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        # if self.cfg.TEST.RERANK.ENABLED:
        #     logger.info("Test with rerank setting")
        #     k1 = self.cfg.TEST.RERANK.K1
        #     k2 = self.cfg.TEST.RERANK.K2
        #     lambda_value = self.cfg.TEST.RERANK.LAMBDA

        #     if self.cfg.TEST.METRIC == "cosine":
        #         query_features = F.normalize(query_features, dim=1)
        #         gallery_features = F.normalize(gallery_features, dim=1)

        #     rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
        #     dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        # cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)
        # logger.info('new test eval for ID switch.......')
        # new_cmc, new_all_ap = new_eval_func(dist, query_pids, gallery_pids, query_camids, gallery_camids, query_ambis, gallery_ambis)

        # mAP = np.mean(all_AP)
        # mINP = np.mean(all_INP)
        # new_mAP = np.mean(new_all_ap)
        # for r in [1, 5]:
        #     self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100

        # self._results['mAP'] = mAP * 100
        # self._results['mINP'] = mINP * 100
        # self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        # self._results['NewRank-1'] = new_cmc[0] * 100
        # self._results['New_mAP'] = new_mAP * 100


        # if self.cfg.TEST.ROC.ENABLED:
        #     scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
        #     fprs, tprs, thres = metrics.roc_curve(labels, scores)

        #     for fpr in [1e-4, 1e-3, 1e-2]:
        #         ind = np.argmin(np.abs(fprs - fpr))
        #         self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)





def gaitevaluation(features, pids, view, seq_type, dataset):
    # dataset = config['dataset']#.split('-')[0]
    # feature, view, seq_type, label = data
    label = np.array(pids)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    
    # ipdb.set_trace()
    probe_seq_dict = {'CASIA_B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OU_LP': [['Seq00']]}
    gallery_seq_dict = {'CASIA_B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OU_LP': [['Seq01']]}

    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    print(acc.shape)
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = features[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = features[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = compute_cosine_distance(probe_x, gallery_x)
                    idx = dist.sort(1)[1].cpu().numpy()
                    if len(gallery_y) == 0:
                        print('Zero !!! ', p, v1, v2, len(gallery_y))
                        acc[p, v1, v2, :] = 0
                        continue
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(
                            np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]],
                            1) > 0, 0)
                        * 100 / dist.shape[0], 2)

    return acc


def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1)
    result = result / (result.shape[0]-1.0)
    if not each_angle:
        result = np.mean(result)
    return result



def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist

@torch.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = torch.from_numpy(features).cuda()
    others = torch.from_numpy(others).cuda()
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    
    return dist_m   # dist_m.cpu().numpy()

