# -*- coding : utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix

def evaluationByMask(predict, groundtruth, mask, filter_indexes, verbose=False):
    predict = torch.sigmoid(predict.detach()).numpy()
    predict[predict >= 0.5] = 1
    predict[predict < 0.5] = 0
    predict = np.uint8(predict)
    groundtruth = groundtruth.numpy()
    groundtruth[groundtruth >= 0.5] = 1
    groundtruth[groundtruth < 0.5] = 0
    groundtruth = np.uint8(groundtruth)
    mask = np.uint8(mask.numpy())
    filter_index_score, not_filter_index_score = [], []
    filter_pd,filter_gt, not_filter_pd, not_filter_gt = None, None, None, None
    mask, predict, groundtruth = mask.T, predict.T, groundtruth.T
    for i, (pmask, pred, ground) in enumerate(zip(mask, predict, groundtruth)):
        pmask, pred, ground = pmask.flatten(), pred.flatten(), ground.flatten()
        if np.all(pmask):
            continue
        #print(ground[pmask == 0], pred[pmask == 0])
        if np.all(ground) or np.all(ground==0):
            not_filter_index_score.append(np.sum(ground == pred) / len(ground))
        else:
            not_filter_index_score.append(f1_score(ground[pmask == 0], pred[pmask == 0]))
        not_filter_pd = pred if not_filter_pd is None else np.hstack((not_filter_pd, pred))
        not_filter_gt = ground if not_filter_gt is None else np.hstack((not_filter_gt, ground))
        if i not in filter_indexes:
            if np.all(ground) or np.all(ground==0):
                filter_index_score.append(np.sum(ground == pred) / len(ground))
            else:
                filter_index_score.append(f1_score(ground[pmask == 0], pred[pmask == 0]))
            filter_pd = pred if filter_pd is None else np.hstack((filter_pd, pred))
            filter_gt = ground if filter_gt is None else np.hstack((filter_gt, ground))

    filter_index_score, not_filter_index_score = np.array(filter_index_score), np.array(not_filter_index_score)
    filter_mean, not_filter_mean = np.mean(filter_index_score), np.mean(not_filter_index_score)
    filter_99, filter_96, filter_90 = np.sum(filter_index_score >= 0.99) / len(filter_index_score), \
                                    np.sum(filter_index_score >= 0.96) / len(filter_index_score),   \
                                    np.sum(filter_index_score >= 0.9) / len(filter_index_score)

    notfilter_99, notfilter_96, notfilter_90 = np.sum(not_filter_index_score >= 0.99) / len(not_filter_index_score), \
                                    np.sum(not_filter_index_score >= 0.96) / len(not_filter_index_score),            \
                                    np.sum(not_filter_index_score >= 0.9) / len(not_filter_index_score)

    if verbose:
        print('未经过MAF过滤的混淆矩阵：')
        print(confusion_matrix(not_filter_gt, not_filter_pd))
        print('经过MAF过滤的混淆矩阵：')
        print(confusion_matrix(filter_gt, filter_pd))
    return (filter_mean, filter_99, filter_96, filter_90), (not_filter_mean, notfilter_99, notfilter_96, notfilter_90), \
            filter_index_score, not_filter_index_score, filter_gt, filter_pd


