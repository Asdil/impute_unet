# -*- coding : utf-8 -*-

import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset

class GeneDataset(Dataset):
    def __init__(self, phase='training', split_rate=0.85, dataflag='compare', fixlen=500, seed=888):
        self.phase = phase
        self.split_rate = split_rate
        self.dataflag = dataflag
        self.fixlen = fixlen
        self.seed = 1024
        print('加载数据')
        self.trainset, self.testset = self._split_data()
        print('数据信息 ： 训练集{}     测试集{}'.format(self.trainset.shape, self.testset.shape))
        self.testset = self.testset[:10000, :]
        self.test_mask = self._sample_test_masks(500)
    
    # def _over_sampling(self):
    #     sub_maf = np.sum(self.trainset, axis=0) / self.trainset.shape[0]
    #     rare_index = np.where(np.logical_and(sub_maf >)) 

    def _split_data(self):
        if self.dataflag == 'compare':
            data = np.load('tmp/compare.npy')
            self.maf = np.sum(data, axis=0) / data.shape[0]
            self.filter_indexes = np.where(np.logical_or(self.maf <0.05, self.maf > 0.95))[0]
            self.restrict_index = np.where(np.logical_or(self.maf <= 0.2, self.maf >=0.8))[0]
            if not os.path.exists('tmp/maf.npy'):
                np.save('tmp/maf.npy', maf)
            train_data = data[:int(data.shape[0] * self.split_rate) + 1, :]
            test_data = data[int(data.shape[0] * self.split_rate) + 1:, :]
            return train_data, test_data
        elif self.dataflag == 'random':
            data = np.load('tmp/impute_data.npy')
            data = data.T
            random.seed(self.seed)
            np.random.seed(self.seed)
            start = random.randint(0, data.shape[1] - self.fixlen)
            data = data[:, start : start+self.fixlen]
            self.maf = np.sum(data, axis=0) / data.shape[0]
            self.filter_indexes = np.where(np.logical_or(self.maf <0.05, self.maf > 0.95))[0]
            self.restrict_index = np.where(np.logical_or(self.maf <= 0.2, self.maf >=0.8))[0]
            print('低频最小：{}   低频最大：{}    删选比例：{:.4f}'.format(np.min(self.maf),np.max(self.maf), len(self.filter_indexes) / data.shape[1]))
            np.random.shuffle(data)
            np.random.seed(None)
            random.seed(None)
            return data[:int(data.shape[0] * self.split_rate),:], data[int(data.shape[0] * self.split_rate):, :]

    def _augment(self, x):
        aug_prob = random.uniform(0, 1)
        if aug_prob >= 0.5:
            allele_prob = np.random.uniform(0, 1, len(x))
            aug_index = np.where(allele_prob > 0.5)[0]
            aug_index = np.array(list(set(list(aug_index)) & set(list(self.restrict_index))))
            x[aug_index] = 1
        return x

    def _sample_mask(self, m, prob_line=0.05):
        prob = np.random.uniform(0, 1, m)
        mask = np.uint8(prob >= prob_line)
        return mask
    
    def _sample_test_masks(self, m, prob_line=0.05):
        np.random.seed(self.seed)
        prob = np.random.uniform(0, 1, (10000,m))
        mask = np.uint8(prob >= prob_line)
        return mask
    

    def _sample_noise(self, m, noise_type='uniform'):
        if noise_type == 'uniform':
            #noise = np.random.uniform(0,1, m)
            noise = np.ones(m) * 0.5
        elif noise_type == 'guassian':
            noise = np.random.normal(0, 1, m)
        else:
            raise ValueError('only support UNIFORM or GUASSIAN noise')
        return noise

    def __getitem__(self, index):
        if self.phase == 'training':
            snp = self.trainset[index, :]
            mask = self._sample_mask(len(snp))
            #snp = self._augment(snp)
        else:
            snp  =self.testset[index, :]
            mask = self.test_mask[index, :]
        inp = snp.copy()
        #inp[inp == 0] = 0.1
        #inp[inp == 1] = 0.9
        
        noise = self._sample_noise(len(inp))
        inp = mask * inp + 0.5 * (1 - mask) + np.exp([x/self.fixlen for x in range(self.fixlen)]) + self.maf
        inp = np.reshape(inp, (1,  -1))

        inp = torch.from_numpy(inp).float()
        snp = torch.from_numpy(snp).float()
        return inp, snp, mask

    def __len__(self):
        return self.trainset.shape[0] if self.phase == 'training'  else self.testset.shape[0]
