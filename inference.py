import torch
from tqdm import tqdm
from module.unet import UNet
from collections import defaultdict
import pickle
import numpy as np

class DeepImpute(object):
    def __init__(self):
        self.imputer = self._load_model()


    def _load_model(self):
        result_dict = torch.load('/home/zhaojifan/Impute/ckpt/20191202-191412/10_0.47_0.72_0.90.ckpt')

        net_weight_dict = result_dict['model']

        imputer = UNet(n_channels=1, n_classes=1)
        imputer.load_state_dict(net_weight_dict)

        return imputer

    def _sample_mask(self, shape, index, prob_line=0.05):
        prob = np.random.uniform(0,1,shape)
        mask = np.uint8(prob >= prob_line)
        mask[:, index] = 0
        return mask

    def impute(self, haplotype : np.ndarray or list) -> np.ndarray:
        """
        :parameters
        haplotype : 需要进行impute的数据， list或者numpy.ndarray类型
        """
        f1_score_array = []
        nf_targets = None
        nf_predicts = None
        f_targets = None
        f_predicts = None
        fixlen = haplotype.shape[1]
        sample_num = 1000
        pos_embedding = np.exp([x/fixlen for x in range(fixlen)])

        from sklearn.metrics import f1_score, confusion_matrix, classification_report
        maf = np.load('tmp/maf.npy')
        filter_indexes = np.where(np.logical_and(maf >= 0.05, maf <= 0.95))[0]
        if isinstance(haplotype, list):
            haplotype = np.array(haplotype)
        if haplotype.ndim == 1:
            haplotype = np.expand_dims(haplotype, 0)
        with torch.set_grad_enabled(False):
            self.imputer.eval()
            for i in tqdm(range(haplotype.shape[1])):
                if not maf[i]:
                    f1_score_array.append(-1)
                    continue
                neg_sample = haplotype[haplotype[:, i] == 0, :]
                pos_sample = haplotype[haplotype[:, i] == 1, :]
                if neg_sample.shape[0] == 0 or pos_sample.shape[0] == 0:
                    f1_score_array.append(-1)
                    continue
                if neg_sample.shape[0] < sample_num:
                    neg_over_sample = np.random.choice(neg_sample.shape[0], sample_num-neg_sample.shape[0])
                    neg_over_sample = neg_sample[neg_over_sample, :]
                    neg_over_sample = np.vstack((neg_sample, neg_over_sample))
                else:
                    neg_over_sample = np.random.choice(neg_sample.shape[0], sample_num, replace=False)
                    neg_over_sample = neg_sample[neg_over_sample, :]
                if pos_sample.shape[0] < sample_num:
                    pos_over_sample = np.random.choice(pos_sample.shape[0], sample_num-pos_sample.shape[0])
                    pos_over_sample = pos_sample[pos_over_sample, :]
                    pos_over_sample = np.vstack((pos_sample, pos_over_sample))
                else:
                    pos_over_sample = np.random.choice(pos_sample.shape[0], sample_num, replace=False)
                    pos_over_sample = pos_sample[pos_over_sample]
                #print(neg_over_sample.shape, pos_over_sample.shape)
                over_sample = np.vstack((neg_over_sample, pos_over_sample))
                target = over_sample[:, i].flatten()
                nf_targets = target if  nf_targets is None else np.hstack((nf_targets, target))
                if i in filter_indexes:
                    f_targets = target if  f_targets is None else np.hstack((f_targets, target))
                mask = self._sample_mask(over_sample.shape, i)
                net_input = over_sample * mask + 0.5 * (1 - mask)  + pos_embedding
                net_input = np.expand_dims(net_input, 1)
                net_input = torch.from_numpy(net_input).float()
                net_prob = self.imputer(net_input)
                predict = np.uint8(net_prob.numpy()>=0.5)[:, i].flatten()
                nf_predicts =  predict if nf_predicts is None else np.hstack((nf_predicts, predict))
                if i in filter_indexes:
                    f_predicts =  predict if f_predicts is None else np.hstack((f_predicts, predict))
                if np.all(target) or not np.any(target) or np.all(predict) or not np.any(predict):
                    f1 = np.sum(target == predict) / len(target)
                else:

                    f1 = f1_score(target, predict)
                print('F1_score : {:.2f}   MAF : {:.2f}'.format(f1, maf[i]))
                f1_score_array.append(f1)
            print('='*50 +' 过滤掉位点的结果 ' + '='*50)
            print(confusion_matrix(f_targets, f_predicts))
            print(classification_report(f_targets, f_predicts))
            filter_f1 = np.array(f1_score_array)[filter_indexes]
            filter_f1 = filter_f1[filter_f1 != -1]
            print('[>=0.99]({:.4f})  [>=0.96]({:.4f})  [>=0.9]({:.4f})'.format(np.sum(filter_f1>=0.99)/len(filter_f1),
                                                                                np.sum(filter_f1>=0.96)/len(filter_f1),
                                                                                np.sum(filter_f1>=0.9)/len(filter_f1)))
            print('')
            print('='*50 +' 全部位点的结果 ' + '='*50)
            print(confusion_matrix(nf_targets, nf_predicts))
            print(classification_report(nf_targets, nf_predicts))
            filter_f1 = np.array(f1_score_array)
            filter_f1 = filter_f1[filter_f1 != -1]
            print('[>=0.99]({:.4f})  [>=0.96]({:.4f})  [>=0.9]({:.4f})'.format(np.sum(filter_f1>=0.99)/len(filter_f1),
                                                                                np.sum(filter_f1>=0.96)/len(filter_f1),
                                                                                np.sum(filter_f1>=0.9)/len(filter_f1)))
if __name__ == "__main__":
    data = np.load('tmp/compare.npy')
    data = data[int(data.shape[0] * 0.93) + 1:,:]
    #test_data = data[:int(data.shape[0] * 0.8) + 1, :][-5:,...]
    #test_data = np.random.uniform(0,1,(5, 500))
    imputer = DeepImpute()
    imputer.impute(data)

