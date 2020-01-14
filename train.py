# -*- coding: utf-8 -*-

from dataset import GeneDataset
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from module.unet import UNet
from utils import evaluationByMask
from colorama import Fore, Back
import torch
import numpy as np
import time
import os
from sklearn.metrics import f1_score, confusion_matrix, classification_report

class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        loss = - self.alpha*(1 - pt) ** self.gamma * target * torch.log(pt) - \
            (1-self.alpha)*pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def save_model(save_path, model, epoch, total_filter_metrics):
    torch.save({
                'model': model.state_dict(),
                }, os.path.join(save_path, '{}_{:.2f}_{:.2f}_{:.2f}.ckpt'.format(epoch,
                                                                        np.sum(total_filter_metrics >=0.99)/len(total_filter_metrics),
                                                                        np.sum(total_filter_metrics >= 0.96)/len(total_filter_metrics),
                                                                        np.sum(total_filter_metrics >= 0.90) /len(total_filter_metrics))))




def train(num_epochs, batch_size):
    trainset, valset = GeneDataset(phase = 'training'), GeneDataset(phase = 'validation')
    trainloader, valloader = DataLoader(dataset=trainset, batch_size=batch_size,  pin_memory=True, num_workers=16), \
                            DataLoader(dataset=valset, batch_size=10000, pin_memory=True, num_workers=16)
    filter_indexes = trainset.filter_indexes


    net = UNet(n_channels=1, n_classes=1)
    state_dict = torch.load('ckpt/20191202-191412/10_0.47_0.72_0.90.ckpt')
    net.load_state_dict(state_dict['model'])
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.BCELoss()#MSELoss(reduction='mean')
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    now = int(time.time())
    timeStruct = time.localtime(now)
    save_dir = time.strftime("%Y%m%d-%H%M%S", timeStruct)
    save_path = os.path.join('ckpt', save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)



    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        accuracy = 0.0
        #schedule.step()
        with torch.set_grad_enabled(True):
            net.train()
            gt, pd = None, None

            for step, (inp, snp, mask) in enumerate(trainloader, start=1):
                out = net(inp)
                loss = criterion(torch.sigmoid(out)[:, filter_indexes], snp[:, filter_indexes])
                epoch_loss += loss.item()
                filter_metrcis, not_filter_metrics, _, _, step_gt, step_pd = evaluationByMask(out, snp, mask, filter_indexes)
                gt =  step_gt if gt is None else np.hstack((gt, step_gt))
                pd = step_pd  if pd is None else np.hstack((pd, step_pd))
                print(Fore.WHITE + Back.BLACK, end='')
                print('Epoch:[{:4d}]  Step:[{:4d}]   Loss:[{:.5f}]   MeanLoss:[{:.5f}]'.format(epoch, step, loss, epoch_loss/step), end='')
                print('   FilterMean:[{:.3f}]   Filter:[(>=0.99){:.3f} (>=0.96){:.3f} (>=0.9){:.3f}]'.format(*filter_metrcis), end='')
                print('   NotFilterMean:[{:.3f}]   NotFilter:[(>=0.99){:.3f} (>=0.96){:.3f} (>=0.9){:.3f}]'.format(*not_filter_metrics))
                print(Fore.RESET + Back.RESET, end='')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        with torch.set_grad_enabled(False):
            net.eval()
            total_filter_metrics, total_not_filter_metrics = None, None
            gt, pd = None, None
            for step, (inp, snp, mask) in enumerate(valloader, start=1):
                out = net(inp)
                loss = criterion(torch.sigmoid(out), snp)
                filter_metrcis, not_filter_metrics, farray,nfarray , step_gt, step_pd = \
                                                    evaluationByMask(out, snp, mask, filter_indexes, verbose=True)
                total_filter_metrics = farray if total_filter_metrics is None else np.hstack((total_filter_metrics, farray))
                total_not_filter_metrics = nfarray if total_not_filter_metrics is None else \
                                            np.hstack((total_not_filter_metrics, nfarray))
                gt =  step_gt if gt is None else np.hstack((gt, step_gt))
                pd = step_pd  if pd is None else np.hstack((pd, step_pd))
                print(Fore.WHITE + Back.BLACK, end='')
                print('[Validation]Epoch:[{:2d}]     [Validation]Loss:[{:.5f}] ------'.format(epoch, loss),end='')
                print('   FilterMean:[{:.3f}]   Filter:[(>=0.99){:.3f} (>=0.96){:.3f} (>=0.9){:.3f}]'.format(*filter_metrcis), end='')
                print('   NotFilterMean:[{:.3f}]   NotFilter:[(>=0.99){:.3f} (>=0.96){:.3f} (>=0.9){:.3f}]'.format(*not_filter_metrics))
                print(Fore.RESET + Back.RESET)
            print(Fore.WHITE + Back.RED, end=' ')
            # print(confusion_matrix(gt, pd))
            # print(classification_report(gt, pd))
            print('[Final Filter] Mean: {:.3f}  [(>=0.99){:.3f} (>=0.96){:.3f} (>=0.9){:.3f}]'.format(np.mean(total_filter_metrics),
                                                                        np.sum(total_filter_metrics >=0.99)/len(total_filter_metrics),
                                                                        np.sum(total_filter_metrics >= 0.96)/len(total_filter_metrics),
                                                                        np.sum(total_filter_metrics >= 0.90) /len(total_filter_metrics)))
            print('[Final Filter] Mean: {:.3f}  [(>=0.99){:.3f} (>=0.96){:.3f} (>=0.9){:.3f}]'.format(np.mean(total_not_filter_metrics),
                                                                        np.sum(total_not_filter_metrics >=0.99)/len(total_not_filter_metrics),
                                                                        np.sum(total_not_filter_metrics >= 0.96)/len(total_not_filter_metrics),
                                                                        np.sum(total_not_filter_metrics >= 0.90) /len(total_not_filter_metrics)))

            print(Fore.RESET + Back.RESET)
            save_model(save_path, net, epoch, total_filter_metrics)
        print('='*180)
        print('\n')

if __name__ == '__main__':
    train(10000, 256)


