import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from utils.schedulers import LinearWarmupCosineAnnealingLR
import torch.optim as optim
import os
import torch.nn as nn 

from utils.dataset_utils import TestDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.MixNet import MixNet
from utils.loss_utils import *

import lightning.pytorch as pl
import torch.nn.functional as F

class UHDModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = MixNet()
        self.loss_l1  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_l1(restored,clean_patch)
        self.log("train_loss", loss)
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--valid_data_dir', type=str, default="data/Test/UHDLOL4K/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="model/MixNet.ckpt", help='checkpoint save path')

    testopt = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = testopt.ckpt_path
    print("CKPT name : {}".format(ckpt_path))

    net = UHDModel().load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    data_path = testopt.valid_data_dir
    dataset_name = data_path.split('/')[-2]
    print(f'Test: {dataset_name}')
    data_set = TestDataset(testopt)
    
    output_path = testopt.output_path + dataset_name + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    testloader = DataLoader(data_set, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for (degraded_name, degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
    
