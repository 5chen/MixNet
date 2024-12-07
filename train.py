import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset, TrainDataset_M
from net.MixNet import MixNet
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.val_utils import AverageMeter, compute_psnr_ssim


class UHDModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.net = MixNet()
        self.loss_l1  = nn.L1Loss()
        self.opt = opt
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_l1(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("loss", loss)
        return loss
 
    def lr_scheduler_step(self,scheduler,*args, **kwargs):
        scheduler.step()
        lr = scheduler.get_last_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        warm_up = self.opt.epochs * 0.15
        max_e = self.opt.epochs * 0.75
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=warm_up,max_epochs=max_e)

        return [optimizer],[scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="MixNet")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")
    
    # TrainDataset_M for demoire
    trainset = TrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs=1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    model = UHDModel(opt)
          
    trainer = pl.Trainer(max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])                
    trainer.fit(model, trainloader)


if __name__ == '__main__':
    main()
