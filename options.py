import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100,help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=4,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

parser.add_argument('--train_data_dir', type=str, default='data/Train/UHDLOL4K/',  help='where training images saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument("--wblogger",type=str,default="MixNet",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="model/",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default=8,help = "Number of GPUs to use for training")

options = parser.parse_args()

