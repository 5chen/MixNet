import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils.image_utils import random_augmentation, crop_img

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  

 
class TrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_ids = []
        self.toTensor = ToTensor()

        self._init_ids()

    def _init_ids(self):
        data = self.args.train_data_dir + "input/"
        file_names = os.listdir(data)
        self.data_ids+= [data + id for id in file_names]
        random.shuffle(self.data_ids)
        num_data = len(self.data_ids)
        print("Total number of training data: {}".format(num_data))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)
        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_data_gt(self, data_name):
        gt_name = data_name.split("input")[0] + 'gt/' + data_name.split('/')[-1]
        return gt_name
    
    def __getitem__(self, index):
        sample = self.data_ids[index]
        degrad_img = crop_img(np.array(Image.open(sample).convert('RGB')), base=16)
        clean_name = self._get_data_gt(sample)
        clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
        degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))
        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        return degrad_patch, clean_patch
    
    def __len__(self):
        return len(self.data_ids)


class TestDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_ids = []
        self.toTensor = ToTensor()
        self._init_ids()

    def _init_ids(self):
        data = self.args.valid_data_dir + "input/"
        file_names = os.listdir(data)
        self.data_ids+= [data + id for id in file_names]

    def _get_data_gt(self, data_name):
        gt_name = data_name.split("input")[0] + 'gt/' + data_name.split('/')[-1]
        return gt_name
    
    def __getitem__(self, index):
        sample = self.data_ids[index]
        degrad_img = crop_img(np.array(Image.open(sample).convert('RGB')), base=16)
        clean_name = self._get_data_gt(sample)
        clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
        degrad_name = sample.split('/')[-1][:-4]
        clean_img = self.toTensor(clean_img)
        degrad_img = self.toTensor(degrad_img)
        return degrad_name, degrad_img, clean_img
    
    def __len__(self):
        return len(self.data_ids)
        
        
class TrainDataset_M(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.moire_paths = []
        self.gt_paths = []
        self.toTensor = ToTensor()

        self._init_ids()

    def _init_ids(self):
        data = self.args.train_data_dir + 'train/'
        for root, dirs, files in os.walk(self.args.train_data_dir):
            for fil in files:
                if fil.endswith("_moire.jpg"):
                    self.moire_paths.append(os.path.join(root, fil))
                elif fil.endswith("_gt.jpg"):
                    self.gt_paths.append(os.path.join(root, fil))
                    
        random.shuffle(self.moire_paths)
        num_data = len(self.moire_paths)
        print("Total number of training data: {}".format(num_data))
        

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)
        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_data_gt(self, data_name):
        gt_name = data_name.replace("_moire.jpg", "_gt.jpg")
        return gt_name
    
    def __getitem__(self, index):
        sample = self.moire_paths[index]
        degrad_img = crop_img(np.array(Image.open(sample).convert('RGB')), base=16)
        clean_name = self._get_data_gt(sample)
        clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
        degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))
        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        return degrad_patch, clean_patch
    
    def __len__(self):
        return len(self.moire_paths)
        
class TestDataset_M(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.moire_paths = []
        self.gt_paths = []
        self.toTensor = ToTensor()

        self._init_ids()

    def _init_ids(self):
        data = self.args.valid_data_dir + 'test_split/test/'
        
        for root, dirs, files in os.walk(self.args.valid_data_dir):
            for fil in files:
                if fil.endswith("_moire.jpg"):
                    self.moire_paths.append(os.path.join(root, fil))
                elif fil.endswith("_gt.jpg"):
                    self.gt_paths.append(os.path.join(root, fil))
                    
    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)
        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_data_gt(self, data_name):
        gt_name = data_name.replace("_moire.jpg", "_gt.jpg")
        return gt_name
    
    def __getitem__(self, index):
        sample = self.moire_paths[index]
        degrad_img = crop_img(np.array(Image.open(sample).convert('RGB')), base=16)
        clean_name = self._get_data_gt(sample)
        clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
        degrad_name = sample.split('/')[-1][:-4]
        # degrad_img, clean_img = random_augmentation(*self._crop_patch(degrad_img, clean_img))
        clean_img = self.toTensor(clean_img)
        degrad_img = self.toTensor(degrad_img)
        return degrad_name, degrad_img, clean_img
    
    def __len__(self):
        return len(self.moire_paths)
