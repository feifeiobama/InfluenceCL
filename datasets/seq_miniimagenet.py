from argparse import Namespace
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path_dataset as base_path
from PIL import Image
import os
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize


class MiniImagenet(Dataset):
    """
    Defines Mini Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool = True, transform: transforms = None,
                 target_transform: transforms = None, download: bool = False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from onedrivedownloader import download
                ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/EYLmey_IMdVPtGCrCBx_CCMBToexGLjdFVy5mz5mo3Wpcg?download=1"
                print('Downloading dataset')
                download(ln, filename=os.path.join(root, 'miniImagenet.zip'), unzip=True, unzip_path=root, clean=True)

        self.data = np.load(os.path.join(
            root, '%s_x.npy' %
                  ('train' if self.train else 'test')))
        self.targets = np.load(os.path.join(
            root, '%s_y.npy' %
                  ('train' if self.train else 'test')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyMiniImagenet(MiniImagenet):
    """
    Defines Mini Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool = True, transform: transforms = None,
                 target_transform: transforms = None, download: bool = False) -> None:
        super(MyMiniImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialMiniImagenet(ContinualDataset):

    NAME = 'seq-miniimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 5
    MEAN = (0.47313006, 0.44905752, 0.40378186)
    STD = (0.27292014, 0.26559181, 0.27953038)
    TRANSFORM = transforms.Compose(
        [transforms.Resize(32),
         # remove data augmentation to reproduce results
         # transforms.RandomCrop(32, padding=4),
         # transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN,
                              STD)])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyMiniImagenet(base_path() + 'MINIIMG',
                                       train=True, download=True, transform=transform)
        train_dataset.not_aug_transform = test_transform  # store normalized images in the buffer
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = MiniImagenet(base_path() + 'MINIIMG',
                                        train=False, download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone():
        return resnet18(SequentialMiniImagenet.N_CLASSES_PER_TASK
                        * SequentialMiniImagenet.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(  # weaken random crop to reproduce results
            [transforms.RandomCrop(32, padding=1), transforms.RandomHorizontalFlip()])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialMiniImagenet.MEAN,
                                         SequentialMiniImagenet.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialMiniImagenet.MEAN,
                                SequentialMiniImagenet.STD)
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialMiniImagenet.get_batch_size()
