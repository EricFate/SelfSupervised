import torch
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import os
# from config import opt
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
from .utils import rotate_img, Denormalize
from skimage.color import rgb2lab, rgb2gray
from skimage import io
from itertools import permutations
import random

TINY_IMAGENET_DIR = '/home/amax/data/tiny-imagenet-200'
CIFAR_DATASET_DIR = '/home/amax/data'


class RandomTranslateWithReflect:
    '''
    Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image


class TransformsC10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])
        # transform for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False, num_imgs_per_cat=None, resize=None):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + "_" + self.split
        self.random_sized_crop = random_sized_crop

        # The num_imgs_per_cats input argument specifies the number
        # of training examples per category that would be used.
        # This input argument was introduced in order to be able
        # to use less annotated examples than what are available
        # in a semi-superivsed experiment. By default all the
        # available training examplers per category are being used.
        self.num_imgs_per_cat = num_imgs_per_cat

        if self.dataset_name == "tinyimagenet":
            assert self.split == "train" or self.split == "val"
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]
            transforms_list = []
            # if self.split != "train":
            #     transforms_list = [
            #         transforms.Scale(256),
            #         transforms.CenterCrop(224),
            #         lambda x: np.asarray(x),
            #     ]
            # else:
            #     if self.random_sized_crop:
            #         transforms_list = [
            #             transforms.RandomSizedCrop(64),
            #             transforms.RandomHorizontalFlip(),
            #             lambda x: np.asarray(x),
            #         ]
            #     else:
            #         transforms_list = [
            #             transforms.Scale(256),
            #             transforms.RandomCrop(224),
            #             transforms.RandomHorizontalFlip(),
            #             lambda x: np.asarray(x),
            #         ]
            if resize is not None:
                transforms_list.append(transforms.Resize(resize))
            self.transform = transforms.Compose(transforms_list)
            split_data_dir = TINY_IMAGENET_DIR + "/" + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform)
        elif self.dataset_name == 'cifar10':
            self.mean_pix = [x / 255.0 for x in [125.3, 123.0, 113.9]]
            self.std_pix = [x / 255.0 for x in [63.0, 62.1, 66.7]]

            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the CIFAR dataset')

            transform = []
            # if (split != 'test'):
            #     transform.append(transforms.RandomCrop(32, padding=4))
            #     transform.append(transforms.RandomHorizontalFlip())
            # transform.append(lambda x: np.asarray(x))
            if resize is not None:
                transform.append(transforms.Resize(resize))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                CIFAR_DATASET_DIR, train=self.split == 'train',
                download=True, transform=self.transform)
        else:
            raise ValueError("Not recognized dataset {0}".format(dataset_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)


class BaseDataLoader:
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True,
                 resize=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix = self.dataset.mean_pix
        std_pix = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1, 2, 0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        # if self.unsupervised:
        # if in unsupervised mode define a loader function that given the
        # index of an image it returns the 4 rotated copies of the image
        # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
        # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.

        # else:  # supervised mode
        # if in supervised mode define a loader function that given the
        # index of an image it returns the image and its categorical label
        # def _load_function(idx):
        #     idx = idx % len(self.dataset)
        #     img, categorical_label = self.dataset[idx]
        #     img = self.transform(img)
        #     return img, categorical_label
        #
        # _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=self._load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=self._collate_fun, num_workers=self.num_workers,
                                           shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size // self.batch_size

    def _load_function(self, idx):
        raise NotImplementedError

    def _collate_fun(self, batch):
        raise NotImplementedError


class OriginDataLoader(BaseDataLoader):

    def _load_function(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label

    def _collate_fun(self, batch):
        return default_collate(batch)


class RotateDataLoader(BaseDataLoader):

    def _load_function(self, idx):
        idx = idx % len(self.dataset)
        img0, _ = self.dataset[idx]
        rotated_imgs = [
            self.transform(img0),
            self.transform(rotate_img(img0, 90)),
            self.transform(rotate_img(img0, 180)),
            self.transform(rotate_img(img0, 270))
        ]
        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        return torch.stack(rotated_imgs, dim=0), rotation_labels

    def _collate_fun(self, batch):
        batch = default_collate(batch)
        assert (len(batch) == 2)
        batch_size, rotations, channels, height, width = batch[0].size()
        batch[0] = batch[0].view([batch_size * rotations, channels, height, width])
        batch[1] = batch[1].view([batch_size * rotations])
        return batch


class JigsawDataLoader(BaseDataLoader):
    def __init__(self, dataset, perm, **kwargs):
        super().__init__(dataset, **kwargs)
        self.perm = perm
        self.perms = list(permutations(range(self.perm * self.perm)))

    def _load_function(self, idx):
        idx = idx % len(self.dataset)
        img0, _ = self.dataset[idx]
        img = self.transform(img0)
        C, W, H = img.size()
        s = W // self.perm

        def crop_image(block):
            i = block // self.perm
            j = block % self.perm
            return img[:, i * s:(i + 1) * s, j * s:(j + 1) * s]

        order = np.random.randint(len(self.perms))
        jigsaw = [crop_image(i) for i in self.perms[order]]
        return torch.stack(jigsaw), order

    def _collate_fun(self, batch):
        return default_collate(batch)


class AMDIMDataLoader(BaseDataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.multi_view_transform = TransformsC10()

    def _load_function(self, idx):
        idx = idx % len(self.dataset)
        img, _ = self.dataset[idx]
        x1, x2 = self.multi_view_transform(img)
        return x1, x2

    def _collate_fun(self, batch):
        return default_collate(batch)


class ExamplarDataLoader(BaseDataLoader):
    def _load_function(self, idx):
        idx = idx % len(self.dataset)
        img, _ = self.dataset[idx]
        img = self.transform(img)
        return img, idx

    def _collate_fun(self, batch):
        return default_collate(batch)


class ColorDataLoader(BaseDataLoader):

    def __init__(self, dataset, train=True, **kwargs):
        super().__init__(dataset, **kwargs)
        self.train = train

    def _load_function(self, idx):
        idx = idx % len(self.dataset)
        img, label = self.dataset[idx]
        img_original = self.transform(img)
        img_original = np.asarray(img_original).transpose((1, 2, 0))
        if self.train:
            img_lab = rgb2lab(img_original)
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
            img_original = rgb2lab(img_original)[:, :, 0] - 50.
            img_original = torch.from_numpy(img_original)
            return img_original.unsqueeze(0), img_ab
        else:
            img_original = rgb2lab(img_original)[:, :, 0] - 50.
            img_original = torch.from_numpy(img_original)
            return img_original.unsqueeze(0), label

    def _collate_fun(self, batch):
        return default_collate(batch)
