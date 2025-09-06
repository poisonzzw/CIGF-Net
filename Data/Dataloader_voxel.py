import os
import os.path
import random
import torch
import torch.utils.data as data
import cv2
from torchvision.transforms import Resize, InterpolationMode, ToTensor
import torchvision.transforms.functional as F
import numpy as np
from Utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

class DualBrightnessTransform:

    def __init__(self):
        self.factor = random.uniform(0.5, 1.5)

    def __call__(self, img_a, img_b):
        return F.adjust_brightness(img_a, self.factor), F.adjust_brightness(img_b, self.factor)


class DualContrastTransform:

    def __init__(self):
        self.factor = random.uniform(0.5, 1.5)

    def __call__(self, img_a, img_b):
        return F.adjust_contrast(img_a, self.factor), F.adjust_contrast(img_b, self.factor)


class DualGammaTransform:

    def __init__(self):
        self.factor = random.uniform(0.8, 1.2)

    def __call__(self, img_a, img_b):
        return F.adjust_gamma(img_a, self.factor), F.adjust_gamma(img_b, self.factor)


class DualSharpnessTransform:

    def __init__(self):
        self.factor = random.uniform(0.5, 1.5)

    def __call__(self, img_a, img_b):
        return F.adjust_sharpness(img_a, self.factor), F.adjust_sharpness(img_b, self.factor)


class DualHFlipTransform:

    def __init__(self):
        self.factor = random.random()

    def __call__(self, img_a, img_b, img_c):
        if self.factor > 0.5:
            return img_a, img_b, img_c
        else:
            return F.hflip(img_a), F.hflip(img_b), F.hflip(img_c)


class DualVFlipTransform:

    def __init__(self):
        self.factor = random.random()

    def __call__(self, img_a, img_b, img_c):
        if self.factor > 0.5:
            return img_a, img_b, img_c
        else:
            return F.vflip(img_a), F.vflip(img_b), F.vflip(img_c)

class MotherData(data.Dataset):
    def __init__(self, directory, FLAGS):
        super(MotherData, self).__init__()
        self.FLAGS = FLAGS
        self.directory = directory
        self.data_list = []
        self.shape = self.FLAGS['Data']['shape']

        if self.FLAGS['Data']['data_type'] == 'train':
            for sequences in self.FLAGS['Data']['split_sequences'][self.FLAGS['Data']['data_type']]:

                file_dir = self.directory + '/' + sequences + '/'

                for f in os.listdir(file_dir):
                    # print(f.split('.')[0][-3:])

                    if f.split('.')[0][-3:] == self.FLAGS['Data']['data_end']['ir']:
                        self.data_list.append(
                            file_dir + f.split('_')[0])
        else:
            file_dir = self.directory + '/' + self.FLAGS['Data']['data_type'] + '/'

            for f in os.listdir(file_dir):
                if f.split('.')[0][-3:] == self.FLAGS['Data']['data_end']['ir']:
                    self.data_list.append(
                        file_dir + f.split('_')[0])


    def __getitem__(self, index):

        if self.FLAGS['Data']['data_type'] == 'train':

            filepath_IR = (self.data_list[index] + self.FLAGS['Data']['data_end']['ir']) + '.png'
            filepath_VI = (self.data_list[index] + self.FLAGS['Data']['data_end']['vis']) + '.png'
            filepath_LABE = self.data_list[index] + '.png'

            image_IR = np.array(cv2.imread(filepath_IR, cv2.IMREAD_GRAYSCALE), dtype=None)
            image_VI = np.array(cv2.imread(filepath_VI, cv2.COLOR_BGR2RGB), dtype=None)
            image_LABEL = np.array(cv2.imread(filepath_LABE, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)

            image_IR = cv2.merge([image_IR, image_IR, image_IR])

            image_VI, image_LABEL, image_IR = self.random_mirror(image_VI, image_LABEL, image_IR)
            train_scale = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
            image_VI, image_LABEL, image_IR, scale = self.random_scale(image_VI, image_LABEL, image_IR, train_scale)

            image_VI = self.normalize(image_VI)
            image_IR = self.normalize(image_IR)

            crop_size = (480, 640)
            crop_pos = generate_random_crop_pos(image_VI.shape[:2], crop_size)

            image_VI, _ = random_crop_pad_to_shape(image_VI, crop_pos, crop_size, 0)
            image_LABEL, _ = random_crop_pad_to_shape(image_LABEL, crop_pos, crop_size, 255)
            image_IR, _ = random_crop_pad_to_shape(image_IR, crop_pos, crop_size, 0)

            image_VI = image_VI.transpose(2, 0, 1)
            image_IR = image_IR.transpose(2, 0, 1)

            image_VI = torch.from_numpy(np.ascontiguousarray(image_VI)).float()
            image_LABEL = torch.from_numpy(np.ascontiguousarray(image_LABEL)).long()
            image_IR = torch.from_numpy(np.ascontiguousarray(image_IR)).float()

            TRAIN_DATA = {
                'image_IR': image_IR,
                'image_VI': image_VI,
                'image_LABEL': image_LABEL
            }

            return TRAIN_DATA

        elif self.FLAGS['Data']['data_type'] == 'valid':

            filepath_IR = (self.data_list[index] + self.FLAGS['Data']['data_end']['ir']) + '.png'
            filepath_VI = (self.data_list[index] + self.FLAGS['Data']['data_end']['vis']) + '.png'
            filepath_LABE = self.data_list[index] + '.png'

            image_IR = np.array(cv2.imread(filepath_IR, cv2.IMREAD_GRAYSCALE), dtype=None)
            image_VI = np.array(cv2.imread(filepath_VI, cv2.COLOR_BGR2RGB), dtype=None)
            image_LABEL = np.array(cv2.imread(filepath_LABE, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)

            image_IR = cv2.merge([image_IR, image_IR, image_IR])

            image_VI = self.normalize(image_VI)
            image_IR = self.normalize(image_IR)

            image_VI = image_VI.transpose(2, 0, 1)
            image_IR = image_IR.transpose(2, 0, 1)

            image_VI = torch.from_numpy(np.ascontiguousarray(image_VI)).float()
            image_LABEL = torch.from_numpy(np.ascontiguousarray(image_LABEL)).long()
            image_IR = torch.from_numpy(np.ascontiguousarray(image_IR)).float()

            VALID_DATA = {
                    'image_IR': image_IR,
                    'image_VI': image_VI,
                    'image_LABEL': image_LABEL,
                    'image_path': self.data_list[index].split('/')[-1]
                }

            return VALID_DATA

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def normalize(img, mean= np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
        # pytorch pretrained model need the input range: 0-1
        img = img.astype(np.float64) / 255.0
        img = img - mean
        img = img / std
        return img

    @staticmethod
    def random_mirror(rgb, gt, modal_x):
        if random.random() >= 0.5:
            rgb = cv2.flip(rgb, 1)
            gt = cv2.flip(gt, 1)
            modal_x = cv2.flip(modal_x, 1)

        return rgb, gt, modal_x

    @staticmethod
    def random_scale(rgb, gt, modal_x, scales):
        scale = random.choice(scales)
        sh = int(rgb.shape[0] * scale)
        sw = int(rgb.shape[1] * scale)
        rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
        modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

        return rgb, gt, modal_x, scale

class TestData(data.Dataset):

    def __init__(self, file_path_a, file_path_b):
        super(TestData, self).__init__()

        self.file_path_a = file_path_a
        self.image_path_a = os.listdir(self.file_path_a)
        self.image_path_a.sort()

        self.file_path_b = file_path_b
        self.image_path_b = os.listdir(self.file_path_b)
        self.image_path_b.sort()

    def __getitem__(self, indxe):
        img_path_a = self.image_path_a[indxe]
        img_a = os.path.join(self.file_path_a, img_path_a)

        img_path_b = self.image_path_b[indxe]
        img_b = os.path.join(self.file_path_b, img_path_b)

        img_a = ToTensor()(img_a)
        img_b = ToTensor()(img_b)
        return img_a, img_b

    def __len__(self):
        return len(self.image_path_a)
