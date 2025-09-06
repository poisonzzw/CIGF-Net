import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch

class TESTData(Dataset):

    def __init__(self, file_path_a, file_path_b, file_path_label):
        super(TESTData, self).__init__()

        self.file_path_a = file_path_a
        self.image_path_a = os.listdir('./Data/test-data/ir')

        self.file_path_b = file_path_b
        self.image_path_b = os.listdir('./Data/test-data/vis')

        self.file_path_label = file_path_label
        self.image_path_label = os.listdir('./Data/test-data/label')



    def __getitem__(self, indxe):

        img_path_a = self.image_path_a[indxe]

        img_path_b = self.image_path_b[indxe]

        image_path_label = self.image_path_label[indxe]

        img_a = self.file_path_a + img_path_a
        img_b = self.file_path_b + img_path_b
        img_label = self.file_path_label + image_path_label

        image_IR = np.array(cv2.imread(img_a, cv2.IMREAD_GRAYSCALE), dtype=None)
        image_VI = np.array(cv2.imread(img_b, cv2.COLOR_BGR2RGB), dtype=None)
        image_LABEL = np.array(cv2.imread(img_label, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)

        image_IR = cv2.merge([image_IR, image_IR, image_IR])

        image_VI = self.normalize(image_VI)
        image_IR = self.normalize(image_IR)

        image_VI = image_VI.transpose(2, 0, 1)
        image_IR = image_IR.transpose(2, 0, 1)

        image_VI = torch.from_numpy(np.ascontiguousarray(image_VI)).float()
        image_LABEL = torch.from_numpy(np.ascontiguousarray(image_LABEL)).long()
        image_IR = torch.from_numpy(np.ascontiguousarray(image_IR)).float()

        return image_IR, image_VI, image_LABEL, image_path_label

    @staticmethod
    def normalize(img, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
        # pytorch pretrained model need the input range: 0-1
        img = img.astype(np.float64) / 255.0
        img = img - mean
        img = img / std
        return img

    def __len__(self):
        return len(self.image_path_a)

def testpath():
    image_ir_path = './Data/test-data/ir/'
    image_vis_path = './Data/test-data/vis/'
    image_label_path = './Data/test-data/label/'
    return image_ir_path, image_vis_path, image_label_path

