import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ISICDataset(Dataset):
    def __init__(self, args, data_path, transform=None, mode='Training', plane=False):
        csv_name = f'ISBI2016_ISIC_Part1_{mode}_GroundTruth.csv'
        df = pd.read_csv(os.path.join(data_path, csv_name), encoding='gbk')
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode

        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 正确写法，使用 tuple
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
        ])

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')  # 单通道灰度图

        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # 将 mask 二值化并转为整数标签，适用于 CrossEntropyLoss
        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 127).astype(np.uint8)
        mask = torch.from_numpy(mask).long()

        return img, mask, name
