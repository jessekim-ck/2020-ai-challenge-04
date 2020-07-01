"""
Pretext model

- Re-ID 모델의 효과적인 학습을 위해, training set을 이용하여 facial semantic information을 학습하도록 한 모델.
- Facial expression(3 expressions), camera angle(9 angles)를 Cross Entropy loss를 통해 학습한다.
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd
from PIL import Image

from src.efficientnet import EfficientNet


class TrainDataset(Dataset):
    """학습을 위한 커스텀 데이터셋"""

    def __init__(self, img_dir="train", csv_path="train_meta.csv", transform=None):
        self.img_dir = os.path.join("/datasets/objstrgzip/03_face_verification_angle", img_dir)
        meta_data = pd.read_csv(os.path.join(self.img_dir, csv_path))
        self.paths = meta_data["file_name"].values
        self.angles = meta_data["file_name"].map(lambda x: x.split("_")[-2])
        self.expressions = meta_data["file_name"].map(lambda x: x.split("_")[-3])

        # reproducibility를 위한 하드코딩
        self.angle_to_idx = {
            "C1": 0,
            "C2": 1,
            "C3": 2,
            "C6": 3,
            "C7": 4,
            "C8": 5,
            "C11": 6,
            "C12": 7,
            "C13": 8
        }  # for reproducibility
        # for idx, angle in enumerate(set(self.angles)):
        #     self.angle_to_idx[angle] = idx

        # reproducibility를 위한 하드코딩
        self.expression_to_idx = {
            "E01": 0,
            "E02": 1,
            "E03": 2
        }
        # for idx, expression in enumerate(set(self.expressions)):
        #     self.expression_to_idx[expression] = idx

        self.transform = transform
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self.transform(Image.open(os.path.join(self.img_dir, path)))
        angle_idx = self.angle_to_idx[self.angles[idx]]
        expression_idx = self.expression_to_idx[self.expressions[idx]]
        return path, img, angle_idx, expression_idx

    def __len__(self):
        return len(self.paths)


class Pretext(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = EfficientNet.from_name("efficientnet-b0")
        out_channels = self.backbone.out_channels
        self.fc = nn.Linear(out_channels, 12, bias=True)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x1 = x[:, :9]  # camera angle logit
        x2 = x[:, 9:]  # facial expression logit
        return x1, x2

    def get_cost(self, data):
        _, x, angle, expression = data

        x1, x2 = self(x.cuda())
        del x
        
        cost1 = nn.CrossEntropyLoss()(x1, angle.cuda())
        del x1
        
        cost2 = nn.CrossEntropyLoss()(x2, expression.cuda())
        del x2
        
        return cost1 + cost2, (cost1.item(), cost2.item())

    def evaluate(self, data):
        raise NotImplementedError

    def get_train_data_loader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=1)
        ])

        data_loader = DataLoader(
            TrainDataset("train", "train_meta.csv", transform=transform),
            batch_size=self.args.batch_size,
            num_workers=0,
            shuffle=True,
            drop_last=True,
            worker_init_fn=lambda wid: np.random.seed(0 + wid)
        )

        return data_loader

    def get_test_data_loader(self):
        return None
