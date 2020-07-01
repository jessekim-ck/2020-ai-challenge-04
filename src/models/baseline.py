"""
Baseline model

- Hard Triplet loss, Identity Softmax loss를 통해 모델을 학습한다.
- 최고 성능: validation accuracy 0.987, test accuracy 0.979
"""

import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms

import numpy as np
import pandas as pd
from PIL import Image

from src.efficientnet import EfficientNet
from src.models.utils.losses import TripletHardLoss, CrossEntropyLabelSmoothLoss, euclidean_dist


class TrainDataset(Dataset):
    """학습을 위한 커스텀 데이터셋"""

    def __init__(self, img_dir="train", csv_path="train_meta.csv", transform=None):
        self.img_dir = os.path.join("/datasets/objstrgzip/03_face_verification_angle", img_dir)
        meta_data = pd.read_csv(os.path.join(self.img_dir, csv_path))
        self.paths = meta_data["file_name"].values
        self.pids = meta_data["face_id"].values
        self.pid_to_idx = {pid: idx for idx, pid in enumerate(set(self.pids))}
        self.transform = transform
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self.transform(Image.open(os.path.join(self.img_dir, path)))
        pid_idx = self.pid_to_idx[self.pids[idx]]
        return path, img, pid_idx

    def __len__(self):
        return len(self.pids)


class TrainSampler(Sampler):
    """(Soft batch) Hard Triplet loss를 위한 train loader의 index를 생성하는 sampler"""

    def __init__(self, img_dir="train", csv_path="train_meta.csv", num_instances=16):
        self.img_dir = os.path.join("/datasets/objstrgzip/03_face_verification_angle", img_dir)
        meta_data = pd.read_csv(os.path.join(self.img_dir, csv_path))
        
        # person id별로 image의 index를 모음
        self.pid_to_img = defaultdict(list)
        for idx, row in meta_data.iterrows():
            self.pid_to_img[row["face_id"]].append(idx)
        self.pid_list = list(self.pid_to_img.keys())

        self.num_classes = len(self.pid_list)
        self.num_instances = num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_classes)  # person id의 순서를 무작위로 섞음
        ret = list()

        # 사람(identity)마다 `num_instances` 개의 이미지를 무작위로 뽑아 index에 포함시킴
        for idx in indices:
            pid = self.pid_list[idx]
            t = np.random.choice(self.pid_to_img[pid], size=self.num_instances, replace=False)
            ret.extend(t)

        return iter(ret)

    def __len__(self):
        return self.num_classes * self.num_instances


class TestDataset(Dataset):
    """Test 및 evaluation을 위한 커스텀 데이터셋"""

    def __init__(self, img_dir="validate", csv_path="validate_label.csv", transform=None):
        self.img_dir = os.path.join("/datasets/objstrgzip/03_face_verification_angle", img_dir)
        label_data = pd.read_csv(os.path.join(self.img_dir, csv_path))
        self.image1_paths = label_data["image1"].values
        self.image2_paths = label_data["image2"].values
        self.labels = label_data["label"].values
        self.transform = transform

    def __getitem__(self, idx):
        image1_path = self.image1_paths[idx]
        image2_path = self.image2_paths[idx]
        image1 = self.transform(Image.open(os.path.join(self.img_dir, image1_path)))
        image2 = self.transform(Image.open(os.path.join(self.img_dir, image2_path)))
        label = self.labels[idx]
        return image1_path, image2_path, image1, image2, label

    def __len__(self):
        return len(self.image1_paths)


class BaseLine(nn.Module):

    def __init__(self, args, num_classes, train=False):
        super().__init__()
        self.args = args
        self.num_classes = num_classes

        self.backbone = EfficientNet.from_name("efficientnet-b0")  # pretrained weight 없이 모델 로드
        out_channels = self.backbone.out_channels
        
        # feature embedding을 위한 layer
        self.fc1 = nn.Linear(out_channels, 512, bias=True)
        self.bn1 = nn.BatchNorm1d(512)
        
        # identity logit을 위한 layer
        self.fc2 = nn.Linear(512, self.num_classes, bias=True) if train else None
        self.bn2 = nn.BatchNorm1d(self.num_classes) if train else None
    
    def forward(self, x, train):
        x = self.backbone(x)
        feature = self.bn1(self.fc1(x))  # feature embedding
        del x
        if not train:
            return feature
        identity = self.bn2(self.fc2(feature))  # identity logit

        # feature = self.fc1(x)
        # del x
        # if not train:
        #     return feature
        # identity = self.bn2(self.fc2(nn.ReLU()(feature)))
        return feature, identity

    def get_cost(self, data):
        """학습을 위한 batch cost를 계산하여 출력"""
        _, x, pid_indices = data
        
        features, identities = self.forward(x.cuda(), train=True)
        del x
        
        identity_cost = CrossEntropyLabelSmoothLoss(num_classes=self.num_classes)(identities, pid_indices.cuda())
        del identities
        
        triplet_cost = TripletHardLoss(margin=0.3)(features, pid_indices.cuda())
        del features
        
        cost = identity_cost + 0.1*triplet_cost
        return cost, [identity_cost.item(), triplet_cost.item()]

    def evaluate(self, data):
        """evaludation을 위해 prediction 결과와 GT label을 출력"""
        with torch.no_grad():
            img1_path, img2_path, img1, img2, labels = data
            features1 = self.forward(img1.cuda(), train=False)
            features2 = self.forward(img2.cuda(), train=False)
            dist_mat = torch.diag(euclidean_dist(features1, features2), 0)
            return img1_path, img2_path, labels, list(dist_mat.cpu().numpy())
 
    def get_train_data_loader(self):
        """학습을 위한 data loader"""

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(),
            transforms.ToTensor(),
        ])

        data_loader = DataLoader(
            TrainDataset("train", "train_meta.csv", transform=transform),
            batch_size=self.args.batch_size*self.args.num_instances,
            num_workers=0,
            sampler=TrainSampler("train", "train_meta.csv", self.args.num_instances),
            drop_last=True,
            shuffle=False
        )

        return data_loader

    def get_test_data_loader(self, img_dir="validate", csv_path="validate_label.csv"):
        """테스트 및 evaluation을 위한 data loader"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])

        data_loader = DataLoader(
            TestDataset(img_dir, csv_path, transform),
            batch_size=2*self.args.batch_size*self.args.num_instances,
            num_workers=0,
            shuffle=False,
        )

        return data_loader
