import os
import random
import argparse

import torch
import numpy as np
import pandas as pd
from PIL import Image

from src import Trainer
from src.models import BaseLine, Triarchy
from utils import random_seed


def train(pretext_model="Pretext_1593000476"):
    # Reproducibility를 위한 모든 random seed 고정
    random_seed()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--epochs", type=int, default=110)
    arg_parser.add_argument("--lr", type=float, default=0.001)
    arg_parser.add_argument("--batch_size", type=int, default=8)
    arg_parser.add_argument("--num_instances", type=int, default=16)
    arg_parser.add_argument("--pretrained_weight", type=str, default=f"results/train/weights/{pretext_model}.pt")
    args = arg_parser.parse_args()

    train_meta = pd.read_csv("/datasets/objstrgzip/03_face_verification_angle/train/train_meta.csv")
    num_classes = len(set(train_meta["face_id"].values))

    # 모델 로드
    model = Triarchy(args, num_classes, train=True)

    # Train dataset의 표정 및 camera angle을 학습하도록 한 pretext model 가중치 로드
    pretrained_weight = torch.load(args.pretrained_weight)
    pretrained_weight.pop("fc.weight")
    pretrained_weight.pop("fc.bias")
    model.load_state_dict(pretrained_weight, strict=False)

    # Trainer 인스턴스 생성 및 학습
    trainer = Trainer(model, args)
    trainer.train()

    return trainer.model_name


if __name__ == "__main__":
    train()
