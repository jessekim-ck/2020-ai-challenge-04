import os
import random
import argparse

import torch
import numpy as np
import pandas as pd
from PIL import Image

from src import Trainer
from src.models import Pretext
from utils import random_seed


def train_pretext():
    # Reproducibility를 위한 모든 random seed 고정
    random_seed()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--epochs", type=int, default=7)
    arg_parser.add_argument("--lr", type=float, default=0.001)
    arg_parser.add_argument("--batch_size", type=int, default=128)
    args = arg_parser.parse_args()
    
    # 모델 로드
    model = Pretext(args)

    # Trainer 인스턴스 생성 및 학습
    trainer = Trainer(model, args, pretext=True)
    trainer.train()

    return trainer.model_name


if __name__ == "__main__":
    train_pretext()
