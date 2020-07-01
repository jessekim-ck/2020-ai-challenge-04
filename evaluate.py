import argparse

import torch
import pandas as pd

from src import Trainer
from src.models import BaseLine, Triarchy
from utils import count_parameters


def evaluate(model_weight=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch_size", type=int, default=128)
    arg_parser.add_argument("--num_instances", type=int, default=1)
    arg_parser.add_argument("--model_weight", type=str, default=f"results/train/weights/{model_weight}.pt")
    args = arg_parser.parse_args()

    train_meta = pd.read_csv("/datasets/objstrgzip/03_face_verification_angle/train/train_meta.csv")
    num_classes = len(set(train_meta["face_id"].values))

    # 모델 및 weight 로드
    model = Triarchy(args, num_classes, train=False)
    model.load_state_dict(torch.load(args.model_weight))
    print(f"Loaded weight {args.model_weight}")
    print(f"Number of Parameters: {count_parameters(model)}")

    # Trainer 인스턴스 생성 및 data loader 로드
    trainer = Trainer(model, args, logging=False)
    data_loader = trainer.model.get_test_data_loader("test", "test_label.csv")

    # Evaluation
    trainer.eval(data_loader, train=False, save=True)


if __name__ == "__main__":
    evaluate()
