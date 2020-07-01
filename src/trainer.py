import os
import time
import random
from collections import Counter

import torch
import numpy as np


def cut_model_state_dict(state_dict):
    state_dict.pop("fc2.weight")
    state_dict.pop("fc2.bias")
    state_dict.pop("bn2.weight")
    state_dict.pop("bn2.bias")
    state_dict.pop("bn2.running_mean")
    state_dict.pop("bn2.running_var")
    state_dict.pop("bn2.num_batches_tracked")
    state_dict.pop("center_features.weight") if state_dict["center_features.weight"] is not None else None


class Trainer:
    """모델 학습 및 평가를 위한 클래스"""

    def __init__(self, model, args, logging=True, pretext=False):
        self.args = args
        self.model = model
        self.model.cuda()
        self.logging = logging
        self.pretext = pretext
        self.model_name = f"{self.model.__class__.__name__}_{str(time.time()).split('.')[0]}"
        self.best_accuracy = 0

    def train(self):
        """모델을 학습"""
        self.write_log(f"Start training {self.model_name}", mode="w")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 80, 120], gamma=0.333)  # changed

        train_data_loader = self.model.get_train_data_loader()
        val_data_loader = self.model.get_test_data_loader()

        self.write_log("")
        for key, val in self.args.__dict__.items():
            self.write_log(f"{key}: {val}")
        self.write_log("")

        for epoch in range(self.args.epochs):
            start_time = time.time()
            self.write_log(f"Epoch {epoch + 1:03d}/{self.args.epochs:03d} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.train_epoch(train_data_loader)
            self.write_log(f"Time elapsed: {(time.time() - start_time) / 60:.2f} mins.")

            if self.pretext:  # pretext 모델인 경우 evaluation 없이 모델 저장
                torch.save(self.model.state_dict(), f"results/train/weights/{self.model_name}.pt")
                self.write_log("Saved model!")
            elif (epoch + 1) % 10 == 0:  # N 에폭마다 evaluation 진행
                self.eval(val_data_loader)
    
    def train_epoch(self, data_loader):
        """모델을 1에폭 학습"""

        self.model.train()

        # 학습 추적을 위한 리스트
        cost_list = list()
        ind_cost_list = list()

        for batch_idx, data in enumerate(data_loader):
            cost, ind_costs = self.model.get_cost(data)

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            cost_list.append(cost.item())
            ind_cost_list.append(ind_costs)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(data_loader):
                self.write_log(f"Batch: {batch_idx + 1:04d}/{len(data_loader):04d} | Cost: {np.mean(cost_list):.4f} {np.round(np.mean(ind_cost_list, axis=0), 3)}")

                # 학습 추적을 위한 리스트 초기화
                cost_list = list()
                ind_cost_list = list()

        self.scheduler.step()
    
    def eval(self, data_loader, train=True, save=False):
        """모델을 evaluation 혹은 추론"""
        self.model.eval()

        img1_path_list = list()
        img2_path_list = list()
        labels_list = list()
        label_preds_list = list()

        for idx, data in enumerate(data_loader):
            print(f"Evaluation Batch: {idx + 1:04d}/{len(data_loader):04d}", end="\r")

            img1_path, img2_path, labels, label_preds = self.model.evaluate(data)
            img1_path_list.extend(img1_path)
            img2_path_list.extend(img2_path)
            labels_list.extend(labels)
            label_preds_list.extend(label_preds)
        print()
        
        threshold = np.median(label_preds_list)
        self.write_log(f"Threshold: {threshold:.4f}")

        label_preds_list = np.array(label_preds_list) > threshold
        accuracy = np.mean(np.array(labels_list).astype(bool) == np.array(label_preds_list).astype(bool))
        self.write_log(f"Accuracy: {accuracy:.6f}")

        # 학습 중일 경우 최고의 validation 성능을 낸 모델 저장
        if train:
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

                # Inference에 사용되지 않는 layer를 덜어내고 모델을 저장
                state_dict = self.model.state_dict()
                cut_model_state_dict(state_dict)
                torch.save(state_dict, f"results/train/weights/{self.model_name}.pt")
                self.write_log("Saved best model!")
        
        # 필요한 경우 prediction 결과를 저장
        if save:
            with open(f"results/test/{self.args.model_weight.split('/')[-1].split('.')[0]}_preds.csv", "w") as result:
                result.write(f"image1,image2,label\n")
                for img1_path, img2_path, pred in zip(img1_path_list, img2_path_list, label_preds_list):
                    result.write(f"{img1_path},{img2_path},{int(pred)}\n")

    def write_log(self, msg, mode="a"):
        """로그 파일 작성"""
        if self.logging:
            with open(f"results/train/logs/{self.model_name}.log", mode) as log:
                log.write(f"{msg}\n")
        print(msg)
