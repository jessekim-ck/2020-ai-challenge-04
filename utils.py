import os
import random

import torch
import numpy as np


def count_parameters(model):
    """모델의 parameter 수를 계산"""
    return sum(p.numel() for p in model.parameters())


def random_seed(seed=0):
    """Reproducibility 확보"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
