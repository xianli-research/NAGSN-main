import random

import numpy as np
import pytorch_lightning as pl
import torch as th


def seed_everything_full(seed: int):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
