import sys
from pathlib import Path

# isort: skip
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary

from pl_modules.models.pl_match import LitMatchStars
from core.checkpoints import require_checkpoint_path

from pl_modules.data import StarInfoDataModule


@hydra.main(config_path='../config', config_name='match', version_base='1.2')
def eval(config: DictConfig):
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    ckpt_path = require_checkpoint_path(config.test.ckpt_path)

    # ---------------------
    # Device
    # ---------------------
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    if not (
        isinstance(gpus, list) and all(isinstance(g, int) for g in gpus)
        or isinstance(gpus, int)
    ):
        raise ValueError(f"gpus must be int or List[int], got: {type(gpus)}, content: {gpus}")
    gpus = gpus if isinstance(gpus, list) else [gpus]

    # ---------------------
    # Data
    # ---------------------
    data_module = StarInfoDataModule(config)
    # ---------------------
    # Model
    # ---------------------
    match_model = LitMatchStars(config)

    # ---------------------
    # Callbacks
    # ---------------------
    callbacks = list()
    callbacks.append(ModelSummary(max_depth=2))
    # ---------------------
    # Eval
    # ---------------------
    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        default_root_dir=None,
        devices=gpus,
        log_every_n_steps=10,
        precision=config.train.precision,
        move_metrics_to_cpu=False,
    )
    with torch.inference_mode():
        trainer.test(model=match_model, ckpt_path=ckpt_path, datamodule=data_module)


if __name__ == '__main__':
    eval()
