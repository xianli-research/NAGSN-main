import sys
from pathlib import Path

# isort: skip
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary

from pl_modules.data.pl_distortion import DistortionDataModule
from pl_modules.models.pl_correct import LitDistortionCorrect
from core.checkpoints import require_checkpoint_path


@hydra.main(config_path='../config', config_name='correct', version_base='1.2')
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
    data_module = DistortionDataModule(config)
    # ---------------------
    # Model
    # ---------------------
    if data_module.image_size is None:
        raise RuntimeError("Image size was not initialized by the data module.")
    correct_model = LitDistortionCorrect(config, data_module.image_size)

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
        trainer.test(model=correct_model, ckpt_path=ckpt_path, datamodule=data_module)


if __name__ == '__main__':
    eval()
