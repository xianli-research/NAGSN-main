import os
import sys

# isort: skip
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from pl_modules.models.pl_match import LitMatchStars
from callbacks import CustomCheckpointCallback, ProgressLoggerCallback

from pl_modules.data import StarInfoDataModule
from core.checkpoints import require_checkpoint_path, resolve_project_path
from core.reproducibility import seed_everything_full


@hydra.main(config_path='../config', config_name='match', version_base='1.2')
def train(config: DictConfig):
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # DDP
    # ---------------------
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    if not (
        isinstance(gpus, list) and all(isinstance(g, int) for g in gpus)
        or isinstance(gpus, int)
    ):
        raise ValueError(f"gpus must be int or List[int], got: {type(gpus)}, content: {gpus}")
    gpus = gpus if isinstance(gpus, list) else [gpus]
    
    distributed_backend = config.hardware.dist_backend
    if distributed_backend not in ('nccl', 'gloo'):
        raise ValueError(f"hardware.dist_backend must be 'nccl' or 'gloo', got {distributed_backend!r}.")
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None
                  
    # ---------------------
    # Reproducibility
    # ---------------------
    if config.reproduce.is_open:
        seed_everything_full(config.reproduce.seed)

    # ---------------------
    # Callbacks
    # ---------------------
    callbacks = list()
    callbacks.append(ProgressLoggerCallback(refresh_rate=10))
    callbacks.append(ModelSummary(max_depth=2))
    checkpoint_cfg = config.checkpoint
    callbacks.append(
        CustomCheckpointCallback(
            monitor=str(checkpoint_cfg.monitor),
            mode=str(checkpoint_cfg.mode),
            dirpath=str(resolve_project_path(checkpoint_cfg.dirpath)),
        )
    )

    # ---------------------
    # Data
    # ---------------------
    data_module = StarInfoDataModule(config)

    # ---------------------
    # Model
    # ---------------------
    match_model = LitMatchStars(config)

    # ---------------------
    # Logger
    # ---------------------
    logger_cfg = config.logger
    logger = TensorBoardLogger(
        save_dir=logger_cfg.default_root, 
        name=logger_cfg.name, 
        version=logger_cfg.version,
        log_graph=logger_cfg.log_graph,
    )

    # ---------------------
    # Training
    # ---------------------
    val_check_interval = config.val.check_interval
    check_val_every_n_epoch = config.val.check_every_n_epoch
    if val_check_interval is not None and check_val_every_n_epoch is not None:
        raise ValueError(
            "Only one of val.check_interval and val.check_every_n_epoch may be set."
        )

    trainer = pl.Trainer(
        accelerator='gpu',
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        devices=gpus,
        limit_train_batches=config.train.limit_batches,
        limit_val_batches=config.val.limit_batches,
        log_every_n_steps=config.train.log_every_n_steps,
        plugins=None,
        precision=config.train.precision,
        max_epochs=config.train.max_epochs,
        max_steps=config.train.max_steps,
        strategy=strategy,
        sync_batchnorm=False if strategy is None else True,
        move_metrics_to_cpu=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        # deterministic=config.reproduce.deterministic_flag,
    )
    ckpt_path = (
        None
        if config.train.ckpt_path is None
        else require_checkpoint_path(config.train.ckpt_path, "train.ckpt_path")
    )
    trainer.fit(model=match_model, ckpt_path=ckpt_path, datamodule=data_module)


if __name__ == '__main__':
    train()
