import os
import shutil
import sys
from pathlib import Path

# isort: skip
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks import CustomCheckpointCallback, ProgressLoggerCallback
from core.checkpoints import require_checkpoint_path, resolve_project_path
from core.reproducibility import seed_everything_full
from pl_modules.data.pl_adapt_data import AdaptDistortionModule
from pl_modules.models.pl_correct import LitDistortionCorrect
from nagsn_runtime.adaptation import AdaptCorrectCl


def _single_gpu_ids(gpu_config):
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    if len(gpus) != 1 or not isinstance(gpus[0], int):
        raise ValueError(
            "Adaptation training supports exactly one integer GPU id; "
            f"got {gpus!r}."
        )
    return gpus


def _find_checkpoint_callback(callbacks):
    checkpoint_callback = next(
        (callback for callback in callbacks if isinstance(callback, CustomCheckpointCallback)),
        None,
    )
    if checkpoint_callback is None:
        raise RuntimeError("Adaptation training requires CustomCheckpointCallback.")
    return checkpoint_callback


@hydra.main(config_path="../config", config_name="identify", version_base="1.2")
def main(config: DictConfig):
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")

    gpus = _single_gpu_ids(config.hardware.gpus)
    if config.reproduce.is_open:
        seed_everything_full(config.reproduce.seed)

    callbacks = [
        ProgressLoggerCallback(refresh_rate=100),
        ModelSummary(max_depth=2),
        CustomCheckpointCallback(
            monitor=str(config.adapt_correct.checkpoint.monitor),
            mode=str(config.adapt_correct.checkpoint.mode),
            dirpath=str(resolve_project_path(config.adapt_correct.checkpoint.dirpath)),
        ),
    ]

    adapt = AdaptCorrectCl(
        data_path=config.data_path,
        config=config,
        device=f"cuda:{gpus[0]}",
    )
    adapt.forward()
    if adapt.image_size <= 0:
        raise RuntimeError("Adaptation did not determine a valid image size.")

    data_module = AdaptDistortionModule(adapt.save_path, config.correct, adapt.image_size)
    correct_model = LitDistortionCorrect(config.correct, adapt.image_size)

    logger_cfg = config.logger
    logger = TensorBoardLogger(
        save_dir=logger_cfg.default_root,
        name=logger_cfg.name,
        version=logger_cfg.version,
        log_graph=logger_cfg.log_graph,
    )

    val_check_interval = config.correct.val.check_interval
    check_val_every_n_epoch = config.correct.val.check_every_n_epoch
    if val_check_interval is not None and check_val_every_n_epoch is not None:
        raise ValueError("Only one of correct.val.check_interval and check_every_n_epoch may be set.")

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        devices=gpus,
        limit_train_batches=config.correct.train.limit_batches,
        limit_val_batches=config.correct.val.limit_batches,
        log_every_n_steps=config.correct.train.log_every_n_steps,
        precision=config.correct.train.precision,
        max_epochs=config.correct.train.max_epochs,
        max_steps=config.correct.train.max_steps,
        move_metrics_to_cpu=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    resume_checkpoint = (
        require_checkpoint_path(config.correct.train.ckpt_path, "correct.train.ckpt_path")
        if config.correct.train.ckpt_path is not None else None
    )
    trainer.fit(
        model=correct_model,
        ckpt_path=resume_checkpoint,
        datamodule=data_module,
    )
    print("Training completed.")

    checkpoint_callback = _find_checkpoint_callback(callbacks)
    mode = str(config.adapt_correct.mode).lower()
    if mode == "last":
        checkpoint_path = checkpoint_callback.last_ckpt_path
    elif mode == "top":
        checkpoint_path = checkpoint_callback.top_ckpt_path
    else:
        raise ValueError("adapt_correct.mode must be either 'last' or 'top'.")
    if checkpoint_path is None:
        raise RuntimeError("No checkpoint was saved during adaptation training.")

    target_dir = resolve_project_path(config.adapt_correct.target_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{adapt.image_size}_{os.path.basename(config.data_path)}.ckpt"
    if target_path.exists() and not bool(config.adapt_correct.overwrite_target):
        raise FileExistsError(
            f"Refusing to overwrite existing adapted checkpoint: {target_path}. "
            "Set adapt_correct.overwrite_target=true to replace it explicitly."
        )
    print(f"Copying the checkpoint `{checkpoint_path}` to `{target_path}`")
    shutil.copy2(checkpoint_path, target_path)
    print("Finished!")


if __name__ == "__main__":
    main()
