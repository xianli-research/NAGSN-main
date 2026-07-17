import hashlib
import json
import warnings

import torch as th
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from losses import get_loss
from models.distortion import HCMModel
from core.checkpoints import model_config_fingerprint


def _correct_compatibility_config(config: DictConfig, image_size: int) -> dict:
    """Return the configuration fields that affect correction checkpoint meaning."""
    return {
        "model": OmegaConf.to_container(config.model.correct, resolve=True),
        "loss": OmegaConf.to_container(config.loss, resolve=True),
        "dataset": OmegaConf.to_container(config.dataset, resolve=True),
        "image_size": int(image_size),
    }


def _config_fingerprint(config: dict) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _aggregate_error_stats(gathered_stats: th.Tensor) -> th.Tensor:
    """Reduce gathered ``[count, sum, sum_sq, max]`` error statistics."""
    if gathered_stats.ndim == 0 or gathered_stats.shape[-1] != 4:
        raise RuntimeError(
            "Gathered correction statistics must end with four values "
            f"[count, sum, sum_sq, max], got shape {tuple(gathered_stats.shape)}."
        )
    stats = gathered_stats.reshape(-1, 4)
    return th.stack(
        [
            stats[:, 0].sum(),
            stats[:, 1].sum(),
            stats[:, 2].sum(),
            stats[:, 3].max(),
        ]
    )


class LitDistortionCorrect(pl.LightningModule):
    def __init__(self,
                 config: DictConfig,
                 image_size: int):
        super().__init__()
        self.config = config
        self.image_size = image_size
        resolved_config = OmegaConf.to_container(config, resolve=True)
        compatibility_config = _correct_compatibility_config(config, image_size)
        self._compatibility_fingerprint = _config_fingerprint(compatibility_config)
        self._model_fingerprint = model_config_fingerprint(
            config.model.correct,
            image_size=image_size,
        )
        self.save_hyperparameters(
            {
                "resolved_config": resolved_config,
                "correct_compatibility_config": compatibility_config,
                "correct_compatibility_fingerprint": self._compatibility_fingerprint,
                "correct_model_fingerprint": self._model_fingerprint,
            }
        )
        self.model = HCMModel(image_size, config.model.correct)

        if not hasattr(config, "loss") or not hasattr(config.loss, "parameters"):
            raise ValueError(
                "A correction loss configuration is required at loss.name and "
                "loss.parameters."
            )
        loss_cfg_dict = OmegaConf.to_container(config.loss.parameters, resolve=True)
        self.criterion = get_loss(config.loss.name, **loss_cfg_dict)  # type: ignore[arg-type]

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint.get("hyper_parameters", {})
        saved_fingerprint = saved.get("correct_compatibility_fingerprint")
        if saved_fingerprint is None:
            warnings.warn(
                "Checkpoint has no correction configuration fingerprint; "
                "compatibility cannot be verified. Re-save it with the current "
                "release.",
                UserWarning,
            )
            return
        if saved_fingerprint != self._compatibility_fingerprint:
            raise ValueError(
                "Checkpoint configuration is incompatible with the current "
                "correction model/loss/dataset configuration. Use the checkpoint's "
                "saved config or start a new run."
            )

    def forward(self, coords_distort):
        coords_correct = self.model(coords_distort)
        return coords_correct

    def training_step(self, batch, batch_idx):
        coords_ori, coords_distort = batch
        coords_correct = self(coords_distort)
        loss = self.criterion(coords_correct, coords_ori)
        self.log("loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch)

    def _eval_step(self, batch):
        coords_ori, coords_distort = batch
        coords_correct = self(coords_distort)
        errors = th.linalg.vector_norm(coords_correct - coords_ori, dim=1)
        return self._error_stats(errors)

    @staticmethod
    def _error_stats(errors: th.Tensor) -> th.Tensor:
        """Return [count, sum, sum_sq, max] without synchronizing to CPU."""
        if errors.ndim != 1:
            raise ValueError(f"errors must be one-dimensional, got {tuple(errors.shape)}.")
        if errors.numel() == 0:
            return th.stack(
                [
                    errors.new_zeros(()),
                    errors.new_zeros(()),
                    errors.new_zeros(()),
                    errors.new_full((), -th.inf),
                ]
            )
        return th.stack(
            [
                errors.new_tensor(float(errors.numel())),
                errors.sum(),
                errors.square().sum(),
                errors.max(),
            ]
        )

    def validation_epoch_end(self, outputs):
        self._log_error_metrics(outputs)

    def test_epoch_end(self, outputs):
        self._log_error_metrics(outputs)

    def _log_error_metrics(self, outputs):
        if not outputs:
            raise RuntimeError("Validation/test produced no error statistics.")
        stats = th.stack(outputs, dim=0)
        local_stats = th.stack(
            [
                stats[:, 0].sum(),
                stats[:, 1].sum(),
                stats[:, 2].sum(),
                stats[:, 3].max(),
            ]
        )
        global_stats = _aggregate_error_stats(self.all_gather(local_stats))
        count, error_sum, error_sum_sq, error_max = global_stats.unbind()
        if count.item() <= 0:
            raise RuntimeError("Validation/test contained no coordinate samples.")

        error_mean = error_sum / count
        error_std = th.sqrt((error_sum_sq / count - error_mean.square()).clamp_min(0))
        error_rmse = th.sqrt(error_sum_sq / count)
        error_rrmse_percent = error_rmse / (self.image_size * 2**0.5) * 100

        self.log(
            "error_max", error_max, prog_bar=True, on_epoch=True,
            on_step=False, sync_dist=True,
        )
        self.log(
            "error_mean", error_mean, prog_bar=True, on_epoch=True,
            on_step=False, sync_dist=True,
        )
        self.log(
            "error_std", error_std, prog_bar=True, on_epoch=True,
            on_step=False, sync_dist=True,
        )
        self.log(
            "error_rrmse",
            error_rrmse_percent,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.model.parameters(), lr=self.config.train.learning_rate)
        scheduler_cfg = self.config.train.lr_scheduler
        if not scheduler_cfg.is_open:
            return optimizer

        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.factor,
            patience=scheduler_cfg.patience,
            threshold=scheduler_cfg.threshold,
            min_lr=scheduler_cfg.min_lr,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": scheduler_cfg.monitor,
                "interval": scheduler_cfg.interval,
                "frequency": scheduler_cfg.frequency,
                "strict": False,
            },
        }
