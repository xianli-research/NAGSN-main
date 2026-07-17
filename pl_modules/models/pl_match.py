import hashlib
import json
import warnings

import torch as th
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf

from models.match import NAGSModel
from nagsn_runtime.matching import match_metric, metric_calculate
from losses import get_loss
from core.checkpoints import model_config_fingerprint


def _checkpoint_compatibility_config(config: DictConfig) -> dict:
    """Return fields that must agree when loading a matching checkpoint."""
    return {
        "model": OmegaConf.to_container(config.model.match, resolve=True),
        "loss": OmegaConf.to_container(config.loss, resolve=True),
        "dataset": OmegaConf.to_container(config.dataset, resolve=True),
    }


def _config_fingerprint(config: dict) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _aggregate_match_stats(gathered_stats: th.Tensor) -> th.Tensor:
    """Reduce gathered ``[tp, fp, fn]`` counts on one or more ranks."""
    if gathered_stats.ndim == 0 or gathered_stats.shape[-1] != 3:
        raise RuntimeError(
            "Gathered matching statistics must end with three values "
            f"[tp, fp, fn], got shape {tuple(gathered_stats.shape)}."
        )
    return gathered_stats.reshape(-1, 3).sum(dim=0)


class LitMatchStars(pl.LightningModule):
    def __init__(self,
                 config: DictConfig):
        super().__init__()
        resolved_config = OmegaConf.to_container(config, resolve=True)
        compatibility_config = _checkpoint_compatibility_config(config)
        self._compatibility_fingerprint = _config_fingerprint(compatibility_config)
        self._model_fingerprint = model_config_fingerprint(config.model.match)
        self.save_hyperparameters(
            {
                "resolved_config": resolved_config,
                "match_compatibility_config": compatibility_config,
                "match_compatibility_fingerprint": self._compatibility_fingerprint,
                "match_model_fingerprint": self._model_fingerprint,
            }
        )
        self.train_cfg = config.train
        self.mdl_cfg = config.model.match
        self.loss_cfg = config.loss
        if self.mdl_cfg.name != "nags":
            raise ValueError(
                f"Unsupported match model {self.mdl_cfg.name!r}; only 'nags' "
                "is included in this release."
            )
        self.model = NAGSModel(self.mdl_cfg.backbone)

        loss_cfg_dict = OmegaConf.to_container(self.loss_cfg.parameters, resolve=True)
        self.criterion = get_loss(self.loss_cfg.name, **loss_cfg_dict) # type: ignore

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint.get("hyper_parameters", {})
        saved_fingerprint = saved.get("match_compatibility_fingerprint")
        if saved_fingerprint is None:
            warnings.warn(
                "Checkpoint has no matching configuration fingerprint; compatibility "
                "cannot be verified. Re-save it with the current release.",
                UserWarning,
            )
            return
        if saved_fingerprint != self._compatibility_fingerprint:
            raise ValueError(
                "Checkpoint configuration is incompatible with the current matching "
                "model/loss/dataset configuration. Use the checkpoint's saved config "
                "or start a new run."
            )

    def forward(self, samples):
        graph1, graph2, match_batch = samples

        x1 = self.model(graph1)
        x2 = self.model(graph2)
        return x1, x2, match_batch

    def training_step(self, batch, batch_idx):
        x1, x2, match_batch = self(batch)
        loss = self.criterion(x1.x, x2.x, match_batch.match_pairs)
        self.log("loss", loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=x1.num_graphs)
        return loss

    def _eval_step(self, batch, batch_idx):
        with th.inference_mode():
            k = self.mdl_cfg.graph.score_thres
            max_exact_nodes = int(self.mdl_cfg.graph.max_exact_nodes)
            x1, x2, match_batch = self(batch)
            metric_device = x1.x.device
            x1 = x1.detach().to_data_list()
            x2 = x2.detach().to_data_list()
            match_pairs = match_batch.to_list()
            stats = th.zeros(3, dtype=th.long, device=metric_device)
            for data1, data2, pair in zip(x1, x2, match_pairs):
                result = match_metric(
                    data1.x,
                    data2.x,
                    pair,
                    k,
                    max_exact_nodes=max_exact_nodes,
                )
                stats += th.as_tensor(result, dtype=th.long, device=metric_device)
        return stats

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def _log_match_metrics(self, outputs):
        if not outputs:
            raise RuntimeError("Validation/test produced no matching statistics.")
        local_stats = th.stack(outputs, dim=0).sum(dim=0)
        global_stats = _aggregate_match_stats(self.all_gather(local_stats))
        tp, fp, fn = global_stats.unbind()
        f1, precision, recall = metric_calculate(tp, fp, fn)

        # ``global_stats`` is identical on every rank; sync_dist keeps the
        # Lightning callback metric contract explicit for DDP checkpointing.
        self.log("f1", f1, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("precision", precision, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("recall", recall, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

    def validation_epoch_end(self, outputs):
        self._log_match_metrics(outputs)

    def test_epoch_end(self, outputs):
        self._log_match_metrics(outputs)

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.model.parameters(),
                                   lr=self.train_cfg.learning_rate)

        scheduler_cfg = self.train_cfg.lr_scheduler
        if scheduler_cfg.is_open:
            scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=scheduler_cfg.factor,
                patience=scheduler_cfg.patience,
                threshold=scheduler_cfg.threshold,
                min_lr=scheduler_cfg.min_lr,
                verbose=True,
            )

            return {'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': scheduler_cfg.monitor,
                        'interval': scheduler_cfg.interval,
                        'frequency': scheduler_cfg.frequency,
                        'strict': False
                    }}
        else:
            return optimizer
