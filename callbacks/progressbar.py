"""Low-overhead, rank-zero progress logging for Lightning workflows."""

import time

import torch as th
from pytorch_lightning.callbacks import Callback


class ProgressLoggerCallback(Callback):
    def __init__(self, refresh_rate: int = 10):
        super().__init__()
        if isinstance(refresh_rate, bool) or not isinstance(refresh_rate, int) or refresh_rate <= 0:
            raise ValueError(f"refresh_rate must be a positive integer, got {refresh_rate!r}.")
        self.refresh_rate = refresh_rate
        self.total_iter_count = 0
        self._last_print_iter = {}
        self.start_time = None
        self.last_time = None

    @staticmethod
    def _format_time(seconds):
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _ensure_timer(self):
        now = time.monotonic()
        if self.start_time is None:
            self.start_time = now
        if self.last_time is None:
            self.last_time = now
        return now

    def _should_print(self, stage, batch_idx, total_batches):
        last_print_iter = self._last_print_iter.get(stage, -1)
        return (
            batch_idx == 0
            or batch_idx - last_print_iter >= self.refresh_rate
            or (total_batches is not None and batch_idx == total_batches - 1)
        )

    @staticmethod
    def _loader_batch_count(batch_counts, dataloader_idx):
        if isinstance(batch_counts, (list, tuple)):
            if dataloader_idx >= len(batch_counts):
                return None
            return batch_counts[dataloader_idx]
        return batch_counts

    @staticmethod
    def _loss_from_outputs(outputs):
        if isinstance(outputs, dict):
            outputs = outputs.get("loss")
        if isinstance(outputs, th.Tensor) and outputs.numel() == 1:
            return float(outputs.detach().item())
        return None

    @staticmethod
    def _version(trainer):
        logger = trainer.logger
        return str(logger.version) if hasattr(logger, "version") else "unknown"

    def _print(self, trainer, message):
        if trainer.is_global_zero:
            # ``Trainer.print`` is not part of the Lightning 1.x public API.
            # Rank-zero gating above keeps ordinary stdout safe under DDP.
            print(message, flush=True)

    def on_train_start(self, trainer, pl_module):
        self._ensure_timer()

    def on_validation_start(self, trainer, pl_module):
        self._ensure_timer()

    def on_test_start(self, trainer, pl_module):
        self._ensure_timer()

    def on_train_epoch_start(self, trainer, pl_module):
        self.last_time = time.monotonic()
        self._last_print_iter["train"] = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.total_iter_count += 1
        total_batches = trainer.num_training_batches
        if not self._should_print("train", batch_idx, total_batches):
            return

        now = self._ensure_timer()
        elapsed = now - self.last_time
        speed = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0
        loss = self._loss_from_outputs(outputs)
        loss_str = f", loss={loss:.4f}" if loss is not None else ""
        learning_rate = None
        if trainer.optimizers and trainer.optimizers[0].param_groups:
            learning_rate = trainer.optimizers[0].param_groups[0].get("lr")
        lr_str = f", lr={learning_rate:.6f}" if learning_rate is not None else ""
        self._print(
            trainer,
            f"Epoch {trainer.current_epoch}: {batch_idx + 1}it "
            f"[{self.total_iter_count} it] [{self._format_time(now - self.start_time)}, "
            f"{speed:.2f}it/s{loss_str}{lr_str}, v_num={self._version(trainer)}]",
        )
        self._last_print_iter["train"] = batch_idx

    def on_validation_epoch_start(self, trainer, pl_module):
        self.last_time = time.monotonic()
        self._last_print_iter = {
            key: value for key, value in self._last_print_iter.items() if not key.startswith("val:")
        }

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        stage = f"val:{dataloader_idx}"
        total_batches = self._loader_batch_count(trainer.num_val_batches, dataloader_idx)
        if not self._should_print(stage, batch_idx, total_batches):
            return
        now = self._ensure_timer()
        elapsed = now - self.last_time
        speed = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0
        self._print(
            trainer,
            f"Validation DataLoader {dataloader_idx}: {batch_idx + 1}it "
            f"[{self._format_time(now - self.start_time)}, {speed:.2f}it/s]",
        )
        self._last_print_iter[stage] = batch_idx

    def on_test_epoch_start(self, trainer, pl_module):
        self.last_time = time.monotonic()
        self._last_print_iter = {
            key: value for key, value in self._last_print_iter.items() if not key.startswith("test:")
        }

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        stage = f"test:{dataloader_idx}"
        total_batches = self._loader_batch_count(trainer.num_test_batches, dataloader_idx)
        if not self._should_print(stage, batch_idx, total_batches):
            return
        now = self._ensure_timer()
        elapsed = now - self.last_time
        speed = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0
        self._print(
            trainer,
            f"Testing DataLoader {dataloader_idx}: {batch_idx + 1}it "
            f"[{self._format_time(now - self.start_time)}, {speed:.2f}it/s]",
        )
        self._last_print_iter[stage] = batch_idx
