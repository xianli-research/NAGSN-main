import os
import random
import string
import math

from pytorch_lightning.callbacks import Callback


def generate_run_id(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


class CustomCheckpointCallback(Callback):
    def __init__(self, monitor: str = "error_max", mode: str = "min", dirpath: str = "checkpoints"):
        super().__init__()
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}.")

        self.monitor = monitor
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.last_ckpt_path = None
        self.top_ckpt_path = None
        self.dirpath = os.path.abspath(os.path.join(dirpath, generate_run_id()))
        print(f"The checkpoints will be saved in {self.dirpath}")

    @property
    def state_key(self):
        return f"{self.__class__.__qualname__}[monitor={self.monitor},mode={self.mode}]"

    def state_dict(self):
        return {
            "best_score": self.best_score,
            "last_ckpt_path": self.last_ckpt_path,
            "top_ckpt_path": self.top_ckpt_path,
            "dirpath": self.dirpath,
        }

    def load_state_dict(self, state_dict):
        self.best_score = float(state_dict.get("best_score", self.best_score))
        self.last_ckpt_path = state_dict.get("last_ckpt_path")
        self.top_ckpt_path = state_dict.get("top_ckpt_path")
        self.dirpath = state_dict.get("dirpath", self.dirpath)

    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            print(f"[WARN] Metric '{self.monitor}' not found. Skipping save.")
            return

        current_score = float(current_score.item())
        if not math.isfinite(current_score):
            print(
                f"[WARN] Metric '{self.monitor}' is non-finite ({current_score}); "
                "skipping checkpoint save."
            )
            return

        os.makedirs(self.dirpath, exist_ok=True)
        epoch = trainer.current_epoch
        step = trainer.global_step

        last_path = os.path.join(self.dirpath, f"last-epoch{epoch}-step{step}.ckpt")
        previous_last_path = self.last_ckpt_path
        previous_best_score = self.best_score
        previous_top_path = self.top_ckpt_path
        is_better = current_score > self.best_score if self.mode == "max" else current_score < self.best_score
        top_path = None
        self.last_ckpt_path = last_path
        if is_better:
            self.best_score = current_score
            top_path = os.path.join(
                self.dirpath,
                f"top-epoch{epoch}-step{step}-{self.monitor}-{current_score:.2f}.ckpt",
            )
            self.top_ckpt_path = top_path
        try:
            trainer.save_checkpoint(last_path)
            if top_path is not None:
                trainer.save_checkpoint(top_path)
        except Exception:
            self.last_ckpt_path = previous_last_path
            self.best_score = previous_best_score
            self.top_ckpt_path = previous_top_path
            if last_path != previous_last_path and os.path.exists(last_path):
                os.remove(last_path)
            if top_path is not None and top_path != previous_top_path and os.path.exists(top_path):
                os.remove(top_path)
            raise

        if previous_last_path and previous_last_path != last_path and os.path.exists(previous_last_path):
            os.remove(previous_last_path)
        if top_path is not None and previous_top_path and previous_top_path != top_path and os.path.exists(previous_top_path):
            os.remove(previous_top_path)
        if top_path is not None:
            print(f"\n[INFO] New best {self.monitor}: {current_score:.4f} -> {top_path}\n")
        else:
            print(
                f"\n[INFO] Validation {self.monitor}: {current_score:.4f} "
                f"(best: {self.best_score:.4f})\n"
            )

        if (
            trainer.logger is not None
            and hasattr(trainer.logger, "experiment")
            and hasattr(trainer.logger.experiment, "add_scalar")
        ):
            trainer.logger.experiment.add_scalar(  # type: ignore[union-attr]
                f"custom_metrics/{self.monitor}", current_score, epoch
            )
