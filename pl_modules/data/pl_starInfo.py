from typing import Optional
from omegaconf import DictConfig

import torch as th
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.star_match import StarInfoData, collate_fn


class StarInfoDataModule(pl.LightningDataModule):
    def __init__(self,
                 config: DictConfig):
        super().__init__()
        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            if self.train_dataset is None:
                self.train_dataset = StarInfoData(
                    self.config.data_path,
                    'train',
                    self.config.dataset,
                )
            if self.val_dataset is None:
                self.val_dataset = StarInfoData(
                    self.config.data_path,
                    'val',
                    self.config.dataset,
                )
        elif stage == 'validate' and self.val_dataset is None:
            self.val_dataset = StarInfoData(
                self.config.data_path,
                'val',
                self.config.dataset,
            )

        if stage in (None, 'test') and self.test_dataset is None:
            self.test_dataset = StarInfoData(
                self.config.data_path,
                'test',
                self.config.dataset,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not initialized; call setup('fit') first.")
        return self._build_dataloader(
            self.train_dataset,
            self.config.train,
            generator=self._make_generator(),
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset is not initialized; call setup('fit' or 'validate') first.")
        return self._build_dataloader(
            self.val_dataset,
            self.config.val,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not initialized; call setup('test') first.")
        return self._build_dataloader(
            self.test_dataset,
            self.config.test,
        )

    def _build_dataloader(self, dataset, cfg, generator=None):
        num_workers = int(cfg.num_workers)
        persistent_workers = bool(cfg.persistent_workers) and num_workers > 0

        return DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            collate_fn=collate_fn,
            shuffle=bool(cfg.shuffle),
            num_workers=num_workers,
            pin_memory=bool(cfg.pin_memory),
            persistent_workers=persistent_workers,
            generator=generator,
        )

    def _make_generator(self):
        if not bool(self.config.reproduce.is_open):
            return None
        return th.Generator().manual_seed(int(self.config.reproduce.seed))
