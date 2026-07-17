from typing import Optional
from omegaconf import DictConfig

import torch as th
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from data.distortion import DistortionDataForAdapt


class AdaptDistortionModule(pl.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 config: DictConfig,
                 image_size: int):
        super().__init__()
        self.data_path = data_path
        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.image_size = image_size
        self.coordinate_tolerance = float(
            self.config.dataset.validation.coordinate_tolerance
        )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit', 'validate'):
            self._setup_train_val_datasets()

        if stage in (None, 'test') and self.test_dataset is None:
            self.test_dataset = DistortionDataForAdapt(
                data_path=self.data_path,
                image_size=self.image_size,
                coordinate_tolerance=self.coordinate_tolerance,
            )

    def _setup_train_val_datasets(self):
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        dataset = DistortionDataForAdapt(
            data_path=self.data_path,
            image_size=self.image_size,
            coordinate_tolerance=self.coordinate_tolerance,
        )
        dataset_size = len(dataset)
        train_ratio = float(self.config.adapt.train_ratio)
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(
                f"adapt.train_ratio must be between 0 and 1, got {train_ratio}."
            )

        train_size = int(dataset_size * train_ratio)
        val_size = dataset_size - train_size
        if train_size == 0 or val_size == 0:
            raise ValueError(
                "The adaptation dataset must contain enough samples to create "
                "non-empty training and validation subsets."
            )

        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=self._make_generator(),
        )

    def _make_generator(self):
        if not bool(self.config.reproduce.is_open):
            return None
        return th.Generator().manual_seed(int(self.config.reproduce.seed))

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
            shuffle=bool(cfg.shuffle),
            num_workers=num_workers,
            pin_memory=bool(cfg.pin_memory),
            persistent_workers=persistent_workers,
            generator=generator,
        )
