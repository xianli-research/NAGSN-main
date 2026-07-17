import re
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import torch as th
import torch.utils.data as Data

from omegaconf import DictConfig

from nagsn_runtime.distortion import select_inliers


@dataclass(frozen=True)
class DistortionMetadata:
    image_size: int
    data_type: str


@dataclass(frozen=True)
class DistortionTestGroup:
    source_path: str
    start: int
    stop: int


def load_distortion_metadata(data_dir: str) -> DistortionMetadata:
    info_path = Path(data_dir) / "info.ini"
    if not info_path.is_file():
        raise FileNotFoundError(f"Distortion metadata file does not exist: {info_path}")

    ini_parser = ConfigParser()
    ini_parser.read(info_path)
    if "Info" not in ini_parser:
        raise ValueError(f"Missing [Info] section in metadata file: {info_path}")

    ini_info = ini_parser["Info"]
    image_size = ini_info.getint("image_size")
    data_type = ini_info.get("type", "").strip().lower()
    if image_size <= 0:
        raise ValueError(f"image_size must be positive in {info_path}, got {image_size}.")
    if data_type not in {"capture", "sim"}:
        raise ValueError(
            f"type must be 'capture' or 'sim' in {info_path}, got {data_type!r}."
        )
    return DistortionMetadata(image_size=image_size, data_type=data_type)


def _load_distortion_csv(data_path: str,
                         image_size: int,
                         coordinate_tolerance: float = 2.0) -> th.Tensor:
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Distortion CSV file does not exist: {path}")

    array = pd.read_csv(path, header=None, dtype=np.float32).to_numpy(copy=False)
    if array.ndim != 2 or array.shape[1] != 4:
        columns = array.shape[1] if array.ndim == 2 else "unknown"
        raise ValueError(
            f"Distortion CSV must contain exactly 4 columns, got {columns}: {path}"
        )
    if array.shape[0] == 0:
        raise ValueError(f"Distortion CSV must contain at least one row: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Distortion CSV contains NaN or infinite values: {path}")

    if coordinate_tolerance < 0:
        raise ValueError(
            "coordinate_tolerance must be non-negative, "
            f"got {coordinate_tolerance}."
        )
    min_coord = float(array.min())
    max_coord = float(array.max())
    lower_bound = -coordinate_tolerance
    upper_bound = image_size + coordinate_tolerance
    if min_coord < lower_bound or max_coord > upper_bound:
        raise ValueError(
            f"Coordinates in {path} must be within "
            f"[{lower_bound}, {upper_bound}], "
            f"got range [{min_coord}, {max_coord}]."
        )
    return th.from_numpy(array)


class DistortionData(Data.Dataset):
    def __init__(self,
                 data_dir: str,
                 aug_cfg: DictConfig,
                 load_type: str,
                 metadata: Optional[DistortionMetadata] = None,
                 coordinate_tolerance: float = 2.0):
        super().__init__()

        if load_type not in {"train", "val", "test"}:
            raise ValueError(f"load_type must be 'train', 'val', or 'test', got {load_type!r}.")

        self.data_dir = data_dir
        self.noise_ratio = aug_cfg.noise_ratio
        self.perturb_std = aug_cfg.perturb_std
        self.load_type = load_type
        self.coordinate_tolerance = coordinate_tolerance

        self.metadata = metadata or load_distortion_metadata(data_dir)
        self.image_size = self.metadata.image_size
        self.data_type = self.metadata.data_type

        self.data_path = None
        self.data = None
        self.test_groups = tuple()

        # Resolve the data path from the expected dataset layout.
        self.data_path = self._resolve_data_path()
        if self.load_type == "test":
            self._init_test()
        else:
            if not isinstance(self.data_path, str) or not self.data_path.lower().endswith(".csv"):
                raise RuntimeError(f"Expected a CSV data path, got {self.data_path!r}.")

        # Load the data(csv)
        self._load_data()

    def __len__(self):
        if self.data is None:
            raise RuntimeError("Data is not loaded.")
        if self.data.dim() != 2:
            raise RuntimeError(f"Data must be a 2D tensor, got shape {tuple(self.data.shape)}.")

        L, _ = self.data.shape
        return L

    def __getitem__(self, item):
        if self.data is None:
            raise RuntimeError("Data is not loaded.")
        if self.data.dim() != 2 or self.data.shape[1] != 4:
            raise RuntimeError(f"Data must have shape [N, 4], got {tuple(self.data.shape)}.")
        if not isinstance(item, int) or not 0 <= item < len(self):
            raise IndexError(f"Item index out of range: {item!r}.")

        return self.data[item, :2], self.data[item, -2:]

    def _resolve_data_path(self):
        data_dir = Path(self.data_dir)
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Data directory does not exist: {data_dir}")

        if self.data_type == "capture":
            data_path = data_dir / f"{self.load_type}.csv"
            if not data_path.is_file():
                raise FileNotFoundError(
                    f"Expected capture data file does not exist: {data_path}"
                )
            return str(data_path)

        if self.load_type == "test":
            test_dir = data_dir / "test"
            if not test_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected simulation test directory does not exist: {test_dir}"
                )
            return str(test_dir)

        candidates = self._simulation_csv_paths(data_dir, self.load_type)
        if len(candidates) != 1:
            candidate_names = [path.name for path in candidates]
            raise ValueError(
                f"Expected exactly one simulation {self.load_type!r} CSV in "
                f"{data_dir}, found {len(candidates)}: {candidate_names}"
            )
        return str(candidates[0])

    @staticmethod
    def _simulation_csv_paths(data_dir: Path, load_type: str):
        number_pattern = r"[+-]?\d+(?:\.\d+)?"
        name_pattern = re.compile(
            rf"{re.escape(load_type)}_{number_pattern}_{number_pattern}\.csv"
        )
        return sorted(
            path for path in data_dir.glob(f"{load_type}_*.csv")
            if path.is_file() and name_pattern.fullmatch(path.name)
        )

    def _init_test(self):
        if self.load_type != "test":
            raise RuntimeError("_init_test() may only be called for the test split.")
        if self.data_type == "sim":
            if not isinstance(self.data_path, str) or not Path(self.data_path).is_dir():
                raise RuntimeError(f"Simulation test data path is not a directory: {self.data_path!r}.")
            test_paths = self._simulation_csv_paths(
                Path(self.data_path),
                self.load_type,
            )
            if not test_paths:
                raise FileNotFoundError(
                    f"No CSV files found in simulation test directory: {self.data_path}"
                )
            self.data_path = [str(path) for path in test_paths]

    def _load_data(self):
        if self.data_type == "sim" and self.load_type == "test":
            if not isinstance(self.data_path, list) or not self.data_path:
                raise RuntimeError("Simulation test data paths must be a non-empty list.")
            
            data_parts = []
            test_groups = []
            start = 0
            for path in self.data_path:
                data_part = _load_distortion_csv(
                    path,
                    self.image_size,
                    self.coordinate_tolerance,
                )
                stop = start + len(data_part)
                data_parts.append(data_part)
                test_groups.append(
                    DistortionTestGroup(
                        source_path=path,
                        start=start,
                        stop=stop,
                    )
                )
                start = stop
            data = th.cat(data_parts, dim=0)
            self.test_groups = tuple(test_groups)
        else:
            if not isinstance(self.data_path, str):
                raise RuntimeError(f"Expected a CSV data path, got {self.data_path!r}.")

            data = _load_distortion_csv(
                self.data_path,
                self.image_size,
                self.coordinate_tolerance,
            )
            if self.load_type == "test":
                self.test_groups = (
                    DistortionTestGroup(
                        source_path=self.data_path,
                        start=0,
                        stop=len(data),
                    ),
                )

        if self.load_type == "train":
            if self.noise_ratio > 0:
                num_noise = int(data.shape[0] * self.noise_ratio)
                noise = th.rand(num_noise, data.shape[1]) * self.image_size
                data = th.cat([data, noise], dim=0)
            if self.perturb_std > 0:
                perturb_value = th.normal(mean=0.0,
                                          std=self.perturb_std,
                                          size=data.size())
                data = data + perturb_value

            inliers = select_inliers(ori_coords=data[:, :2],
                                     distort_coords=data[:, -2:],
                                     image_size=self.image_size,
                                     thres_angle=3.0,
                                     thres_length=1.3)
            data = data[inliers]
        self.data = data


class DistortionDataForAdapt(Data.Dataset):
    def __init__(self,
                 data_path: str,
                 image_size: int,
                 coordinate_tolerance: float = 2.0):
        super().__init__()
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}.")
        self.data_path = data_path
        self.image_size = image_size
        self.coordinate_tolerance = coordinate_tolerance

        self._init_data()

    def __len__(self):
        L, _ = self.data.shape
        return L

    def __getitem__(self, item):
        return self.data[item, :2], self.data[item, -2:]

    def _init_data(self):
        self.data = _load_distortion_csv(
            self.data_path,
            self.image_size,
            self.coordinate_tolerance,
        )
