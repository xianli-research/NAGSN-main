import os
import sys
from pathlib import Path

# isort: skip
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from workflows.identify import identify_process


@hydra.main(config_path='../config', config_name='identify', version_base='1.2')
def main(config: DictConfig):
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # Device
    # ---------------------
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(
        gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    if len(gpus) != 1:
        raise ValueError(
            "Identification currently supports exactly one GPU; "
            f"got hardware.gpus={gpus!r}."
        )
    if not isinstance(gpus[0], int) or gpus[0] < 0:
        raise ValueError(f"hardware.gpus must contain one non-negative GPU index, got {gpus!r}.")

    # ---------------------
    # Eval
    # ---------------------
    print("\n================================")
    print("Start processing the real images.")
    print("================================")
    capd_path = os.path.join(config.data_path, "capd")
    if not os.path.isdir(capd_path):
        raise FileNotFoundError(f"CAPD identification data directory does not exist: {capd_path}")
    identify_process(
        data_dir=capd_path,
        config=config,
        device=f"cuda:{gpus[0]}",
    )

    print("\n================================")
    print("Start processing the distortion simulation images.")
    print("================================")
    dist_path = os.path.join(config.data_path, "sim/0")
    if not os.path.isdir(dist_path):
        raise FileNotFoundError(f"Simulation identification data directory does not exist: {dist_path}")
    identify_process(
        data_dir=dist_path,
        config=config,
        device=f"cuda:{gpus[0]}",
    )
    dist_path = os.path.join(config.data_path, "sim/1")
    if not os.path.isdir(dist_path):
        raise FileNotFoundError(f"Simulation identification data directory does not exist: {dist_path}")
    identify_process(
        data_dir=dist_path,
        config=config,
        device=f"cuda:{gpus[0]}",
    )
    dist_path = os.path.join(config.data_path, "sim/2")
    if not os.path.isdir(dist_path):
        raise FileNotFoundError(f"Simulation identification data directory does not exist: {dist_path}")
    identify_process(
        data_dir=dist_path,
        config=config,
        device=f"cuda:{gpus[0]}",
    )
    dist_path = os.path.join(config.data_path, "sim/3")
    if not os.path.isdir(dist_path):
        raise FileNotFoundError(f"Simulation identification data directory does not exist: {dist_path}")
    identify_process(
        data_dir=dist_path,
        config=config,
        device=f"cuda:{gpus[0]}",
    )
    print("Finished!\n")


if __name__ == '__main__':
    main()
