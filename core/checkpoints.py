import hashlib
import json
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_value) -> Path:
    """Resolve relative project paths independently of Hydra's run directory."""
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_checkpoint_path(ckpt_path, config_key: str = "test.ckpt_path") -> str:
    """Validate and resolve an existing checkpoint path from the project root."""
    if ckpt_path is None or not str(ckpt_path).strip():
        raise ValueError(f"{config_key} must point to an existing checkpoint file.")

    path = resolve_project_path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{config_key} must reference an existing checkpoint file, got: {path}"
        )
    return str(path.resolve())


def model_config_fingerprint(model_config, *, image_size: Optional[int] = None) -> str:
    """Create a stable fingerprint for the architecture used by a checkpoint."""
    if OmegaConf.is_config(model_config):
        model_config = OmegaConf.to_container(model_config, resolve=True)
    payload = {"model": model_config}
    if image_size is not None:
        payload["image_size"] = int(image_size)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

