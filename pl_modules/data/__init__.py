from .pl_distortion import DistortionDataModule
from .pl_adapt_data import AdaptDistortionModule

__all__ = [
    "AdaptDistortionModule",
    "DistortionDataModule",
    "StarInfoDataModule",
]


def __getattr__(name):
    if name == "StarInfoDataModule":
        from .pl_starInfo import StarInfoDataModule

        globals()[name] = StarInfoDataModule
        return StarInfoDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
