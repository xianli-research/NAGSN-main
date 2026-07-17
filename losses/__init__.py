from torch import nn

from .infonce import InfoNceLoss


_LOSS_FACTORIES = {
    "huber": nn.HuberLoss,
    "infonce": InfoNceLoss,
}


def get_loss(loss_name: str, **kwargs) -> nn.Module:
    """Create one of the losses supported by the public release."""
    try:
        factory = _LOSS_FACTORIES[loss_name.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(_LOSS_FACTORIES))
        raise ValueError(
            f"Unsupported loss {loss_name!r}; supported losses: {supported}."
        ) from error
    return factory(**kwargs)
