"""ClearView: single-image deraining with PyTorch.

Models (U-Net variants, ResNet-UNet, NAFNet, Restormer, Histoformer), losses,
metrics, datasets and a training loop for removing rain from images.

Example:
    >>> from clearview.api import DerainingModel
    >>> model = DerainingModel.from_pretrained(
    ...     'nafnet', weights='clearview-derain-nafnet.pth'
    ... )
    >>> clean = model.process('rainy.jpg', output_path='derained.jpg')
"""

__author__ = "Saumya Kumaar Saksena"
__license__ = "Apache-2.0"

try:
    # Single source of truth for the version: the installed package metadata,
    # which is generated from `pyproject.toml`'s `[project].version`. This
    # avoids the two files silently drifting out of sync.
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("clearview")
except (ImportError, PackageNotFoundError):
    # Fallback for editable/uninstalled checkouts (e.g. running from source
    # without `pip install -e .` having been run yet).
    __version__ = "1.0.0"

from clearview.losses import (
    CombinedLoss,
    EdgeLoss,
    PerceptualLoss,
    SSIMLoss,
)
from clearview.models import (
    AttentionUNet,
    UNet,
    get_model,
    list_models,
)
from clearview.utils import (
    compute_metrics,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Models
    "UNet",
    "AttentionUNet",
    "get_model",
    "list_models",
    # Losses
    "CombinedLoss",
    "PerceptualLoss",
    "SSIMLoss",
    "EdgeLoss",
    # Utils
    "load_checkpoint",
    "save_checkpoint",
    "compute_metrics",
]
