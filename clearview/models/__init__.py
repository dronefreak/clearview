"""Neural network models for image deraining.

This module provides various encoder-decoder architectures optimized for
image restoration tasks, including U-Net and attention-based variants.
"""

from typing import Dict, List, Any, Type


from clearview.models.base import BaseModel
from clearview.models.unet import UNet
from clearview.models.attention_unet import AttentionUNet

# Model registry for factory pattern
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "attn_unet": AttentionUNet,  # Alias
}


def register_model(name: str, model_class: Type[BaseModel]) -> None:
    """Register a custom model class.

    Args:
        name: Name to register the model under
        model_class: Model class (must inherit from BaseModel)

    Example:
        >>> class MyCustomModel(BaseModel):
        ...     pass
        >>> register_model('custom', MyCustomModel)
        >>> model = get_model('custom', in_channels=3, out_channels=3)
    """
    if not issubclass(model_class, BaseModel):
        raise ValueError(f"Model must inherit from BaseModel, got {model_class}")

    _MODEL_REGISTRY[name.lower()] = model_class


def get_model(name: str, **kwargs: Any) -> BaseModel:
    """Get a model by name using factory pattern.

    Args:
        name: Model name (e.g., 'unet', 'attention_unet')
        **kwargs: Model-specific arguments

    Returns:
        Instantiated model

    Raises:
        ValueError: If model name is not recognized

    Example:
        >>> model = get_model('unet', in_channels=3, out_channels=3)
        >>> model = get_model('attention_unet', encoder='resnet34', pretrained=True)
    """
    name_lower = name.lower()

    if name_lower not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available models: {available}")

    model_class = _MODEL_REGISTRY[name_lower]
    return model_class(**kwargs)


def list_models() -> List[str]:
    """List all available model names.

    Returns:
        List of registered model names

    Example:
        >>> models = list_models()
        >>> print(models)
        ['unet', 'attention_unet', 'attn_unet']
    """
    return sorted(_MODEL_REGISTRY.keys())


__all__ = [
    # Base
    "BaseModel",
    # Models
    "UNet",
    "AttentionUNet",
    # Factory functions
    "register_model",
    "get_model",
    "list_models",
]
