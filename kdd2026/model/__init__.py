"""
Model Package
"""

from .fno import RobustFNO, RobustFNO2d, get_fno_model
from .specboost import NewFNO, NewFNO2D, TwoStageSpecBoost, get_specboost_models
from .linear import LinearModel, ConvLinearModel, get_linear_model
from .unet import Unet1d, UNet2D, get_unet_model


def get_model(config, dim="1d"):
    """
    모델 생성 통합 함수
    
    Args:
        config: 설정 딕셔너리
        dim: "1d" or "2d"
    
    Returns:
        model
    """
    model_type = config['model']['type'].lower()
    
    if model_type == "fno":
        return get_fno_model(config, dim)
    elif model_type == "linear":
        return get_linear_model(config)
    elif model_type == "unet":
        return get_unet_model(config, dim)
    elif model_type == "specboost":
        return get_specboost_models(config, dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    # FNO
    'RobustFNO',
    'RobustFNO2d',
    'get_fno_model',
    # SpecBoost
    'NewFNO',
    'NewFNO2D',
    'TwoStageSpecBoost',
    'get_specboost_models',
    # Linear
    'LinearModel',
    'ConvLinearModel',
    'get_linear_model',
    # UNet
    'Unet1d',
    'UNet2D',
    'get_unet_model',
    # Unified
    'get_model',
]
