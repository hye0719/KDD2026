"""
Utility Functions for Metalens Project
- Device detection (CUDA, MPS, CPU)
- Path management
- Config handling
"""

import os
import shutil
import torch
import yaml
from datetime import datetime


def get_device(device_config="auto"):
    """
    Device 자동 감지 및 설정
    
    Args:
        device_config: "auto", "cuda", "mps", "cpu"
    
    Returns:
        torch.device
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🔥 Device: CUDA ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"🍎 Device: MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print(f"💻 Device: CPU")
    elif device_config == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🔥 Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print("⚠️ CUDA not available, falling back to CPU")
            device = torch.device("cpu")
    elif device_config == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"🍎 Device: MPS (Apple Silicon)")
        else:
            print("⚠️ MPS not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print(f"💻 Device: CPU")
    
    return device


def create_experiment_dir(config, model_type, dim):
    """
    실험 디렉토리 생성 (checkpoint, result)
    
    Args:
        config: 설정 딕셔너리
        model_type: 모델 타입
        dim: 차원
    
    Returns:
        checkpoint_dir, result_dir
    """
    output_cfg = config['output']
    
    # 타임스탬프
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 실험 이름
    exp_name = f"{model_type}_{dim}_{timestamp}"
    
    # 디렉토리 경로
    checkpoint_dir = os.path.join(output_cfg['checkpoint_dir'], exp_name)
    result_dir = os.path.join(output_cfg['result_dir'], exp_name)
    
    # 디렉토리 생성
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"📁 Checkpoint dir: {checkpoint_dir}")
    print(f"📁 Result dir: {result_dir}")
    
    return checkpoint_dir, result_dir


def save_config(config, save_dir, filename="config.yaml"):
    """
    설정 파일을 저장 디렉토리에 복사
    
    Args:
        config: 설정 딕셔너리
        save_dir: 저장 디렉토리
        filename: 파일명
    """
    save_path = os.path.join(save_dir, filename)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"📝 Config saved to {save_path}")


def load_config(config_path):
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        config dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """
    랜덤 시드 설정
    
    Args:
        seed: 시드 값
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # MPS의 경우 별도 시드 설정 없음
    
    print(f"🎲 Random seed set to {seed}")


def count_parameters(model):
    """
    모델 파라미터 수 계산
    
    Args:
        model: PyTorch 모델
    
    Returns:
        총 파라미터 수, 학습 가능 파라미터 수
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_info(model, model_type, dim):
    """
    모델 정보 출력
    
    Args:
        model: PyTorch 모델
        model_type: 모델 타입
        dim: 차원
    """
    total, trainable = count_parameters(model)
    
    print(f"\n{'='*50}")
    print(f"📦 Model: {model_type.upper()} ({dim.upper()})")
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    print(f"{'='*50}\n")
