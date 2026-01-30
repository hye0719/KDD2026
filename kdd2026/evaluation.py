"""
Evaluation Script for Trained Models

Usage:
    python evaluation.py --config config/configure.yaml --checkpoint checkpoints/fno_1d_best.pth
    python evaluation.py --config config/configure.yaml --checkpoint checkpoints/specboost_2d_best.pth --model specboost
"""

import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy import stats

from data.dataset import get_dataloaders
from model import get_model
from model.specboost import get_specboost_models, TwoStageSpecBoost
from utils import get_device


def load_model(config, checkpoint_path, model_type, dim, device):
    """
    체크포인트에서 모델 로드
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_type == "specboost":
        model_0, model_1 = get_specboost_models(config, dim=dim)
        model_0.load_state_dict(checkpoint['model_0'])
        model_1.load_state_dict(checkpoint['model_1'])
        model = TwoStageSpecBoost(model_0, model_1)
    else:
        config['model']['type'] = model_type
        model = get_model(config, dim=dim)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            for key in checkpoint.keys():
                if 'state_dict' in key.lower():
                    model.load_state_dict(checkpoint[key])
                    break
            else:
                model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def compute_metrics(y_true, y_pred):
    """
    평가 지표 계산
    """
    # Flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # MSE
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    
    # MAE
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # R² Score
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Pearson correlation
    pearson_r, _ = stats.pearsonr(y_true_flat, y_pred_flat)
    
    # Spearman correlation
    spearman_r, _ = stats.spearmanr(y_true_flat, y_pred_flat)
    
    # Relative error
    relative_error = np.mean(np.abs(y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-8))
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Pearson_R': pearson_r,
        'Spearman_R': spearman_r,
        'Relative_Error': relative_error,
    }


def evaluate(config, checkpoint_path, model_type=None, dim=None, save_results=True):
    """
    모델 평가 메인 함수
    """
    # 설정
    general_cfg = config['general']
    device = get_device(general_cfg.get('device', 'auto'))
    
    model_type = model_type if model_type else config['model']['type']
    dim = dim if dim else config['model']['dim']
    
    print(f"📦 Model: {model_type.upper()} ({dim.upper()})")
    print(f"📂 Checkpoint: {checkpoint_path}")
    
    # 데이터 로드
    dataset, _, val_loader = get_dataloaders(config, dim=dim)
    
    # 모델 로드
    model = load_model(config, checkpoint_path, model_type, dim, device)
    
    # 예측
    all_true = []
    all_pred = []
    
    print("\n🔄 Running evaluation...")
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            # 2D 모델의 경우 permute (specboost 제외)
            if dim == "2d" and model_type != "specboost":
                x_in = x.permute(0, 2, 3, 1).contiguous()
            else:
                x_in = x
            
            pred = model(x_in)
            
            # Denormalize
            y_denorm = dataset.denormalize(y.cpu().numpy())
            pred_denorm = dataset.denormalize(pred.cpu().numpy())
            
            all_true.append(y_denorm)
            all_pred.append(pred_denorm)
    
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    
    # 지표 계산
    metrics = compute_metrics(all_true, all_pred)
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("📊 Evaluation Results")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    print("=" * 50)
    
    # 결과 저장
    if save_results:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        result_path = os.path.join(checkpoint_dir, 'evaluation_results.yaml')
        with open(result_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        print(f"📝 Results saved to {result_path}")
    
    return metrics, all_true, all_pred


def main():
    parser = argparse.ArgumentParser(description='Evaluate Metalens Models')
    parser.add_argument('--config', type=str, default='config/configure.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--model', type=str, default=None,
                        choices=['fno', 'linear', 'unet', 'specboost'],
                        help='Model type (overrides config)')
    parser.add_argument('--dim', type=str, default=None,
                        choices=['1d', '2d'],
                        help='Dimension (overrides config)')
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 평가 실행
    evaluate(config, args.checkpoint, model_type=args.model, dim=args.dim)


if __name__ == "__main__":
    main()
