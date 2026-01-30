"""
Visualization Utilities for Metalens Project
- Top-K results visualization and saving
- Consistent visualization for both 1D and 2D
- Saves input, output, ground_truth for each sample
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from data.dataset import get_dataloaders
from model import get_model
from model.specboost import get_specboost_models, TwoStageSpecBoost


def load_model(config, checkpoint_path, model_type, dim, device):
    """체크포인트에서 모델 로드"""
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


def compute_sample_errors(y_true, y_pred):
    """
    각 샘플별 오차 계산
    
    Returns:
        sample_errors: 샘플별 MAE
    """
    # y_true, y_pred: (N, 1, ...) shape
    N = y_true.shape[0]
    sample_errors = []
    
    for i in range(N):
        mae = np.mean(np.abs(y_true[i] - y_pred[i]))
        sample_errors.append(mae)
    
    return np.array(sample_errors)


def get_1d_representation(data, dim):
    """
    데이터를 1D 표현으로 변환 (2D의 경우 y축 평균)
    
    Args:
        data: (1, H, W) or (1, L) shape
        dim: "1d" or "2d"
    
    Returns:
        1D array
    """
    if dim == "1d":
        return data[0]  # (L,)
    else:
        return data[0].mean(axis=1)  # (H, W) -> (H,)


def save_single_sample_visualization(
    sample_idx, 
    x_input, 
    y_true, 
    y_pred, 
    dim, 
    save_dir, 
    error, 
    dpi=150
):
    """
    단일 샘플 시각화 저장
    - 1D Line Plot (1D, 2D 공통)
    - 2D Heatmap (2D만)
    - Input 시각화
    
    Args:
        sample_idx: 샘플 인덱스
        x_input: 입력 데이터 (C, H, W) or (C, L)
        y_true: Ground truth (1, H, W) or (1, L)
        y_pred: Prediction (1, H, W) or (1, L)
        dim: "1d" or "2d"
        save_dir: 저장 디렉토리
        error: MAE 값
        dpi: 저장 해상도
    """
    sample_dir = os.path.join(save_dir, f"sample_{sample_idx:03d}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 1D 표현 추출
    true_1d = get_1d_representation(y_true, dim)
    pred_1d = get_1d_representation(y_pred, dim)
    
    # ==========================================
    # 1. 1D Line Plot (공통)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # True vs Pred
    axes[0].plot(true_1d, 'b-', linewidth=1.5, label='Ground Truth')
    axes[0].plot(pred_1d, 'r--', linewidth=1.5, label='Prediction')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Adjoint Gradient')
    axes[0].set_title(f'Comparison (MAE={error:.6f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Ground Truth only
    axes[1].plot(true_1d, 'b-', linewidth=1.5)
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Adjoint Gradient')
    axes[1].set_title('Ground Truth')
    axes[1].grid(True, alpha=0.3)
    
    # Prediction only
    axes[2].plot(pred_1d, 'r-', linewidth=1.5)
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Adjoint Gradient')
    axes[2].set_title('Prediction')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, 'line_plot.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # 2. 2D Heatmap (2D만)
    # ==========================================
    if dim == "2d":
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Ground Truth
        im0 = axes[0].imshow(y_true[0], aspect='auto', cmap='viridis')
        axes[0].set_title('Ground Truth')
        axes[0].set_xlabel('Y')
        axes[0].set_ylabel('X')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Prediction
        im1 = axes[1].imshow(y_pred[0], aspect='auto', cmap='viridis')
        axes[1].set_title('Prediction')
        axes[1].set_xlabel('Y')
        axes[1].set_ylabel('X')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Error map
        error_map = y_true[0] - y_pred[0]
        im2 = axes[2].imshow(error_map, aspect='auto', cmap='RdBu_r')
        axes[2].set_title(f'Error (MAE={error:.6f})')
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('X')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'heatmap.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
    
    # ==========================================
    # 3. Input 시각화
    # ==========================================
    if dim == "1d":
        channel_names = ['Geometry', 'Grid', 'Edge Map']
        num_channels = min(x_input.shape[0], len(channel_names))
        
        fig, axes = plt.subplots(1, num_channels, figsize=(5*num_channels, 4))
        if num_channels == 1:
            axes = [axes]
        
        for c in range(num_channels):
            axes[c].plot(x_input[c], linewidth=1.5)
            axes[c].set_title(channel_names[c])
            axes[c].set_xlabel('Position')
            axes[c].grid(True, alpha=0.3)
        
    else:  # 2d
        channel_names = ['Geometry', 'Y Grid', 'X Grid', 'Edge Map']
        num_channels = min(x_input.shape[0], len(channel_names))
        
        fig, axes = plt.subplots(1, num_channels, figsize=(4*num_channels, 4))
        if num_channels == 1:
            axes = [axes]
        
        for c in range(num_channels):
            im = axes[c].imshow(x_input[c], aspect='auto', cmap='viridis')
            axes[c].set_title(channel_names[c] if c < len(channel_names) else f'Channel {c}')
            plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, 'input.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # 4. 데이터 저장 (numpy)
    # ==========================================
    np.save(os.path.join(sample_dir, 'input.npy'), x_input)
    np.save(os.path.join(sample_dir, 'ground_truth.npy'), y_true)
    np.save(os.path.join(sample_dir, 'prediction.npy'), y_pred)


def save_top_k_results(
    config, 
    model, 
    val_loader, 
    dataset, 
    device, 
    dim, 
    model_type,
    result_dir,
    k=None
):
    """
    Top-K 결과 저장 (일반 모델용)
    
    Args:
        config: 설정 딕셔너리
        model: 학습된 모델
        val_loader: 검증 데이터로더
        dataset: 데이터셋 (denormalize용)
        device: 디바이스
        dim: "1d" or "2d"
        model_type: 모델 타입
        result_dir: 결과 저장 디렉토리
        k: Top-K 수 (None이면 config에서 가져옴)
    """
    if k is None:
        k = config['output'].get('top_k_samples', 10)
    
    dpi = config.get('visualization', {}).get('dpi', 150)
    
    print(f"\n📊 Saving Top-{k} results...")
    
    model.eval()
    
    # 모든 예측 수집
    all_inputs = []
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            if dim == "2d":
                x_in = x.permute(0, 2, 3, 1).contiguous()
            else:
                x_in = x
            
            pred = model(x_in)
            all_inputs.append(x.cpu().numpy())
            all_true.append(dataset.denormalize(y.cpu().numpy()))
            all_pred.append(dataset.denormalize(pred.cpu().numpy()))

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    
    # 샘플별 오차 계산
    sample_errors = compute_sample_errors(all_true, all_pred)
    
    # Top-K (오차가 가장 작은 K개)
    top_k_indices = np.argsort(sample_errors)[:k]
    #top_k_indices = np.argsort(sample_errors)[-k:]
    # 결과 저장 디렉토리
    top_k_dir = os.path.join(result_dir, f"top_{k}_samples")
    os.makedirs(top_k_dir, exist_ok=True)
    
    # 각 샘플 저장
    for rank, idx in enumerate(top_k_indices):
        save_single_sample_visualization(
            sample_idx=rank + 1,
            x_input=all_inputs[idx],
            y_true=all_true[idx],
            y_pred=all_pred[idx],
            dim=dim,
            save_dir=top_k_dir,
            error=sample_errors[idx],
            dpi=dpi
        )
        print(f"  Saved sample {rank + 1}/{k} (MAE={sample_errors[idx]:.6f})")
    
    # Summary plot
    save_summary_plot(all_true, all_pred, sample_errors, top_k_indices, result_dir, dim, dpi)
    
    print(f"✅ Top-{k} results saved to {top_k_dir}")


def save_top_k_results_specboost(
    config, 
    model_0, 
    model_1, 
    val_loader, 
    dataset, 
    device, 
    dim, 
    result_dir,
    k=None
):
    """
    Top-K 결과 저장 (SpecBoost용)
    """
    if k is None:
        k = config['output'].get('top_k_samples', 10)
    
    dpi = config.get('visualization', {}).get('dpi', 150)
    
    print(f"\n📊 Saving Top-{k} results (SpecBoost)...")
    
    model_0.eval()
    model_1.eval()
    
    # 모든 예측 수집
    all_inputs = []
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            pred_0 = model_0(x)
            input_1 = torch.cat([x, pred_0], dim=1)
            pred_1 = model_1(input_1)
            pred = pred_0 + pred_1
            
            all_inputs.append(x.cpu().numpy())
            all_true.append(dataset.denormalize(y.cpu().numpy()))
            all_pred.append(dataset.denormalize(pred.cpu().numpy()))
    
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    
    # 샘플별 오차 계산
    sample_errors = compute_sample_errors(all_true, all_pred)
    
    # Top-K
    top_k_indices = np.argsort(sample_errors)[:k]
    
    # 결과 저장 디렉토리
    top_k_dir = os.path.join(result_dir, f"top_{k}_samples")
    os.makedirs(top_k_dir, exist_ok=True)
    
    # 각 샘플 저장
    for rank, idx in enumerate(top_k_indices):
        save_single_sample_visualization(
            sample_idx=rank + 1,
            x_input=all_inputs[idx],
            y_true=all_true[idx],
            y_pred=all_pred[idx],
            dim=dim,
            save_dir=top_k_dir,
            error=sample_errors[idx],
            dpi=dpi
        )
        print(f"  Saved sample {rank + 1}/{k} (MAE={sample_errors[idx]:.6f})")
    
    # Summary plot
    save_summary_plot(all_true, all_pred, sample_errors, top_k_indices, result_dir, dim, dpi)
    
    print(f"✅ Top-{k} results saved to {top_k_dir}")


def save_summary_plot(all_true, all_pred, sample_errors, top_k_indices, result_dir, dim, dpi=150):
    """
    전체 결과 요약 플롯 저장
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Flatten for scatter plot
    true_flat = all_true.flatten()
    pred_flat = all_pred.flatten()
    
    # 1. Error distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(sample_errors, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=np.mean(sample_errors), color='r', linestyle='--', linewidth=2, label=f'Mean={np.mean(sample_errors):.4f}')
    ax1.set_xlabel('Sample MAE')
    ax1.set_ylabel('Density')
    ax1.set_title('Error Distribution')
    ax1.legend()
    
    # 2. True vs Pred scatter
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(true_flat[::100], pred_flat[::100], alpha=0.1, s=1, color='steelblue')  # 샘플링
    min_val = min(true_flat.min(), pred_flat.min())
    max_val = max(true_flat.max(), pred_flat.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    r, _ = stats.pearsonr(true_flat, pred_flat)
    ax2.set_xlabel('Ground Truth')
    ax2.set_ylabel('Prediction')
    ax2.set_title(f'Correlation (R={r:.4f})')
    
    # 3. Top-K samples comparison
    ax3 = fig.add_subplot(gs[0, 2])
    k = len(top_k_indices)
    for i, idx in enumerate(top_k_indices[:5]):  # 상위 5개만
        true_1d = get_1d_representation(all_true[idx], dim)
        pred_1d = get_1d_representation(all_pred[idx], dim)
        ax3.plot(true_1d, alpha=0.5, label=f'True #{i+1}')
        ax3.plot(pred_1d, '--', alpha=0.5, label=f'Pred #{i+1}')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Value')
    ax3.set_title('Top-5 Samples Comparison')
    ax3.legend(fontsize=6, ncol=2)
    
    # 4. Error vs Sample index
    ax4 = fig.add_subplot(gs[1, 0])
    sorted_errors = np.sort(sample_errors)
    ax4.plot(sorted_errors, linewidth=1)
    ax4.axhline(y=np.median(sample_errors), color='r', linestyle='--', label=f'Median={np.median(sample_errors):.4f}')
    ax4.set_xlabel('Sample Rank')
    ax4.set_ylabel('MAE')
    ax4.set_title('Sorted Sample Errors')
    ax4.legend()
    
    # 5. Statistics summary
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    stats_text = f"""
    Summary Statistics
    ==================
    Total Samples: {len(sample_errors)}
    
    MAE:
      Mean:   {np.mean(sample_errors):.6f}
      Std:    {np.std(sample_errors):.6f}
      Median: {np.median(sample_errors):.6f}
      Min:    {np.min(sample_errors):.6f}
      Max:    {np.max(sample_errors):.6f}
    
    Correlation:
      Pearson R:  {r:.6f}
    """
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # 6. Best prediction comparison
    ax6 = fig.add_subplot(gs[1, 2])
    best_idx = top_k_indices[0]
    true_1d = get_1d_representation(all_true[best_idx], dim)
    pred_1d = get_1d_representation(all_pred[best_idx], dim)
    ax6.plot(true_1d, 'b-', linewidth=2, label='Ground Truth')
    ax6.plot(pred_1d, 'r--', linewidth=2, label='Prediction')
    ax6.set_xlabel('Position')
    ax6.set_ylabel('Value')
    ax6.set_title(f'Best Sample (MAE={sample_errors[best_idx]:.6f})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'summary.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"📊 Summary plot saved to {os.path.join(result_dir, 'summary.png')}")


def main():
    """
    커맨드라인에서 시각화 실행
    """
    from utils import get_device
    
    parser = argparse.ArgumentParser(description='Visualize Metalens Model Results')
    parser.add_argument('--config', type=str, default='config/configure.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--model', type=str, default=None,
                        choices=['fno', 'linear', 'unet', 'specboost'],
                        help='Model type')
    parser.add_argument('--dim', type=str, default=None,
                        choices=['1d', '2d'],
                        help='Dimension')
    parser.add_argument('--output', type=str, default='./viz_results',
                        help='Output directory')
    parser.add_argument('--k', type=int, default=None,
                        help='Number of top samples to save')
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_type = args.model if args.model else config['model']['type']
    dim = args.dim if args.dim else config['model']['dim']
    k = args.k if args.k else config['output'].get('top_k_samples', 10)
    
    # Device
    device = get_device(config['general'].get('device', 'auto'))
    
    # 데이터 로드
    dataset, _, val_loader = get_dataloaders(config, dim=dim)
    
    # 모델 로드
    model = load_model(config, args.checkpoint, model_type, dim, device)
    
    # 결과 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 시각화 실행
    if model_type == "specboost":
        # SpecBoost의 경우 모델을 다시 로드해야 함
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_0, model_1 = get_specboost_models(config, dim=dim)
        model_0.load_state_dict(checkpoint['model_0'])
        model_1.load_state_dict(checkpoint['model_1'])
        model_0 = model_0.to(device)
        model_1 = model_1.to(device)
        
        save_top_k_results_specboost(
            config=config,
            model_0=model_0,
            model_1=model_1,
            val_loader=val_loader,
            dataset=dataset,
            device=device,
            dim=dim,
            result_dir=args.output,
            k=k
        )
    else:
        save_top_k_results(
            config=config,
            model=model,
            val_loader=val_loader,
            dataset=dataset,
            device=device,
            dim=dim,
            model_type=model_type,
            result_dir=args.output,
            k=k
        )


if __name__ == "__main__":
    if 'seed' in general_cfg:
        set_seed(general_cfg['seed'])
        print("set seed")
    main()
