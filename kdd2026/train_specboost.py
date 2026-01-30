"""
Training Script for SpecBoost (Two-Stage Boosting)
Supports both 1D and 2D configurations

Usage:
    python train_specboost.py --config config/configure.yaml
    python train_specboost.py --config config/configure.yaml --dim 2d
"""

import osz
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.dataset import get_dataloaders
from model.specboost import get_specboost_models, TwoStageSpecBoost
from utils import (
    get_device, 
    create_experiment_dir, 
    save_config, 
    set_seed,
    print_model_info
)


def get_criterion(loss_type):
    """손실 함수 생성"""
    if loss_type.lower() == "mse":
        return nn.MSELoss()
    elif loss_type.lower() == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_stage0(model_0, train_loader, optimizer, criterion, device, dim):
    """Stage 0 학습 (Base FNO)"""
    model_0.train()
    total_loss = 0.0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model_0(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def train_stage1(model_0, model_1, train_loader, optimizer, criterion, device, dim):
    """Stage 1 학습 (Residual FNO)"""
    model_0.eval()
    model_1.train()
    total_loss = 0.0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        # Stage 0 prediction (frozen)
        with torch.no_grad():
            pred_0 = model_0(x)
            residual_target = y - pred_0
        
        # Concatenate input with Stage 0 output
        input_1 = torch.cat([x, pred_0], dim=1)
        
        optimizer.zero_grad()
        pred_residual = model_1(input_1)
        loss = criterion(pred_residual, residual_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_specboost(model_0, model_1, val_loader, criterion, device, dim):
    """SpecBoost 검증 (두 모델 결합)"""
    model_0.eval()
    model_1.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            pred_0 = model_0(x)
            input_1 = torch.cat([x, pred_0], dim=1)
            pred_1 = model_1(input_1)
            final_pred = pred_0 + pred_1
            
            total_loss += criterion(final_pred, y).item()
    
    return total_loss / len(val_loader)


def plot_training_results(train_losses, val_losses, save_path):
    """학습 곡선 저장"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Stage 1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Loss curve saved to {save_path}")


def train_specboost(config, override_dim=None):
    """
    SpecBoost 학습 메인 함수
    
    Args:
        config: 설정 딕셔너리
        override_dim: 차원 오버라이드 (옵션)
    """
    # 설정 추출
    general_cfg = config['general']
    train_cfg = config['training']
    sb_train_cfg = config['specboost_training']
    output_cfg = config['output']
    
    dim = override_dim if override_dim else config['model']['dim']
    model_type = "specboost"
    
    # 시드 설정
    if 'seed' in general_cfg:
        set_seed(general_cfg['seed'])
    
    # Device 설정
    device = get_device(general_cfg.get('device', 'auto'))
    
    # 실험 디렉토리 생성
    checkpoint_dir, result_dir = create_experiment_dir(config, model_type, dim)
    
    # 설정 파일 저장
    save_config(config, checkpoint_dir)
    
    # 데이터 로드
    dataset, train_loader, val_loader = get_dataloaders(config, dim=dim)
    
    # 데이터 확인
    x, y = next(iter(train_loader))
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    
    # 모델 생성
    model_0, model_1 = get_specboost_models(config, dim=dim)
    model_0 = model_0.to(device)
    model_1 = model_1.to(device)
    
    # 모델 정보 출력
    print_model_info(model_0, "specboost_stage0", dim)
    print_model_info(model_1, "specboost_stage1", dim)
    
    # 손실 함수
    criterion = get_criterion(sb_train_cfg['loss'])
    
    model_name = f"specboost_{dim}"
    
    # ==========================================
    # Stage 0: Base FNO 학습
    # ==========================================
    stage0_cfg = sb_train_cfg['stage0']
    optimizer_0 = torch.optim.AdamW(
        model_0.parameters(),
        lr=stage0_cfg['learning_rate'],
        weight_decay=stage0_cfg['weight_decay']
    )
    
    print(f"\n🚀 [Stage 0] Training Base FNO...")
    stage0_losses = []
    
    for epoch in range(stage0_cfg['epochs']):
        train_loss = train_stage0(model_0, train_loader, optimizer_0, criterion, device, dim)
        stage0_losses.append(train_loss)
        
        if (epoch + 1) % stage0_cfg['log_interval'] == 0:
            print(f"Stage 0 Epoch {epoch + 1:03d} | Loss: {train_loss:.6f}")
    
    # Freeze Stage 0
    for param in model_0.parameters():
        param.requires_grad = False
    model_0.eval()
    print("✅ Stage 0 training complete. Model frozen.")
    
    # ==========================================
    # Stage 1: Residual FNO 학습
    # ==========================================
    stage1_cfg = sb_train_cfg['stage1']
    optimizer_1 = torch.optim.AdamW(
        model_1.parameters(),
        lr=stage1_cfg['learning_rate'],
        weight_decay=stage1_cfg['weight_decay']
    )
    
    print(f"\n🚀 [Stage 1] Training Residual FNO...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = stage1_cfg.get('patience', 20)
    trigger_times = 0
    
    save_path_best = os.path.join(checkpoint_dir, f"{model_name}{output_cfg['best_model_suffix']}")
    save_path_final = os.path.join(checkpoint_dir, f"{model_name}{output_cfg['final_model_suffix']}")
    
    for epoch in range(stage1_cfg['epochs']):
        # Train
        train_loss = train_stage1(model_0, model_1, train_loader, optimizer_1, criterion, device, dim)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_specboost(model_0, model_1, val_loader, criterion, device, dim)
        val_losses.append(val_loss)
        
        # Log
        if (epoch + 1) % stage1_cfg['log_interval'] == 0:
            print(f"Stage 1 Epoch {epoch + 1:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save({
                'epoch': epoch + 1,
                'model_0': model_0.state_dict(),
                'model_1': model_1.state_dict(),
                'val_loss': best_val_loss,
                'config': config,
            }, save_path_best)
            print(f"✅ Best model saved at epoch {epoch + 1} (Val: {best_val_loss:.6f})")
        else:
            trigger_times += 1
            if stage1_cfg.get('use_early_stopping', False) and trigger_times >= patience:
                print(f"⚠️ Early stopping at epoch {epoch + 1}")
                break
    
    # Save final model
    torch.save({
        'epoch': stage1_cfg['epochs'],
        'model_0': model_0.state_dict(),
        'model_1': model_1.state_dict(),
        'val_loss': val_losses[-1],
        'config': config,
    }, save_path_final)
    print(f"💾 Final model saved to {save_path_final}")
    
    # Plot training curves
    loss_plot_path = os.path.join(result_dir, f"{model_name}_loss_curve.png")
    plot_training_results(train_losses, val_losses, loss_plot_path)
    
    # Create combined model for visualization
    combined_model = TwoStageSpecBoost(model_0, model_1)
    
    # Return paths for visualization
    return {
        'model': combined_model,
        'model_0': model_0,
        'model_1': model_1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'checkpoint_dir': checkpoint_dir,
        'result_dir': result_dir,
        'best_checkpoint': save_path_best,
        'device': device,
        'dim': dim,
        'model_type': model_type,
        'dataset': dataset,
        'val_loader': val_loader,
    }


def main():
    parser = argparse.ArgumentParser(description='Train SpecBoost Model')
    parser.add_argument('--config', type=str, default='config/configure.yaml',
                        help='Path to config file')
    parser.add_argument('--dim', type=str, default=None,
                        choices=['1d', '2d'],
                        help='Dimension (overrides config)')
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 학습 실행
    result = train_specboost(config, override_dim=args.dim)
    
    # 시각화 실행
    if config['output'].get('plot_results', True):
        from visualization import save_top_k_results_specboost
        save_top_k_results_specboost(
            config=config,
            model_0=result['model_0'],
            model_1=result['model_1'],
            val_loader=result['val_loader'],
            dataset=result['dataset'],
            device=result['device'],
            dim=result['dim'],
            result_dir=result['result_dir']
        )


if __name__ == "__main__":
    main()
