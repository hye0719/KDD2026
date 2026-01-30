"""
Training Script for FNO, Linear, UNet Models
Supports both 1D and 2D configurations

Usage:
    python train.py --config config/configure.yaml
    python train.py --config config/configure.yaml --model fno --dim 2d
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.dataset import get_dataloaders
from model import get_model
from utils import (
    get_device, 
    create_experiment_dir, 
    save_config, 
    set_seed,
    print_model_info
)


import torch.nn.functional as F

class SignComboLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.p = cfg.get("percentile", 98) / 100.0
        self.topk = cfg.get("topk_percent", 10) / 100.0
        self.alpha = cfg.get("alpha", 10.0)

        self.l_sign = cfg.get("lambda_sign", 1.0)
        self.l_cos  = cfg.get("lambda_cos", 0.5)
        self.l_mag  = cfg.get("lambda_mag", 0.2)

        self.mag_type = cfg.get("mag_type", "huber")
        self.beta = cfg.get("beta_critical", 10.0)
        self.tv_w = cfg.get("tv_weight", 0.0)

        self.edge_drop = int(cfg.get("edge_drop", 0))
        self.edge_weight = float(cfg.get("edge_weight", 1.0))
        self.eps = float(cfg.get("eps", 1e-6))

    def _edge_weights_1d(self, L, device):
        w = torch.ones(L, device=device)
        if self.edge_drop > 0:
            d = min(self.edge_drop, L // 2)
            w[:d] *= self.edge_weight
            w[-d:] *= self.edge_weight
        return w.view(1, 1, L)  # (1,1,L)

    def forward(self, pred, y):
        """
        pred, y: (B, 1, L)  (또는 y가 (B,L)이면 아래에서 reshape됨)
        """
        if y.dim() == 2:
            y = y.unsqueeze(1)
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)

        B, C, L = y.shape
        device = y.device

        # --- robust scale: s = percentile(|g|, p) per-sample ---
        abs_y = y.abs()
        # torch.quantile은 (B,*)에서 per-sample 처리하려면 flatten 필요
        abs_flat = abs_y.view(B, -1)
        s = torch.quantile(abs_flat, self.p, dim=1, keepdim=True)  # (B,1)
        s = s.view(B, 1, 1)  # (B,1,1)

        y_n = y / (s + self.eps)
        pred_n = pred / (s + self.eps)

        # --- critical mask: top-k% by |y| (GT 기준) ---
        # threshold = quantile(|y|, 1-topk)
        thr = torch.quantile(abs_flat, 1.0 - self.topk, dim=1, keepdim=True).view(B, 1, 1)
        M = (abs_y >= thr).float()  # (B,1,L)

        # --- edge down-weight (patch 경계 노이즈) ---
        w_edge = self._edge_weights_1d(L, device)  # (1,1,L)

        # weight: critical 크게, edge는 낮게
        w = (1.0 + self.beta * M) * w_edge  # (B,1,L) broadcast

        # (A) sign loss (mask에서 강하게)
        # BCEWithLogits expects logits, target in {0,1}
        target = (y_n > 0).float()
        logits = self.alpha * pred_n
        sign_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        sign_loss = (sign_loss * (M * w_edge)).sum() / ((M * w_edge).sum() + self.eps)

        # (B) cosine directional loss (mask 영역)
        a = (pred_n * M * w_edge).view(B, -1)
        b = (y_n    * M * w_edge).view(B, -1)
        dot = (a * b).sum(dim=1)
        na = torch.sqrt((a * a).sum(dim=1) + self.eps)
        nb = torch.sqrt((b * b).sum(dim=1) + self.eps)
        cos_loss = (1.0 - dot / (na * nb + self.eps)).mean()

        # (C) magnitude loss (robust)
        if self.mag_type.lower() == "huber":
            mag_elem = F.smooth_l1_loss(pred_n, y_n, reduction="none")
        else:
            # Charbonnier: sqrt((x)^2 + eps^2)
            mag_elem = torch.sqrt((pred_n - y_n) ** 2 + (1e-3) ** 2)

        mag_loss = (mag_elem * w).sum() / (w.sum() + self.eps)

        # (D) optional TV
        tv_loss = torch.tensor(0.0, device=device)
        if self.tv_w > 0:
            tv_loss = (pred_n[..., 1:] - pred_n[..., :-1]).abs().mean()

        total = self.l_sign * sign_loss + self.l_cos * cos_loss + self.l_mag * mag_loss + self.tv_w * tv_loss

        # 로깅용
        return total, {
            "sign": sign_loss.detach(),
            "cos": cos_loss.detach(),
            "mag": mag_loss.detach(),
            "tv": tv_loss.detach()
        }

def get_criterion(loss_type, config=None):
    """손실 함수 생성"""
    if loss_type.lower() == "mse":
        return nn.MSELoss()
    elif loss_type.lower() == "l1":
        return nn.L1Loss()
    elif loss_type.lower() == "sign_combo":
        return SignComboLoss(config["training"]["sign_loss"])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, dim):
    """한 에폭 학습"""
    model.train()
    total_loss = 0.0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        # 2D 모델의 경우 permute 필요 (FNO2d, UNet2D는 (B, H, W, C) 입력)
        if dim == "2d":
            x_in = x.permute(0, 2, 3, 1).contiguous()  # (B, C, H, W) -> (B, H, W, C)
        else:
            x_in = x
        
        optimizer.zero_grad()
        pred = model(x_in)
        out = criterion(pred, y)
        if isinstance(out, tuple):
            loss, parts = out
        else:
            loss = out

        #loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += float(loss.detach().cpu().item())
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, dim):
    """검증"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            if dim == "2d":
                x_in = x.permute(0, 2, 3, 1).contiguous()
            else:
                x_in = x
            
            pred = model(x_in)
            out = criterion(pred, y)
            if isinstance(out, tuple):
                loss, parts = out
            else:
                loss = out
            total_loss += float(loss.detach().cpu().item())
    
    return total_loss / len(val_loader)


def plot_training_results(train_losses, val_losses, save_path):
    """학습 곡선 저장"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Loss curve saved to {save_path}")


def train(config, override_model=None, override_dim=None):
    """
    메인 학습 함수
    
    Args:
        config: 설정 딕셔너리
        override_model: 모델 타입 오버라이드 (옵션)
        override_dim: 차원 오버라이드 (옵션)
    """
    # 설정 추출
    general_cfg = config['general']
    train_cfg = config['training']
    output_cfg = config['output']
    
    # 오버라이드 적용
    model_type = override_model if override_model else config['model']['type']
    dim = override_dim if override_dim else config['model']['dim']
    
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
    
    # 모델 생성
    original_model_type = config['model']['type']
    config['model']['type'] = model_type
    model = get_model(config, dim=dim)
    config['model']['type'] = original_model_type
    
    model = model.to(device)
    
    # 모델 정보 출력
    print_model_info(model, model_type, dim)
    
    # 옵티마이저 & 스케줄러
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg['learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=train_cfg['epochs']
    )
    
    # 손실 함수
    #criterion = get_criterion(train_cfg['loss']).to(device)
    criterion = get_criterion(train_cfg['loss'], config).to(device)


    # 학습 루프
    print(f"\n🚀 Start Training ({model_type.upper()} {dim.upper()})...")
    train_losses, val_losses = [], []
    best_val = float('inf')
    
    model_name = f"{model_type}_{dim}"
    save_path_best = os.path.join(checkpoint_dir, f"{model_name}{output_cfg['best_model_suffix']}")
    save_path_final = os.path.join(checkpoint_dir, f"{model_name}{output_cfg['final_model_suffix']}")
    
    for epoch in range(train_cfg['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, dim)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, dim)
        val_losses.append(val_loss)
        
        # Log
        if (epoch + 1) % train_cfg['log_interval'] == 0:
            print(f"Epoch {epoch + 1:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val and epoch > train_cfg.get('min_epoch_for_save', 10) and epoch%10==0:
            best_val = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val,
                'config': config,
            }, save_path_best)
            print(f"✅ Best model saved at epoch {epoch + 1} (Val: {best_val:.6f})")
    
    # Save final model
    torch.save({
        'epoch': train_cfg['epochs'],
        'model_state_dict': model.state_dict(),
        'val_loss': val_losses[-1],
        'config': config,
    }, save_path_final)
    print(f"💾 Final model saved to {save_path_final}")
    
    # Plot training curves
    loss_plot_path = os.path.join(result_dir, f"{model_name}_loss_curve.png")
    plot_training_results(train_losses, val_losses, loss_plot_path)
    
    # Return paths for visualization
    return {
        'model': model,
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
    parser = argparse.ArgumentParser(description='Train Metalens Models')
    parser.add_argument('--config', type=str, default='config/configure.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        choices=['fno', 'linear', 'unet'],
                        help='Model type (overrides config)')
    parser.add_argument('--dim', type=str, default=None,
                        choices=['1d', '2d'],
                        help='Dimension (overrides config)')
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 학습 실행
    result = train(config, override_model=args.model, override_dim=args.dim)
    
    # 시각화 실행
    if config['output'].get('plot_results', True):
        from visualization import save_top_k_results
        save_top_k_results(
            config=config,
            model=result['model'],
            val_loader=result['val_loader'],
            dataset=result['dataset'],
            device=result['device'],
            dim=result['dim'],
            model_type=result['model_type'],
            result_dir=result['result_dir']
        )


if __name__ == "__main__":
    main()
