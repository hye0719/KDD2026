import torch
from torch.utils.data import Dataset, DataLoader

from neuralop.models import FNO
from neuralop.training import Trainer
from neuralop.losses import LpLoss

"""
Metalens Dataset Module
- MiniMetalensDataset: 1D version
- MiniMetalensDataset2D: 2D version
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MiniMetalensDataset(Dataset):
    """1D Metalens Dataset"""
    
    def __init__(self, search_path, scale_factor=100.0, Ny=20, 
                 max_stat_samples=500, max_samples=None):
        """
        Args:
            search_path: 데이터 경로
            scale_factor: 스케일링 팩터
            Ny: Y 차원 크기
            max_stat_samples: 통계 계산용 최대 샘플 수
            max_samples: 사용할 최대 데이터 수 (None이면 전체)
        """
        self.file_list = sorted(glob.glob(os.path.join(search_path, "*.npz")))
        self.scale_factor = scale_factor
        self.Ny = Ny

        if len(self.file_list) == 0:
            print(f"⚠️ No files found in {search_path}")
        
        # 데이터 수 제한
        if max_samples is not None and max_samples < len(self.file_list):
            self.file_list = self.file_list[:max_samples]
            print(f"[INFO] Limited to {max_samples} samples.")

        print(f"[INFO] Found {len(self.file_list)} samples.")

        # 통계 계산
        print("[INFO] Calculating statistics...")
        all_targets = []
        sample_count = min(len(self.file_list), max_stat_samples)

        for f in self.file_list[:sample_count]:
            try:
                d = np.load(f)
                grad = np.mean(d['adjoint_gradient'].reshape(-1, self.Ny), axis=1)
                all_targets.append(grad * self.scale_factor)
            except:
                pass

        if len(all_targets) > 0:
            all_targets = np.concatenate(all_targets)
            self.mean = float(np.mean(all_targets))
            self.std = float(np.std(all_targets) + 1e-8)
        else:
            self.mean, self.std = 0.0, 1.0
            print("⚠️ Stats failed (no valid samples). Using Mean=0, Std=1")
        
        print(f"✅ Stats: Mean={self.mean:.4f}, Std={self.std:.4f}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            d = np.load(self.file_list[idx])
            geom_1d = d['geometry'].reshape(-1, self.Ny)[:, 0]
            grad_1d = np.mean(d['adjoint_gradient'].reshape(-1, self.Ny), axis=1)

            Nx_curr = geom_1d.shape[0]
            grid = np.linspace(-1, 1, Nx_curr, dtype=np.float32)

            edge_map = np.gradient(geom_1d)
            edge_map = np.abs(edge_map) / (np.max(np.abs(edge_map)) + 1e-8)

            input_stack = np.stack([geom_1d, grid, edge_map], axis=0).astype(np.float32)
            grad_scaled = (grad_1d * self.scale_factor - self.mean) / self.std

            return (
                torch.tensor(input_stack, dtype=torch.float32),
                torch.tensor(grad_scaled, dtype=torch.float32).unsqueeze(0)
            )
        except:
            return self.__getitem__(0)

    def denormalize(self, tensor):
        """역정규화"""
        return (tensor * self.std + self.mean) / self.scale_factor


class MiniMetalensDataset2D(Dataset):
    """2D Metalens Dataset"""
    
    def __init__(self, search_path, Ny=20, scale_factor=100.0, 
                 max_stat_samples=500, max_samples=None):
        """
        Args:
            search_path: 데이터 경로
            Ny: Y 차원 크기
            scale_factor: 스케일링 팩터
            max_stat_samples: 통계 계산용 최대 샘플 수
            max_samples: 사용할 최대 데이터 수 (None이면 전체)
        """
        self.file_list = sorted(glob.glob(os.path.join(search_path, "*.npz")))
        self.scale_factor = scale_factor
        self.Ny = Ny

        if len(self.file_list) == 0:
            print(f"⚠️ No files found in {search_path}")
        
        # 데이터 수 제한
        if max_samples is not None and max_samples < len(self.file_list):
            self.file_list = self.file_list[:max_samples]
            print(f"[INFO] Limited to {max_samples} samples.")
            
        print(f"[INFO] Found {len(self.file_list)} samples.")

        # 통계 계산 (2D 타깃 전체 픽셀 기준)
        print("[INFO] Calculating statistics...")
        all_targets = []
        sample_count = min(len(self.file_list), max_stat_samples)

        for f in self.file_list[:sample_count]:
            try:
                d = np.load(f)
                grad_2d = d["adjoint_gradient"].reshape(-1, self.Ny).astype(np.float32)
                all_targets.append((grad_2d * self.scale_factor).reshape(-1))
            except:
                pass

        if len(all_targets) == 0:
            self.mean, self.std = 0.0, 1.0
            print("⚠️ Stats failed (no valid samples). Using Mean=0, Std=1")
        else:
            all_targets = np.concatenate(all_targets, axis=0)
            self.mean = float(np.mean(all_targets))
            self.std = float(np.std(all_targets) + 1e-8)
            print(f"✅ Stats: Mean={self.mean:.4f}, Std={self.std:.4f}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            d = np.load(self.file_list[idx])

            # (Nx, Ny)
            geom_2d = d["geometry"].reshape(-1, self.Ny).astype(np.float32)
            grad_2d = d["adjoint_gradient"].reshape(-1, self.Ny).astype(np.float32)

            Nx, Ny = geom_2d.shape

            # grid: x,y 각각 [-1,1]
            x = np.linspace(-1, 1, Nx, dtype=np.float32)
            y = np.linspace(-1, 1, Ny, dtype=np.float32)
            X, Y = np.meshgrid(y, x)  # X: (Nx, Ny), Y: (Nx, Ny)

            # edge map (간단히 2D gradient magnitude)
            gx, gy = np.gradient(geom_2d)
            edge_map = np.sqrt(gx**2 + gy**2)
            edge_map = edge_map / (np.max(edge_map) + 1e-8)

            # input: (C, H, W) = (4, Nx, Ny)
            input_stack = np.stack([geom_2d, Y, X, edge_map], axis=0)

            # target: (1, Nx, Ny)
            grad_scaled = (grad_2d * self.scale_factor - self.mean) / self.std
            target = grad_scaled[None, :, :]

            return (
                torch.tensor(input_stack, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32)
            )

        except:
            return self.__getitem__(0)

    def denormalize(self, tensor):
        """역정규화"""
        return (tensor * self.std + self.mean) / self.scale_factor


def get_dataloaders(config, dim="1d"):
    """
    데이터로더 생성 함수
    
    Args:
        config: 설정 딕셔너리
        dim: "1d" or "2d"
    
    Returns:
        dataset, train_loader, val_loader
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    # max_samples 처리
    max_samples = data_cfg.get('max_samples', None)
    
    # 데이터셋 생성
    if dim == "1d":
        dataset = MiniMetalensDataset(
            search_path=data_cfg['path'],
            scale_factor=data_cfg['scale_factor'],
            Ny=data_cfg['Ny'],
            max_stat_samples=data_cfg.get('max_stat_samples', 500),
            max_samples=max_samples
        )
    else:  # 2d
        dataset = MiniMetalensDataset2D(
            search_path=data_cfg['path'],
            Ny=data_cfg['Ny'],
            scale_factor=data_cfg['scale_factor'],
            max_stat_samples=data_cfg.get('max_stat_samples', 500),
            max_samples=max_samples
        )
    
    # Train/Val 분할
    train_size = int(data_cfg['train_ratio'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 데이터로더 생성
    num_workers = config['general'].get('num_workers', 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=num_workers
    )
    
    return dataset, train_loader, val_loader

import yaml

with open("config/configure.yaml", "r") as f:
    config = yaml.safe_load(f)

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

    # 1D 테스트
    print("=" * 50)
    print("Testing 1D Dataset")
    print("=" * 50)
    dataset_1d, train_loader_1d, val_loader_1d = get_dataloaders(config, dim="1d")
    
    class DictDataset(Dataset):
        def __init__(self, xs, ys):
            self.xs = xs  # (N,2,H,W)
            self.ys = ys  # (N,1,H,W)

        def __len__(self):
            return self.xs.shape[0]

        def __getitem__(self, idx):
            return {"x": self.xs[idx], "y": self.ys[idx]}  # ★ dict로 반환

    # 1) model
    model = FNO(
        n_modes=(16, 16),
        hidden_channels=64,
        in_channels=2,
        out_channels=1
    )

    # 2) loader
    # train_loader = DataLoader(DictDataset(xs_train, ys_train), batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader   = DataLoader(DictDataset(xs_val, ys_val),   batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # 3) loss / opt / sched
    training_loss = LpLoss(d=1, p=2)   # 2D 필드라 d=2 (1D면 d=1)
    eval_losses = {"l2": LpLoss(d=1, p=2)}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 4) trainer
    trainer = Trainer(
        model=model,
        n_epochs=100,
        device="mps",
        mixed_precision=False,
        eval_interval=1,
        verbose=True,
    )

    # 5) train
    #test_loaders는 dict로 전달
    trainer.train(
        train_loader=train_loader_1d,
        test_loaders={"val": val_loader_1d},
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_dir="./ckpt",
        save_best="val_l2",  # "loaderName_lossName" 형태로 모니터링
    )

if __name__ == '__main__':
    run()