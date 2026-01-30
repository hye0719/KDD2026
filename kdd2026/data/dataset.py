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
            grad_1d = np.sum(d['adjoint_gradient'].reshape(-1, self.Ny), axis=1)

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

# #양 끝 잘라냄
#     def __getitem__(self, idx):
#         try:
#             d = np.load(self.file_list[idx])

#             geom = d['geometry'].reshape(-1, self.Ny)
#             grad = d['adjoint_gradient'].reshape(-1, self.Ny)

#             geom_1d = geom[:, 0]
#             grad_1d = np.mean(grad, axis=1)

#             Nx_curr = geom_1d.shape[0]
#             target_Nx = 50

#             # ===============================
#             # 🔥 Center crop if needed
#             # ===============================
#             if Nx_curr > target_Nx:
#                 start = (Nx_curr - target_Nx) // 2
#                 end = start + target_Nx
#                 geom_1d = geom_1d[start:end]
#                 grad_1d = grad_1d[start:end]
#                 Nx_curr = target_Nx

#             # (선택) 혹시 더 짧은 경우 예외 처리
#             elif Nx_curr < target_Nx:
#                 raise ValueError(f"Nx={Nx_curr} < target_Nx={target_Nx}")

#             # ===============================
#             # Grid & Edge map
#             # ===============================
#             grid = np.linspace(-1, 1, Nx_curr, dtype=np.float32)

#             edge_map = np.gradient(geom_1d)
#             edge_map = np.abs(edge_map) / (np.max(np.abs(edge_map)) + 1e-8)

#             input_stack = np.stack(
#                 [geom_1d, grid, edge_map], axis=0
#             ).astype(np.float32)

#             grad_scaled = (grad_1d * self.scale_factor - self.mean) / self.std

#             return (
#                 torch.tensor(input_stack, dtype=torch.float32),
#                 torch.tensor(grad_scaled, dtype=torch.float32).unsqueeze(0)
#             )

#         except Exception as e:
#             print(f"[Dataset Error] {e}")
#             return self.__getitem__(0)



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
    
    # # Train/Val 분할
    train_size = int(data_cfg['train_ratio'] * len(dataset))
    val_size = len(dataset) - train_size

    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, val_size]
    # )
    
    # Train/Val 분할 (시드 고정)
    seed = config['general'].get('seed', 42)
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=g
    )


    # 데이터로더 생성
    num_workers = config['general'].get('num_workers', 2)
    if data_cfg['train_ratio']==0:
        train_loader = None
    else:  
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=True,
            num_workers=num_workers
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )
    
    return dataset, train_loader, val_loader


if __name__ == "__main__":
    # 테스트
    import yaml
    
    with open("../config/configure.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 1D 테스트
    print("=" * 50)
    print("Testing 1D Dataset")
    print("=" * 50)
    dataset_1d, train_loader_1d, val_loader_1d = get_dataloaders(config, dim="1d")
    x, y = next(iter(train_loader_1d))
    print(f"1D - x: {x.shape}, y: {y.shape}")
    
    for i, batch in enumerate(val_loader_1d):
        x, y = batch
        print(x)
        print(x.shape)
        print(y.shape)
        exit()
    

    # # 2D 테스트
    # print("\n" + "=" * 50)
    # print("Testing 2D Dataset")
    # print("=" * 50)
    # dataset_2d, train_loader_2d, val_loader_2d = get_dataloaders(config, dim="2d")
    # x, y = next(iter(train_loader_2d))
    # print(f"2D - x: {x.shape}, y: {y.shape}")

    # import numpy as np

    # data = np.load("/home/work/Dataset/test/000001.npz")
    # print(data.files)
    # import numpy as np
    # import glob
    # import hashlib
    # from tqdm import tqdm

    # files = sorted(glob.glob("/home/work/KDD/data/test2/*.npz"))

    # hash_map = {}      # hash -> filename
    # duplicates = []   # (file, duplicate_of)
    # corrupted = []
    # for f in tqdm(files):
    #     try:
    #         d = np.load(f)
    #         geom = d["geometry"].astype(np.float32)

    #         h = hashlib.md5(geom.tobytes()).hexdigest()

    #         if h in hash_map:
    #             duplicates.append((f, hash_map[h]))
    #         else:
    #             hash_map[h] = f

    #     except Exception as e:
    #         corrupted.append((f, str(e)))
    #         continue

    # print(f"\n총 파일 수: {len(files)}")
    # print(f"중복 구조 파일 수: {len(duplicates)}")
    # print(f"깨진 파일 수: {len(corrupted)}")

    # # 예시 출력
    # for a, b in duplicates[:10]:
    #     print(f"❌ {a} == {b}")
