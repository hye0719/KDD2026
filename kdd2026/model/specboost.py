"""
SpecBoost FNO Models (Dropout 제거 & Dim 복구)
- NewFNO: 1D version
- NewFNO2D: 2D version

Two-stage boosting architecture for residual learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1D NewFourierBlock
# ==========================================
class NewFourierBlock(nn.Module):
    def __init__(self, indim, outdim, mode, bias=False):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        self.mode = mode
        self.scale = (1 / (indim * outdim))
        self.weights = nn.Parameter(
            self.scale * torch.rand(indim, outdim, mode, dtype=torch.cfloat)
        )
        self.bias = bias
        if self.bias:
            self.bias_param = nn.Parameter(
                torch.zeros(1, outdim, mode, dtype=torch.cfloat)
            )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bim,iom->bom", input, weights)

    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)
        x_ft_ = self.compl_mul1d(x_ft[:, :, :self.mode], self.weights)
        if self.bias:
            x_ft_ = x_ft_ + self.bias_param
        out_ft[:, :, :self.mode] = x_ft_
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


# ==========================================
# 1D NewFNO (Dropout 제거, Dim 기본값 복구)
# ==========================================
class NewFNO(nn.Module):
    """
    1D NewFNO for SpecBoost
    
    Input:  (B, indim, L)
    Output: (B, 1, L)
    """
    def __init__(self, indim=3, dim=64, mode=32, layer_num=4):
        super().__init__()
        self.stem = nn.Linear(indim, dim)
        self.fnos = nn.ModuleList([])
        self.projs = nn.ModuleList([])

        for _ in range(layer_num):
            self.fnos.append(NewFourierBlock(dim, dim, mode))
            self.projs.append(nn.Linear(dim, dim))

        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, C, L)
        x = x.permute(0, 2, 1)  # (B, L, C)
        x = self.stem(x)        # (B, L, dim)
        x = x.permute(0, 2, 1)  # (B, dim, L)

        for fno, proj in zip(self.fnos, self.projs):
            x1 = fno(x)
            x2 = proj(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = F.silu(x1 + x2)  # Activation만 남김

        x = x.permute(0, 2, 1)  # (B, L, dim)
        x = self.head(x)        # (B, L, 1)
        x = x.permute(0, 2, 1)  # (B, 1, L)
        return x


# ==========================================
# 2D NewFourierBlock
# ==========================================
class NewFourierBlock2D(nn.Module):
    def __init__(self, indim, outdim, modes1, modes2, bias=False):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (indim * outdim)
        self.weights = nn.Parameter(
            self.scale * torch.rand(indim, outdim, modes1, modes2, dtype=torch.cfloat)
        )
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(
                torch.zeros(1, outdim, modes1, modes2, dtype=torch.cfloat)
            )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # x: (B, C, H, W)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # (B, C, H, W//2+1)
        B, C, H, Wc = x_ft.shape

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, Wc)

        out_ft = torch.zeros_like(x_ft)
        out_low = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2])

        if self.bias:
            out_low = out_low + self.bias_param[:, :, :m1, :m2]

        out_ft[:, :, :m1, :m2] = out_low
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1))
        return x


# ==========================================
# 2D NewFNO
# ==========================================
class NewFNO2D(nn.Module):
    """
    2D NewFNO for SpecBoost
    
    Input:  (B, indim, H, W)
    Output: (B, 1, H, W)
    """
    def __init__(self, indim=4, dim=64, modes1=12, modes2=6, layer_num=4):
        super().__init__()
        self.stem = nn.Conv2d(indim, dim, kernel_size=1)

        self.fnos = nn.ModuleList([
            NewFourierBlock2D(dim, dim, modes1, modes2) for _ in range(layer_num)
        ])
        self.projs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1) for _ in range(layer_num)
        ])

        self.head = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, indim, H, W)
        x = self.stem(x)  # (B, dim, H, W)
        for fno, proj in zip(self.fnos, self.projs):
            x = F.silu(fno(x) + proj(x))
        return self.head(x)  # (B, 1, H, W)


# ==========================================
# Two-Stage SpecBoost Wrapper
# ==========================================
class TwoStageSpecBoost(nn.Module):
    """
    Two-stage SpecBoost model wrapper
    
    Stage 0: Base FNO (predicts initial output)
    Stage 1: Residual FNO (predicts residual)
    Final output = Stage 0 output + Stage 1 output
    """
    def __init__(self, model_0, model_1):
        super().__init__()
        self.model_0 = model_0
        self.model_1 = model_1
    
    def forward(self, x):
        # Stage 0 prediction
        pred_0 = self.model_0(x)
        
        # Concatenate input with Stage 0 output for Stage 1
        input_1 = torch.cat([x, pred_0], dim=1)
        
        # Stage 1 (residual) prediction
        pred_residual = self.model_1(input_1)
        
        # Final output
        return pred_0 + pred_residual
    
    def get_stage0_pred(self, x):
        """Stage 0 예측만 반환"""
        return self.model_0(x)


def get_specboost_models(config, dim="2d"):
    """
    SpecBoost 모델들 생성 함수
    
    Args:
        config: 설정 딕셔너리
        dim: "1d" or "2d"
    
    Returns:
        model_0, model_1 (두 개의 모델)
    """
    sb_cfg = config['model']['specboost']
    
    if dim == "1d":
        cfg = sb_cfg['1d']
        model_0 = NewFNO(
            indim=cfg['indim'],
            dim=cfg['dim'],
            mode=cfg['mode'],
            layer_num=cfg['layer_num']
        )
        model_1 = NewFNO(
            indim=cfg['indim_stage1'],
            dim=cfg['dim'],
            mode=cfg['mode'],
            layer_num=cfg['layer_num']
        )
    else:  # 2d
        cfg = sb_cfg['2d']
        model_0 = NewFNO2D(
            indim=cfg['indim'],
            dim=cfg['dim'],
            modes1=cfg['modes1'],
            modes2=cfg['modes2'],
            layer_num=cfg['layer_num']
        )
        model_1 = NewFNO2D(
            indim=cfg['indim_stage1'],
            dim=cfg['dim'],
            modes1=cfg['modes1'],
            modes2=cfg['modes2'],
            layer_num=cfg['layer_num']
        )
    
    return model_0, model_1


if __name__ == "__main__":
    # 테스트
    print("Testing NewFNO 1D...")
    model_1d = NewFNO(indim=3, dim=64, mode=32, layer_num=4)
    x_1d = torch.randn(4, 3, 200)  # (B, C, L)
    y_1d = model_1d(x_1d)
    print(f"1D - Input: {x_1d.shape}, Output: {y_1d.shape}")
    
    print("\nTesting NewFNO2D...")
    model_2d = NewFNO2D(indim=4, dim=64, modes1=12, modes2=6, layer_num=4)
    x_2d = torch.randn(4, 4, 200, 20)  # (B, C, H, W)
    y_2d = model_2d(x_2d)
    print(f"2D - Input: {x_2d.shape}, Output: {y_2d.shape}")
    
    print("\nTesting TwoStageSpecBoost 2D...")
    model_0 = NewFNO2D(indim=4, dim=64, modes1=12, modes2=6, layer_num=4)
    model_1 = NewFNO2D(indim=5, dim=64, modes1=12, modes2=6, layer_num=4)
    two_stage = TwoStageSpecBoost(model_0, model_1)
    y_final = two_stage(x_2d)
    print(f"TwoStage - Input: {x_2d.shape}, Output: {y_final.shape}")
