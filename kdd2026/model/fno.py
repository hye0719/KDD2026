"""
Robust FNO (Fourier Neural Operator) Models
- RobustFNO: 1D version with Dropout
- RobustFNO2d: 2D version with Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

# MPS 관련 resize 경고 메시지 숨기기
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

# ==========================================
# 1D Spectral Convolution
# ==========================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        actual_modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :actual_modes] = self.compl_mul1d(
            x_ft[:, :, :actual_modes], self.weights[:, :, :actual_modes]
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


# ==========================================
# 1D Robust FNO (with Dropout)
# ==========================================
class RobustFNO(nn.Module):
    """
    1D Robust FNO with Dropout
    
    Input:  (B, C, L) where C = input_channels (default 3)
    Output: (B, 1, L)
    """
    def __init__(self, modes=12, width=32, input_channels=3, dropout_p=0.2):
        super().__init__()
        self.fc0 = nn.Linear(input_channels, width)

        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)

        # Dropout 추가 (과적합 방지)
        self.dropout = nn.Dropout(dropout_p)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, C, L)
        x = x.permute(0, 2, 1)  # (B, L, C)
        x = self.fc0(x)         # (B, L, width)
        x = x.permute(0, 2, 1)  # (B, width, L)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        x = self.dropout(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        x = self.dropout(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x = x.permute(0, 2, 1)  # (B, L, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)          # (B, L, 1)
        return x.permute(0, 2, 1)  # (B, 1, L)


# ==========================================
# 2D Spectral Convolution
# ==========================================
class SpectralConv2d(nn.Module):
    """
    x: (B, C_in, H, W)  ->  y: (B, C_out, H, W)
    rfft2 gives (B, C_in, H, W//2+1) complex
    We keep only low-frequency modes:
      - first modes1 along H
      - first modes2 along W_freq
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # input:  (B, C_in, m1, m2)
        # weights:(C_in, C_out, m1, m2)
        # output: (B, C_out, m1, m2)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)  # (B, C, H, W//2+1)

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes1, x_ft.shape[-2])
        m2 = min(self.modes2, x_ft.shape[-1])

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights[:, :, :m1, :m2]
        )

        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x


# ==========================================
# 2D Robust FNO (with Dropout)
# ==========================================
class RobustFNO2d(nn.Module):
    """
    2D Robust FNO with Dropout
    
    Input:  (B, H, W, in_dim) - permute 필요
    Output: (B, 1, H, W)
    """
    def __init__(self, in_dim=4, modes1=12, modes2=12, width=32, dropout_p=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.width = width

        # pointwise lift: (in_dim -> width)
        self.fc0 = nn.Linear(in_dim, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)

        # 1x1 conv residual branch (channel mixing)
        self.w0 = nn.Conv2d(width, width, kernel_size=1)
        self.w1 = nn.Conv2d(width, width, kernel_size=1)
        self.w2 = nn.Conv2d(width, width, kernel_size=1)

        self.dropout = nn.Dropout2d(dropout_p)

        # pointwise projection head
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, H, W, in_dim)
        B, H, W, C = x.shape
        assert C == self.in_dim, f"Expected in_dim={self.in_dim}, got {C}"

        # Lift
        x = self.fc0(x)                 # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)       # (B, width, H, W)

        # Block 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        x = self.dropout(x)

        # Block 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        x = self.dropout(x)

        # Block 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        # Head
        x = x.permute(0, 2, 3, 1)       # (B, H, W, width)
        x = self.fc1(x)                 # (B, H, W, 128)
        x = F.gelu(x)
        x = self.fc2(x)                 # (B, H, W, 1)
        x = x.permute(0, 3, 1, 2)       # (B, 1, H, W)

        return x


def get_fno_model(config, dim="1d"):
    """
    FNO 모델 생성 함수
    
    Args:
        config: 설정 딕셔너리
        dim: "1d" or "2d"
    
    Returns:
        model
    """
    fno_cfg = config['model']['fno']
    
    if dim == "1d":
        cfg = fno_cfg['1d']
        model = RobustFNO(
            modes=cfg['modes'],
            width=cfg['width'],
            input_channels=cfg.get('input_channels', 3)
        )
    else:  # 2d
        cfg = fno_cfg['2d']
        model = RobustFNO2d(
            in_dim=cfg['in_dim'],
            modes1=cfg['modes1'],
            modes2=cfg['modes2'],
            width=cfg['width'],
            dropout_p=cfg.get('dropout_p', 0.2)
        )
    
    return model


if __name__ == "__main__":
    # 테스트
    print("Testing RobustFNO 1D...")
    model_1d = RobustFNO(modes=12, width=32)
    x_1d = torch.randn(4, 3, 200)  # (B, C, L)
    y_1d = model_1d(x_1d)
    print(f"1D - Input: {x_1d.shape}, Output: {y_1d.shape}")
    
    print("\nTesting RobustFNO2d...")
    model_2d = RobustFNO2d(in_dim=4, modes1=12, modes2=6, width=48)
    x_2d = torch.randn(4, 200, 20, 4)  # (B, H, W, C)
    y_2d = model_2d(x_2d)
    print(f"2D - Input: {x_2d.shape}, Output: {y_2d.shape}")
