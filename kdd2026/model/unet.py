"""
UNet Model Module
- Unet1d: 1D version
- UNet2D: 2D version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1D UNet
# ==========================================
class Unet1d(nn.Module):
    """
    1D U-Net Architecture
    
    Input:  (B, C, L) where C = input_channels (default 3)
    Output: (B, 1, L)
    """
    def __init__(self, dim=16, input_channels=3, return_hs=False):
        super().__init__()
        self.return_hs = return_hs

        self.down_sample = nn.MaxPool1d(2)
        self.up_sample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.input_layer = nn.Sequential(
            nn.Conv1d(input_channels, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.ReLU()
        )

        self.down1 = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(dim * 2, dim * 2, 3, 1, 1),
            nn.ReLU()
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(dim * 2, dim * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(dim * 4, dim * 4, 3, 1, 1),
            nn.ReLU()
        )

        self.down3 = nn.Sequential(
            nn.Conv1d(dim * 4, dim * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(dim * 4, dim * 4, 3, 1, 1),
            nn.ReLU()
        )

        self.mid_blocks = nn.Sequential(
            nn.Conv1d(dim * 4, dim * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(dim * 4, dim * 4, 3, 1, 1),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.Conv1d(dim * 8, dim * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(dim * 2, dim * 2, 3, 1, 1),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.Conv1d(dim * 4, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.Conv1d(dim * 2, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.ReLU()
        )

        self.output_layer = nn.Conv1d(dim, 1, kernel_size=1)

    def match_size_concat(self, x1, x2):
        """Skip connection with size matching"""
        pad = x1.shape[-1] - x2.shape[-1]
        x2 = F.pad(x2, (0, pad))
        return torch.cat([x1, x2], dim=1)

    def forward(self, inp):
        hs = []

        if inp.dim() == 2:
            inp = inp[:, None, :]  # (B, 1, L)

        h1 = self.input_layer(inp)
        hs.append(h1)
        h2 = self.down1(self.down_sample(h1))
        hs.append(h2)
        h3 = self.down2(self.down_sample(h2))
        hs.append(h3)
        h4 = self.down3(self.down_sample(h3))
        hs.append(h4)

        mid = self.mid_blocks(h4)
        hs.append(mid)

        u_h3 = self.up1(self.match_size_concat(h3, self.up_sample(mid)))
        hs.append(u_h3)
        u_h2 = self.up2(self.match_size_concat(h2, self.up_sample(u_h3)))
        hs.append(u_h2)
        u_h1 = self.up3(self.match_size_concat(h1, self.up_sample(u_h2)))
        hs.append(u_h1)

        out = self.output_layer(u_h1)  # (B, 1, L)
        return (out, hs) if self.return_hs else out


# ==========================================
# 2D UNet Components
# ==========================================
class DoubleConv2D(nn.Module):
    """Double convolution block for UNet2D"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2D UNet
# ==========================================
class UNet2D(nn.Module):
    """
    2D U-Net Architecture
    
    Input:  (B, H, W, C) - 기존 Training loop 호환용
    Output: (B, 1, H, W)
    """
    def __init__(self, in_dim=4, base_dim=48, return_hs=False):
        super().__init__()
        self.return_hs = return_hs
        self.in_dim = in_dim

        self.enc0 = DoubleConv2D(in_dim, base_dim)
        self.pool0 = nn.MaxPool2d(2)

        self.enc1 = DoubleConv2D(base_dim, base_dim * 2)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv2D(base_dim * 2, base_dim * 4)
        self.pool2 = nn.MaxPool2d(2)

        # bottleneck
        self.mid = DoubleConv2D(base_dim * 4, base_dim * 4)

        # decoder
        self.dec2 = DoubleConv2D(base_dim * 4 + base_dim * 4, base_dim * 2)
        self.dec1 = DoubleConv2D(base_dim * 2 + base_dim * 2, base_dim)
        self.dec0 = DoubleConv2D(base_dim + base_dim, base_dim)

        self.out = nn.Conv2d(base_dim, 1, kernel_size=1)

    def forward(self, x_bhwc):
        hs = []

        # (B, H, W, C) -> (B, C, H, W)
        x = x_bhwc.permute(0, 3, 1, 2).contiguous()

        e0 = self.enc0(x)
        hs.append(e0)
        p0 = self.pool0(e0)

        e1 = self.enc1(p0)
        hs.append(e1)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        hs.append(e2)
        p2 = self.pool2(e2)

        m = self.mid(p2)
        hs.append(m)

        # 업샘플은 skip과 동일 size로 맞춤 (concat 에러 방지)
        u2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([e2, u2], dim=1))
        hs.append(d2)

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([e1, u1], dim=1))
        hs.append(d1)

        u0 = F.interpolate(d1, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.dec0(torch.cat([e0, u0], dim=1))
        hs.append(d0)

        out = self.out(d0)  # (B, 1, H, W)
        return (out, hs) if self.return_hs else out


def get_unet_model(config, dim="1d"):
    """
    UNet 모델 생성 함수
    
    Args:
        config: 설정 딕셔너리
        dim: "1d" or "2d"
    
    Returns:
        model
    """
    unet_cfg = config['model']['unet']
    
    if dim == "1d":
        cfg = unet_cfg['1d']
        model = Unet1d(
            dim=cfg['dim'],
            input_channels=cfg.get('input_channels', 3)
        )
    else:  # 2d
        cfg = unet_cfg['2d']
        model = UNet2D(
            in_dim=cfg['in_dim'],
            base_dim=cfg['base_dim']
        )
    
    return model


if __name__ == "__main__":
    # 테스트
    print("Testing Unet1d...")
    model_1d = Unet1d(dim=16, input_channels=3)
    x_1d = torch.randn(4, 3, 200)  # (B, C, L)
    y_1d = model_1d(x_1d)
    print(f"1D - Input: {x_1d.shape}, Output: {y_1d.shape}")

    print("\nTesting UNet2D...")
    model_2d = UNet2D(in_dim=4, base_dim=48)
    x_2d = torch.randn(4, 200, 20, 4)  # (B, H, W, C)
    y_2d = model_2d(x_2d)
    print(f"2D - Input: {x_2d.shape}, Output: {y_2d.shape}")
