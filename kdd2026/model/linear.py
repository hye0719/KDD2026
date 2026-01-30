"""
Linear Model Module
- LinearModel: Simple MLP model
- ConvLinearModel: CNN-based linear model (1D)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_embedding(timesteps: torch.Tensor, dim, max_period=10000):
    r"""Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.
    Returns:
        embedding (torch.Tensor): [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LinearModel(nn.Module):
    """
    Simple MLP Model
    
    Input:  (B, C, L) or (B, L) - will be flattened
    Output: (B, input_dim) - same as flattened input
    """
    def __init__(self, input_dim, dim=32, layer_num=4, p=0.6,
                 residual=False, bn=True, neural_rep=False,
                 fourier_embed=False, return_hs=False):
        super().__init__()
        self.residual = residual
        self.dim = dim
        self.neural_rep = neural_rep
        self.fourier_embed = fourier_embed
        self.return_hs = return_hs
        self.input_dim = input_dim

        if self.neural_rep:
            self.neural = nn.Parameter(torch.randn(1, input_dim), requires_grad=True)

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU()
        ))

        for _ in range(layer_num - 2):
            if bn:
                self.layers.append(nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(p)
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Dropout(p)
                ))

        self.layers.append(nn.Sequential(
            nn.Linear(dim, input_dim)
        ))

    def forward(self, inp):
        # 입력 형태 처리
        original_shape = inp.shape
        if inp.dim() == 3:
            inp = inp.flatten(1)  # (B, C*L)

        if self.neural_rep:
            inp = inp + self.neural

        hs = []
        for i, layer in enumerate(self.layers):
            if self.residual and (0 < i < len(self.layers) - 1):
                inp = layer(inp) + inp
            else:
                inp = layer(inp)
            hs.append(inp)

        return (inp, hs) if self.return_hs else inp


class ConvLinearModel(nn.Module):
    """
    CNN-based Linear Model (assumes locality)
    
    Input:  (B, L) - 1D signal
    Output: (B, L)
    """
    def __init__(self, input_dim, dim=8, layer_num=4, p=0.6,
                 residual=False, bn=True, neural_rep=False,
                 return_hs=False):
        super().__init__()
        self.residual = residual
        self.neural_rep = neural_rep
        self.return_hs = return_hs
        self.input_dim = input_dim

        if self.neural_rep:
            self.neural = nn.Parameter(torch.randn(1, input_dim), requires_grad=True)

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Sequential(
            nn.Conv1d(1, dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        ))

        for _ in range(layer_num - 2):
            if bn:
                self.layers.append(nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(dim),
                    nn.SiLU(),
                    nn.Dropout(p)
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Dropout(p)
                ))

        self.last_layer = nn.Linear(dim * input_dim, input_dim)

    def forward(self, inp):
        if self.neural_rep:
            inp = inp + self.neural

        hs = []
        BS, L = inp.shape
        x = inp.view(BS, 1, L)

        for i, layer in enumerate(self.layers):
            if self.residual and i > 0:
                x = layer(x) + x
            else:
                x = layer(x)
            hs.append(x)

        out = self.last_layer(x.view(BS, -1))
        return (out, hs) if self.return_hs else out


def get_linear_model(config):
    """
    Linear 모델 생성 함수
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        model
    """
    lin_cfg = config['model']['linear']
    
    model = LinearModel(
        input_dim=lin_cfg['input_dim'],
        dim=lin_cfg['dim'],
        layer_num=lin_cfg['layer_num'],
        p=lin_cfg.get('dropout', 0.6),
        residual=lin_cfg.get('residual', False),
        bn=lin_cfg.get('bn', True)
    )
    
    return model


if __name__ == "__main__":
    # 테스트
    print("Testing LinearModel...")
    model = LinearModel(input_dim=150, dim=32, layer_num=4)
    x = torch.randn(4, 150)  # (B, L)
    y = model(x)
    print(f"LinearModel - Input: {x.shape}, Output: {y.shape}")
    
    print("\nTesting with 3D input...")
    x_3d = torch.randn(4, 3, 50)  # (B, C, L) -> will be flattened to (B, 150)
    model_3d = LinearModel(input_dim=150, dim=32, layer_num=4)
    y_3d = model_3d(x_3d)
    print(f"LinearModel 3D - Input: {x_3d.shape}, Output: {y_3d.shape}")
    
    print("\nTesting ConvLinearModel...")
    conv_model = ConvLinearModel(input_dim=200, dim=8, layer_num=4)
    x_conv = torch.randn(4, 200)  # (B, L)
    y_conv = conv_model(x_conv)
    print(f"ConvLinearModel - Input: {x_conv.shape}, Output: {y_conv.shape}")
