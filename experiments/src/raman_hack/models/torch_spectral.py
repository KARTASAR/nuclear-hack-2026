"""Torch spectral models used in v1 experiments."""

from __future__ import annotations

import hashlib
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import nnls
from sklearn.decomposition import NMF, PCA

from .multitaper import MultiTaperConfig, RamanMultiTaperEmbedding
from .moe import RamanMoENet, build_sector_bounds


def _act() -> nn.Module:
    return nn.GELU()


class ResidualBlock1D(nn.Module):
    """Residual 1D block with optional projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act(out)


class ResNet1DClassifier(nn.Module):
    """1D ResNet-style classifier for Raman spectra."""

    def __init__(
        self,
        n_classes: int,
        *,
        base_ch: int = 32,
        blocks: tuple[int, int, int] = (2, 2, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
        )
        ch = base_ch
        layers: list[nn.Module] = []
        for _ in range(max(1, int(blocks[0]))):
            layers.append(ResidualBlock1D(ch, ch, stride=1, dropout=dropout))
        layers.append(ResidualBlock1D(ch, ch * 2, stride=2, dropout=dropout))
        ch *= 2
        for _ in range(max(1, int(blocks[1])) - 1):
            layers.append(ResidualBlock1D(ch, ch, stride=1, dropout=dropout))
        layers.append(ResidualBlock1D(ch, ch * 2, stride=2, dropout=dropout))
        ch *= 2
        for _ in range(max(1, int(blocks[2])) - 1):
            layers.append(ResidualBlock1D(ch, ch, stride=1, dropout=dropout))
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class RamanNet1D(nn.Module):
    """RamanNet-like segment MLP network with shared token projection."""

    def __init__(
        self,
        n_classes: int,
        *,
        n_segments: int = 64,
        segment_len: int = 16,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_segments = int(max(8, n_segments))
        self.segment_len = int(max(4, segment_len))
        self.target_len = self.n_segments * self.segment_len
        self.token_proj = nn.Linear(self.segment_len, embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.hidden_dim = max(64, self.n_segments * embed_dim // 4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_segments * embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, n_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding before classification."""
        if x.shape[-1] != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)
        z = x.squeeze(1).view(x.size(0), self.n_segments, self.segment_len)
        z = self.token_proj(z)
        z = self.token_norm(z)
        return self.head[:4](z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)
        z = x.squeeze(1).view(x.size(0), self.n_segments, self.segment_len)
        z = self.token_proj(z)
        z = self.token_norm(z)
        return self.head(z)


class EncoderLayerWithAttn(nn.Module):
    """Transformer encoder layer exposing attention maps for rollout-style XAI."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        if self.head_dim * self.nhead != self.d_model:
            raise ValueError("d_model must be divisible by nhead")

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.lin1 = nn.Linear(self.d_model, int(dim_ff))
        self.lin2 = nn.Linear(int(dim_ff), self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.act = nn.GELU()

        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tokens, channels = x.shape
        q = self.q_proj(x).view(bsz, tokens, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, tokens, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, tokens, self.nhead, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        self.last_attn = attn

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, tokens, channels)
        out = self.out_proj(out)
        x = self.norm1(x + self.drop(out))

        ff = self.lin2(self.drop(self.act(self.lin1(x))))
        return self.norm2(x + self.drop(ff))


class TransformerEncoderWithAttn(nn.Module):
    """Stack of attention-exposing encoder layers."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayerWithAttn(
                    d_model=d_model,
                    nhead=nhead,
                    dim_ff=dim_ff,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def get_attn_stack(self) -> list[torch.Tensor]:
        stack = [layer.last_attn for layer in self.layers if layer.last_attn is not None]
        return stack

    def get_attn_grad_stack(self) -> list[torch.Tensor | None]:
        return [None if layer.last_attn is None else layer.last_attn.grad for layer in self.layers]

    def retain_attn_grads(self) -> None:
        for layer in self.layers:
            if layer.last_attn is not None:
                layer.last_attn.retain_grad()


class SpectralTransformerClassifier(nn.Module):
    """Lightweight 1D ViT/RaT-like classifier for spectra."""

    def __init__(
        self,
        n_classes: int,
        *,
        patch_size: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        frontend: str = "raw",
        multitaper_nw: float = 3.0,
        multitaper_n_tapers: int = 5,
        multitaper_remove_dc: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = int(max(4, patch_size))
        self.frontend_name = str(frontend).lower()
        if self.frontend_name == "multitaper":
            self.frontend: nn.Module = RamanMultiTaperEmbedding(
                MultiTaperConfig(
                    nw=float(multitaper_nw),
                    n_tapers=int(multitaper_n_tapers),
                    remove_dc=bool(multitaper_remove_dc),
                )
            )
        else:
            self.frontend = nn.Identity()
        self.patch = nn.Conv1d(
            1, d_model, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.encoder = TransformerEncoderWithAttn(
            d_model=d_model,
            nhead=nhead,
            num_layers=max(1, int(num_layers)),
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        self.last_patch_tokens: torch.Tensor | None = None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS embedding before classification."""
        x = self.frontend(x)
        if x.shape[-1] < self.patch_size:
            x = F.interpolate(
                x, size=self.patch_size, mode="linear", align_corners=False
            )
        z = self.patch(x).transpose(1, 2)
        if z.requires_grad:
            z.retain_grad()
        self.last_patch_tokens = z
        cls = self.cls_token.expand(z.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = self.pos_drop(z)
        z = self.encoder(z)
        return self.norm(z[:, 0, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))


class Inception1DBlock(nn.Module):
    """Inception-like 1D block with multi-kernel branches."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1) -> None:
        super().__init__()
        bch = max(4, out_ch // 4)
        self.b1 = nn.Sequential(
            nn.Conv1d(in_ch, bch, kernel_size=1, bias=False),
            nn.BatchNorm1d(bch),
            _act(),
            nn.Conv1d(bch, bch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(bch),
            _act(),
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(in_ch, bch, kernel_size=1, bias=False),
            nn.BatchNorm1d(bch),
            _act(),
            nn.Conv1d(bch, bch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(bch),
            _act(),
        )
        self.b3 = nn.Sequential(
            nn.Conv1d(in_ch, bch, kernel_size=1, bias=False),
            nn.BatchNorm1d(bch),
            _act(),
            nn.Conv1d(bch, bch, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(bch),
            _act(),
        )
        self.b4 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, bch, kernel_size=1, bias=False),
            nn.BatchNorm1d(bch),
            _act(),
        )
        self.mix = nn.Sequential(
            nn.Conv1d(bch * 4, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            _act(),
            nn.Dropout(dropout),
        )
        self.shortcut: nn.Module
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        y = self.mix(y)
        return _act()(y + self.shortcut(x))


class Inception1DClassifier(nn.Module):
    """Inception-1D classifier for Raman spectra."""

    def __init__(self, n_classes: int, *, base_ch: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        ch = max(8, int(base_ch))
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
        )
        self.block1 = Inception1DBlock(ch, ch, dropout=dropout)
        self.block2 = Inception1DBlock(ch, ch * 2, dropout=dropout)
        self.down = nn.MaxPool1d(kernel_size=2, stride=2)
        self.block3 = Inception1DBlock(ch * 2, ch * 2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.down(z)
        z = self.block3(z)
        z = self.pool(z).squeeze(-1)
        return self.head(z)


class ResidualShrinkageBlock1D(nn.Module):
    """Residual shrinkage block with channel-wise soft thresholding."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        hidden = max(8, out_ch // 4)
        self.fc = nn.Sequential(
            nn.Linear(out_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_ch),
            nn.Sigmoid(),
        )
        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.bn1(self.conv1(x)), inplace=True)
        z = self.bn2(self.conv2(z))
        abs_mean = z.abs().mean(dim=-1)
        scale = self.fc(abs_mean)
        thresh = (abs_mean * scale).unsqueeze(-1)
        z = torch.sign(z) * F.relu(z.abs() - thresh)
        z = z + self.shortcut(x)
        return F.relu(z, inplace=True)


class DRSN1DClassifier(nn.Module):
    """1D Deep Residual Shrinkage classifier."""

    def __init__(
        self, n_classes: int, *, base_ch: int = 32, blocks: tuple[int, int, int] = (2, 2, 2)
    ) -> None:
        super().__init__()
        ch = max(8, int(base_ch))
        self.stem = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
        )
        layers: list[nn.Module] = []
        for _ in range(max(1, int(blocks[0]))):
            layers.append(ResidualShrinkageBlock1D(ch, ch, stride=1))
        layers.append(ResidualShrinkageBlock1D(ch, ch * 2, stride=2))
        ch *= 2
        for _ in range(max(1, int(blocks[1])) - 1):
            layers.append(ResidualShrinkageBlock1D(ch, ch, stride=1))
        layers.append(ResidualShrinkageBlock1D(ch, ch * 2, stride=2))
        ch *= 2
        for _ in range(max(1, int(blocks[2])) - 1):
            layers.append(ResidualShrinkageBlock1D(ch, ch, stride=1))
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.encoder(z)
        z = self.pool(z).squeeze(-1)
        return self.head(z)


class SqueezeExcite1D(nn.Module):
    """Squeeze-and-excitation for 1D features."""

    def __init__(self, ch: int, ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(4, int(ch * ratio))
        self.fc = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)
        s = self.fc(s).unsqueeze(-1)
        return x * s


class MBConv1D(nn.Module):
    """Mobile inverted bottleneck block for 1D signals."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        expand: int = 4,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hid = max(in_ch, in_ch * expand)
        self.use_skip = stride == 1 and in_ch == out_ch
        self.expand = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=1, bias=False),
            nn.BatchNorm1d(hid),
            nn.SiLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                hid,
                hid,
                kernel_size=5,
                stride=stride,
                padding=2,
                groups=hid,
                bias=False,
            ),
            nn.BatchNorm1d(hid),
            nn.SiLU(inplace=True),
        )
        self.se = SqueezeExcite1D(hid, ratio=0.25)
        self.project = nn.Sequential(
            nn.Conv1d(hid, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.expand(x)
        z = self.depthwise(z)
        z = self.se(z)
        z = self.project(z)
        z = self.drop(z)
        if self.use_skip:
            z = z + x
        return z


class EfficientNet1DClassifier(nn.Module):
    """EfficientNet-lite style classifier for 1D spectra."""

    def __init__(
        self, n_classes: int, *, width_mult: float = 1.0, dropout: float = 0.2
    ) -> None:
        super().__init__()
        def _c(v: int) -> int:
            return max(8, int(v * float(width_mult)))

        c0, c1, c2, c3 = _c(24), _c(40), _c(64), _c(96)
        self.stem = nn.Sequential(
            nn.Conv1d(1, c0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c0),
            nn.SiLU(inplace=True),
        )
        self.b1 = nn.Sequential(MBConv1D(c0, c0, expand=2, stride=1), MBConv1D(c0, c1, expand=4, stride=2))
        self.b2 = nn.Sequential(MBConv1D(c1, c1, expand=4, stride=1), MBConv1D(c1, c2, expand=4, stride=2))
        self.b3 = nn.Sequential(MBConv1D(c2, c2, expand=4, stride=1), MBConv1D(c2, c3, expand=4, stride=2))
        self.head_conv = nn.Sequential(
            nn.Conv1d(c3, _c(128), kernel_size=1, bias=False),
            nn.BatchNorm1d(_c(128)),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(_c(128), n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.b1(z)
        z = self.b2(z)
        z = self.b3(z)
        z = self.head_conv(z)
        z = self.pool(z).squeeze(-1)
        return self.head(z)


class ResidualPreprocessor1D(nn.Module):
    """Learnable residual preprocessor for single-step pipeline."""

    def __init__(self, channels: int = 32, n_blocks: int = 3) -> None:
        super().__init__()
        ch = max(8, int(channels))
        layers: list[nn.Module] = [
            nn.Conv1d(1, ch, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
        ]
        for _ in range(max(1, int(n_blocks))):
            layers.extend(
                [
                    nn.Conv1d(ch, ch, kernel_size=5, padding=2, bias=False),
                    nn.BatchNorm1d(ch),
                    _act(),
                ]
            )
        layers.append(nn.Conv1d(ch, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.net(x)
        return x + delta


class TinyUNet1D(nn.Module):
    """Small 1D U-Net used as learnable spectral preprocessor."""

    def __init__(self, base_ch: int = 16) -> None:
        super().__init__()
        ch = max(8, int(base_ch))
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
        )
        self.down = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(ch, ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch * 2),
            _act(),
            nn.Conv1d(ch * 2, ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch * 2),
            _act(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(ch * 3, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
        )
        self.out = nn.Conv1d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        z = self.down(s1)
        z = self.bottleneck(z)
        z = F.interpolate(z, size=s1.shape[-1], mode="linear", align_corners=False)
        z = torch.cat([z, s1], dim=1)
        z = self.dec(z)
        delta = self.out(z)
        return x + delta


class SingleStepResidualPreprocClassifier(nn.Module):
    """Residual-preprocessor + classifier head."""

    def __init__(
        self,
        n_classes: int,
        *,
        n_features: int,
        preproc_channels: int = 32,
        preproc_blocks: int = 3,
        head: str = "ramannet",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.preproc = ResidualPreprocessor1D(
            channels=preproc_channels, n_blocks=preproc_blocks
        )
        self.head_name = str(head).lower()
        if self.head_name == "spectral_transformer":
            self.head = SpectralTransformerClassifier(
                n_classes=n_classes, patch_size=16, d_model=128, nhead=4, num_layers=3
            )
        else:
            segment_len = max(4, int(n_features // 64))
            self.head = RamanNet1D(
                n_classes=n_classes,
                n_segments=64,
                segment_len=segment_len,
                embed_dim=64,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.preproc(x)
        return self.head(z)


class SingleStepUNetPreprocClassifier(nn.Module):
    """UNet-preprocessor + classifier head."""

    def __init__(
        self,
        n_classes: int,
        *,
        n_features: int,
        unet_base_ch: int = 16,
        head: str = "spectral_transformer",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.preproc = TinyUNet1D(base_ch=unet_base_ch)
        self.head_name = str(head).lower()
        if self.head_name == "ramannet":
            segment_len = max(4, int(n_features // 64))
            self.head = RamanNet1D(
                n_classes=n_classes,
                n_segments=64,
                segment_len=segment_len,
                embed_dim=64,
                dropout=dropout,
            )
        else:
            self.head = SpectralTransformerClassifier(
                n_classes=n_classes, patch_size=16, d_model=128, nhead=4, num_layers=3
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.preproc(x)
        return self.head(z)


def _infer_center_from_wavenumbers(wavenumbers: np.ndarray | None) -> str:
    if wavenumbers is None:
        return "1500"
    wn = np.asarray(wavenumbers, dtype=float)
    if wn.size == 0:
        return "1500"
    return "1500" if float(np.nanmedian(wn)) < 2200.0 else "2900"


class RSGAN1D(nn.Module):
    """Lightweight 1D GAN used to synthesize realistic noisy Raman spectra."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        ch = int(max(8, channels))
        self.generator = nn.Sequential(
            nn.Conv1d(2, ch, kernel_size=7, padding=3),
            _act(),
            nn.Conv1d(ch, ch, kernel_size=5, padding=2),
            _act(),
            nn.Conv1d(ch, 1, kernel_size=1),
        )
        self.discriminator = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ch, ch * 2, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch * 2, 1),
        )

    def generate(self, clean: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean)
        return self.generator(torch.cat([clean, noise], dim=1))

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class _BackgroundUNet1D(nn.Module):
    """Small U-shaped block to estimate baseline/background contour."""

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        ch = int(max(8, base_ch))
        self.enc = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(ch, ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch * 2),
            _act(),
            nn.Conv1d(ch * 2, ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch * 2),
            _act(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(ch * 3, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
        )
        self.contour_head = nn.Conv1d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.enc(x)
        z = self.pool(skip)
        z = self.bottleneck(z)
        z = F.interpolate(z, size=skip.shape[-1], mode="linear", align_corners=False)
        z = torch.cat([z, skip], dim=1)
        feat = self.dec(z)
        contour = self.contour_head(feat)
        return feat, contour


class RSBPCNN1D(nn.Module):
    """Two-stream denoiser inspired by RSBPCNN."""

    def __init__(self, base_ch: int = 32, use_l2_norm: bool = True) -> None:
        super().__init__()
        ch = int(max(8, base_ch))
        self.use_l2_norm = bool(use_l2_norm)
        self.patches_branch = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
        )
        self.bg_branch = _BackgroundUNet1D(base_ch=ch)
        self.pre_enc_patches = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
        )
        self.pre_enc_bg = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
            _act(),
        )
        self.inception_stack = nn.Sequential(
            Inception1DBlock(ch, ch),
            Inception1DBlock(ch, ch),
        )
        self.clean_head = nn.Conv1d(ch, 1, kernel_size=1)
        # Default Grad-CAM hook for RGT explanations.
        self.xai_target_layer: nn.Module = self.clean_head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self.patches_branch(x)
        bg_feat, bg_contour = self.bg_branch(x)
        p = self.pre_enc_patches(patches)
        b = self.pre_enc_bg(bg_feat)
        merged = p - b
        z = self.inception_stack(merged)
        clean = self.clean_head(z)
        if self.use_l2_norm:
            clean = F.normalize(clean, p=2, dim=-1, eps=1e-12)
        return clean, bg_contour


def inject_physical_artifacts(
    clean: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Add baseline/noise/spikes to clean spectra and return (noisy, baseline)."""
    X = np.asarray(clean, dtype=np.float64)
    n, m = X.shape
    grid = np.linspace(-1.0, 1.0, m, dtype=np.float64)
    baseline = np.zeros_like(X)
    noisy = X.copy()
    for i in range(n):
        a = float(rng.uniform(0.02, 0.15))
        b = float(rng.uniform(-0.03, 0.03))
        c = float(rng.uniform(0.0, 0.05))
        bl = a * (grid**2) + b * grid + c
        baseline[i] = bl
        sig = np.clip(X[i], 0.0, None)
        poisson = rng.poisson(lam=np.clip(sig * 20.0, 0.0, None)) / 20.0 - sig
        gaussian = rng.normal(0.0, 0.03 + 0.01 * float(np.std(X[i])), size=m)
        s = X[i] + bl + poisson + gaussian
        n_spikes = int(rng.integers(0, 3))
        if n_spikes > 0:
            pos = rng.integers(0, m, size=n_spikes)
            amps = rng.uniform(0.5, 1.5, size=n_spikes)
            s[pos] += amps
        noisy[i] = s
    return noisy.astype(np.float64, copy=False), baseline.astype(np.float64, copy=False)


def nfindr_endmembers(
    X: np.ndarray,
    k: int,
    *,
    seed: int = 42,
    max_iter: int = 3,
    candidate_pool: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate endmembers with a compact N-FINDR variant."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_samples, n_features = arr.shape
    if n_samples < int(k):
        raise ValueError(f"n_samples ({n_samples}) must be >= k ({k})")
    rng = np.random.default_rng(int(seed))
    if n_samples > int(candidate_pool):
        cand_global = np.sort(rng.choice(n_samples, size=int(candidate_pool), replace=False))
        cand = arr[cand_global]
    else:
        cand_global = np.arange(n_samples, dtype=int)
        cand = arr
    k_int = int(k)
    if float(np.var(cand)) <= 1e-12:
        fallback_idx = cand_global[:k_int]
        return arr[fallback_idx].astype(np.float64, copy=False), fallback_idx.astype(
            np.int64, copy=False
        )
    pca_dim = max(1, int(k) - 1)
    pca_dim = min(pca_dim, cand.shape[0] - 1, cand.shape[1])
    if pca_dim <= 0:
        pca_scores = cand[:, :1]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pca_scores = PCA(n_components=pca_dim, random_state=int(seed)).fit_transform(cand)
        if not np.isfinite(pca_scores).all():
            pca_scores = np.nan_to_num(cand[:, :pca_dim], nan=0.0, posinf=0.0, neginf=0.0)
    if pca_scores.shape[1] < int(k) - 1:
        pad = np.zeros((pca_scores.shape[0], int(k) - 1 - pca_scores.shape[1]), dtype=np.float64)
        pca_scores = np.concatenate([pca_scores, pad], axis=1)
    current = rng.choice(cand.shape[0], size=k_int, replace=False)

    def simplex_volume(indices: np.ndarray) -> float:
        points = pca_scores[indices, : k_int - 1]
        mat = np.concatenate([np.ones((k_int, 1), dtype=np.float64), points], axis=1)
        return float(abs(np.linalg.det(mat)))

    best_vol = simplex_volume(current)
    for _ in range(max(1, int(max_iter))):
        improved = False
        for i in range(k_int):
            local_best = current[i]
            local_best_vol = best_vol
            for cand_idx in range(pca_scores.shape[0]):
                if cand_idx != current[i] and cand_idx in current:
                    continue
                trial = current.copy()
                trial[i] = cand_idx
                v = simplex_volume(trial)
                if v > local_best_vol + 1e-12:
                    local_best = cand_idx
                    local_best_vol = v
            if local_best != current[i]:
                current[i] = local_best
                best_vol = local_best_vol
                improved = True
        if not improved:
            break
    global_idx = cand_global[current]
    endmembers = arr[global_idx]
    return endmembers.astype(np.float64, copy=False), global_idx.astype(np.int64, copy=False)


def nmf_endmembers(
    X: np.ndarray,
    k: int,
    *,
    seed: int = 42,
    max_iter: int = 300,
) -> np.ndarray:
    """Estimate endmembers via non-negative matrix factorization."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if arr.shape[0] < int(k):
        raise ValueError(f"n_samples ({arr.shape[0]}) must be >= k ({k})")
    min_val = float(np.min(arr))
    if min_val < 0.0:
        arr = arr - min_val
    arr = np.clip(arr, 0.0, None)
    model = NMF(
        n_components=int(k),
        init="nndsvda",
        random_state=int(seed),
        max_iter=max(100, int(max_iter)),
    )
    _ = model.fit_transform(arr)
    endmembers = np.asarray(model.components_, dtype=np.float64)
    return endmembers.astype(np.float64, copy=False)


def nnls_abundances(X: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    """Compute non-negative abundance vectors and L1-normalize each row."""
    data = np.asarray(X, dtype=np.float64)
    E = np.asarray(endmembers, dtype=np.float64)
    if data.ndim != 2 or E.ndim != 2:
        raise ValueError("X and endmembers must be 2D arrays")
    if data.shape[1] != E.shape[1]:
        raise ValueError(
            f"Feature mismatch: X has {data.shape[1]}, endmembers have {E.shape[1]}"
        )
    A = E.T
    out = np.zeros((data.shape[0], E.shape[0]), dtype=np.float64)
    for i in range(data.shape[0]):
        try:
            coeff, _ = nnls(A, data[i])
        except Exception:
            coeff = np.maximum(np.linalg.lstsq(A, data[i], rcond=None)[0], 0.0)
        s = float(np.sum(coeff))
        if s > 1e-12:
            coeff = coeff / s
        out[i] = coeff
    return out


def top_abundance_components(
    abundances: np.ndarray,
    *,
    top_k: int = 3,
    prefix: str = "E",
) -> list[str]:
    arr = np.asarray(abundances, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return []
    order = np.argsort(arr)[::-1][: max(1, int(top_k))]
    return [f"{prefix}{int(i + 1)}:{float(arr[i]):.4f}" for i in order]


class RGTClassifier(nn.Module):
    """Raman-GAN-Transformer pipeline: denoise + unmix + deep classification."""

    def __init__(
        self,
        n_classes: int,
        *,
        n_features: int,
        backbone: str = "ramannet",
        unmix_k: int = 5,
        use_l2_norm: bool = True,
        denoiser_channels: int = 32,
        ramannet_segments: int = 64,
        ramannet_embed_dim: int = 64,
        ramannet_dropout: float = 0.1,
        transformer_patch_size: int = 16,
        transformer_d_model: int = 128,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 3,
        transformer_dim_ff: int = 256,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.unmix_k = int(max(2, unmix_k))
        self.use_l2_norm = bool(use_l2_norm)
        self.backbone_name = str(backbone).lower()
        self.denoiser = RSBPCNN1D(
            base_ch=int(max(8, denoiser_channels)),
            use_l2_norm=bool(use_l2_norm),
        )
        self.rsgan = RSGAN1D(channels=int(max(8, denoiser_channels)))
        if self.backbone_name == "spectral_transformer":
            self.backbone: nn.Module = SpectralTransformerClassifier(
                n_classes=n_classes,
                patch_size=int(transformer_patch_size),
                d_model=int(transformer_d_model),
                nhead=int(transformer_nhead),
                num_layers=int(transformer_num_layers),
                dim_ff=int(transformer_dim_ff),
                dropout=float(transformer_dropout),
            )
            emb_dim = int(transformer_d_model)
        else:
            n_segments = int(max(8, ramannet_segments))
            segment_len = max(4, int(self.n_features // n_segments))
            self.backbone = RamanNet1D(
                n_classes=n_classes,
                n_segments=n_segments,
                segment_len=segment_len,
                embed_dim=int(max(8, ramannet_embed_dim)),
                dropout=float(ramannet_dropout),
            )
            emb_dim = int(getattr(self.backbone, "hidden_dim", 128))
        self.embedding_dim = emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim + self.unmix_k, max(32, self.embedding_dim // 2)),
            _act(),
            nn.Dropout(0.1),
            nn.Linear(max(32, self.embedding_dim // 2), n_classes),
        )
        self.register_buffer(
            "endmembers",
            torch.zeros((self.unmix_k, self.n_features), dtype=torch.float32),
            persistent=True,
        )
        self.endmembers_hash: str = ""
        self.last_aux: dict[str, Any] = {}
        self.xai_target_layer: nn.Module = self.denoiser.xai_target_layer
        # Expose transformer encoder for attention rollout compatibility.
        self.encoder = getattr(self.backbone, "encoder", None)

    def set_endmembers(self, endmembers: np.ndarray) -> None:
        arr = np.asarray(endmembers, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("endmembers must be a 2D array")
        if arr.shape[1] != self.n_features:
            raise ValueError(
                f"endmembers feature mismatch: expected {self.n_features}, got {arr.shape[1]}"
            )
        if arr.shape[0] != self.unmix_k:
            raise ValueError(
                f"endmembers count mismatch: expected {self.unmix_k}, got {arr.shape[0]}"
            )
        with torch.no_grad():
            self.endmembers.copy_(torch.from_numpy(arr))
        self.endmembers_hash = hashlib.sha256(arr.tobytes()).hexdigest()

    def has_endmembers(self) -> bool:
        return bool(torch.any(self.endmembers.abs() > 0))

    def _abundances_from_clean(self, clean: torch.Tensor) -> torch.Tensor:
        if not self.has_endmembers():
            return torch.zeros(
                (clean.shape[0], self.unmix_k),
                dtype=clean.dtype,
                device=clean.device,
            )
        clean_np = clean.detach().squeeze(1).cpu().numpy()
        em_np = self.endmembers.detach().cpu().numpy()
        abund_np = nnls_abundances(clean_np, em_np).astype(np.float32, copy=False)
        return torch.from_numpy(abund_np).to(clean.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean, bg_contour = self.denoiser(x)
        if self.use_l2_norm:
            clean = F.normalize(clean, p=2, dim=-1, eps=1e-12)
        embedding = self.backbone.extract_features(clean)  # type: ignore[attr-defined]
        abundances = self._abundances_from_clean(clean)
        logits = self.classifier(torch.cat([embedding, abundances], dim=1))
        aux = {
            "clean_spectrum": clean,
            "bg_contour": bg_contour,
            "abundances": abundances,
            "endmembers": self.endmembers,
            "embedding": embedding,
        }
        self.last_aux = aux
        return logits, aux


def build_torch_spectral_model(
    model_cfg: Any,
    n_features: int,
    n_classes: int,
    *,
    transformed_wavenumbers: np.ndarray | None = None,
) -> nn.Module:
    """Factory for spectral torch models."""
    family = str(getattr(model_cfg, "model_family", "")).lower()
    if family == "resnet1d":
        blocks_raw = getattr(model_cfg, "resnet_blocks", [2, 2, 2])
        if isinstance(blocks_raw, (list, tuple)) and len(blocks_raw) == 3:
            blocks = (int(blocks_raw[0]), int(blocks_raw[1]), int(blocks_raw[2]))
        else:
            blocks = (2, 2, 2)
        return ResNet1DClassifier(
            n_classes=n_classes,
            base_ch=int(getattr(model_cfg, "resnet_base_ch", 32)),
            blocks=blocks,
            dropout=float(getattr(model_cfg, "resnet_dropout", 0.1)),
        )
    if family == "ramannet":
        n_segments = int(getattr(model_cfg, "ramannet_segments", 64))
        segment_len = max(4, int(n_features // max(8, n_segments)))
        return RamanNet1D(
            n_classes=n_classes,
            n_segments=n_segments,
            segment_len=segment_len,
            embed_dim=int(getattr(model_cfg, "ramannet_embed_dim", 64)),
            dropout=float(getattr(model_cfg, "ramannet_dropout", 0.1)),
        )
    if family == "spectral_transformer":
        return SpectralTransformerClassifier(
            n_classes=n_classes,
            patch_size=int(getattr(model_cfg, "transformer_patch_size", 16)),
            d_model=int(getattr(model_cfg, "transformer_d_model", 128)),
            nhead=int(getattr(model_cfg, "transformer_nhead", 4)),
            num_layers=int(getattr(model_cfg, "transformer_num_layers", 3)),
            dim_ff=int(getattr(model_cfg, "transformer_dim_ff", 256)),
            dropout=float(getattr(model_cfg, "transformer_dropout", 0.1)),
            frontend=str(getattr(model_cfg, "transformer_frontend", "raw")),
            multitaper_nw=float(getattr(model_cfg, "multitaper_nw", 3.0)),
            multitaper_n_tapers=int(getattr(model_cfg, "multitaper_n_tapers", 5)),
            multitaper_remove_dc=bool(getattr(model_cfg, "multitaper_remove_dc", True)),
        )
    if family == "rgt_pipeline":
        backbone = str(getattr(model_cfg, "rgt_backbone", "auto")).lower()
        if backbone == "auto":
            center = _infer_center_from_wavenumbers(transformed_wavenumbers)
            backbone = "ramannet" if center == "1500" else "spectral_transformer"
        return RGTClassifier(
            n_classes=n_classes,
            n_features=n_features,
            backbone=backbone,
            unmix_k=int(getattr(model_cfg, "rgt_unmix_k", 5)),
            use_l2_norm=bool(getattr(model_cfg, "rgt_use_l2_norm", True)),
            denoiser_channels=int(
                getattr(model_cfg, "single_step_preproc_channels", 32)
            ),
            ramannet_segments=int(getattr(model_cfg, "ramannet_segments", 64)),
            ramannet_embed_dim=int(getattr(model_cfg, "ramannet_embed_dim", 64)),
            ramannet_dropout=float(getattr(model_cfg, "ramannet_dropout", 0.1)),
            transformer_patch_size=int(getattr(model_cfg, "transformer_patch_size", 16)),
            transformer_d_model=int(getattr(model_cfg, "transformer_d_model", 128)),
            transformer_nhead=int(getattr(model_cfg, "transformer_nhead", 4)),
            transformer_num_layers=int(getattr(model_cfg, "transformer_num_layers", 3)),
            transformer_dim_ff=int(getattr(model_cfg, "transformer_dim_ff", 256)),
            transformer_dropout=float(getattr(model_cfg, "transformer_dropout", 0.1)),
        )
    if family == "raman_moe":
        n_freq_bins_raw = int(getattr(model_cfg, "moe_n_freq_bins", 0))
        n_freq_bins = int(n_features) if n_freq_bins_raw <= 0 else n_freq_bins_raw
        bounds = build_sector_bounds(
            sector_source=str(getattr(model_cfg, "moe_sector_source", "hybrid")),
            sector_bounds_raw=getattr(model_cfg, "moe_sector_bounds", None),
            wavenumbers=transformed_wavenumbers,
        )
        return RamanMoENet(
            n_classes=n_classes,
            emb_channels=int(getattr(model_cfg, "moe_emb_channels", 64)),
            n_experts=int(getattr(model_cfg, "moe_n_experts", 3)),
            expert_hidden=int(getattr(model_cfg, "moe_expert_hidden", 64)),
            use_sectors_for_gating=bool(
                getattr(model_cfg, "moe_use_sectors_for_gating", True)
            ),
            sector_bounds=bounds,
            sector_reduce=str(getattr(model_cfg, "moe_sector_reduce", "mean")),
            sector_learnable_proj=bool(getattr(model_cfg, "moe_learnable_proj", True)),
            sector_proj_dim=int(getattr(model_cfg, "moe_proj_dim", 32)),
            multitaper_nw=float(getattr(model_cfg, "multitaper_nw", 3.0)),
            multitaper_n_tapers=int(getattr(model_cfg, "multitaper_n_tapers", 5)),
            multitaper_remove_dc=bool(getattr(model_cfg, "multitaper_remove_dc", True)),
            n_freq_bins=n_freq_bins,
            wavenumbers=transformed_wavenumbers,
        )
    if family == "inception1d":
        return Inception1DClassifier(
            n_classes=n_classes,
            base_ch=int(getattr(model_cfg, "inception_base_ch", 32)),
            dropout=float(getattr(model_cfg, "inception_dropout", 0.1)),
        )
    if family == "drsn1d":
        blocks_raw = getattr(model_cfg, "drsn_blocks", [2, 2, 2])
        if isinstance(blocks_raw, (list, tuple)) and len(blocks_raw) == 3:
            blocks = (int(blocks_raw[0]), int(blocks_raw[1]), int(blocks_raw[2]))
        else:
            blocks = (2, 2, 2)
        return DRSN1DClassifier(
            n_classes=n_classes,
            base_ch=int(getattr(model_cfg, "drsn_base_ch", 32)),
            blocks=blocks,
        )
    if family == "efficientnet1d":
        return EfficientNet1DClassifier(
            n_classes=n_classes,
            width_mult=float(getattr(model_cfg, "efficientnet_width_mult", 1.0)),
            dropout=float(getattr(model_cfg, "efficientnet_dropout", 0.2)),
        )
    if family == "single_step_residual":
        return SingleStepResidualPreprocClassifier(
            n_classes=n_classes,
            n_features=n_features,
            preproc_channels=int(getattr(model_cfg, "single_step_preproc_channels", 32)),
            preproc_blocks=int(getattr(model_cfg, "single_step_preproc_blocks", 3)),
            head=str(getattr(model_cfg, "single_step_head", "ramannet")),
            dropout=float(getattr(model_cfg, "single_step_head_dropout", 0.1)),
        )
    if family == "single_step_unet":
        return SingleStepUNetPreprocClassifier(
            n_classes=n_classes,
            n_features=n_features,
            unet_base_ch=int(getattr(model_cfg, "single_step_unet_base_ch", 16)),
            head=str(getattr(model_cfg, "single_step_head", "spectral_transformer")),
            dropout=float(getattr(model_cfg, "single_step_head_dropout", 0.1)),
        )
    raise ValueError(f"Unsupported torch model family: {family}")
