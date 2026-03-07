"""Torch spectral models used in v1 experiments."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        hidden = max(64, self.n_segments * embed_dim // 4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_segments * embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)
        z = x.squeeze(1).view(x.size(0), self.n_segments, self.segment_len)
        z = self.token_proj(z)
        z = self.token_norm(z)
        return self.head(z)


class RamanNetSE1D(nn.Module):
    """RamanNet variant with squeeze-excitation style token gating."""

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
        gate_hidden = max(8, embed_dim // 4)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, embed_dim),
            nn.Sigmoid(),
        )
        hidden = max(64, self.n_segments * embed_dim // 4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_segments * embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)
        z = x.squeeze(1).view(x.size(0), self.n_segments, self.segment_len)
        z = self.token_proj(z)
        z = self.token_norm(z)
        g = self.gate(z.mean(dim=1)).unsqueeze(1)
        z = z * g
        return self.head(z)


class RamanNetMultiScale1D(nn.Module):
    """RamanNet-style classifier with multi-scale convolutional tokenization."""

    def __init__(
        self,
        n_classes: int,
        *,
        n_segments: int = 64,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_segments = int(max(8, n_segments))
        branch_ch = max(8, embed_dim // 2)

        def _branch(kernel_size: int) -> nn.Module:
            pad = kernel_size // 2
            return nn.Sequential(
                nn.Conv1d(1, branch_ch, kernel_size=kernel_size, padding=pad, bias=False),
                nn.BatchNorm1d(branch_ch),
                _act(),
            )

        self.b3 = _branch(3)
        self.b7 = _branch(7)
        self.b15 = _branch(15)
        self.token_proj = nn.Linear(branch_ch * 3, embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        hidden = max(64, self.n_segments * embed_dim // 4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_segments * embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.b3(x), self.b7(x), self.b15(x)], dim=1)
        z = F.interpolate(z, size=self.n_segments, mode="linear", align_corners=False)
        z = z.transpose(1, 2)
        z = self.token_proj(z)
        z = self.token_norm(z)
        return self.head(z)


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
    ) -> None:
        super().__init__()
        self.patch_size = int(max(4, patch_size))
        self.patch = nn.Conv1d(
            1, d_model, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.patch_size:
            x = F.interpolate(
                x, size=self.patch_size, mode="linear", align_corners=False
            )
        z = self.patch(x).transpose(1, 2)
        cls = self.cls_token.expand(z.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = self.pos_drop(z)
        z = self.encoder(z)
        z = self.norm(z[:, 0, :])
        return self.head(z)


class SpectralTransformerPatchMixClassifier(nn.Module):
    """Transformer with mixed patch embeddings at two scales."""

    def __init__(
        self,
        n_classes: int,
        *,
        patch_size: int = 8,
        patch_size_b: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size_a = int(max(4, patch_size))
        self.patch_size_b = int(max(4, patch_size_b))
        d_a = max(16, d_model // 2)
        d_b = int(max(16, d_model - d_a))
        self.patch_a = nn.Conv1d(
            1,
            d_a,
            kernel_size=self.patch_size_a,
            stride=self.patch_size_a,
            bias=False,
        )
        self.patch_b = nn.Conv1d(
            1,
            d_b,
            kernel_size=self.patch_size_b,
            stride=self.patch_size_b,
            bias=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_len = max(self.patch_size_a, self.patch_size_b)
        if x.shape[-1] < min_len:
            x = F.interpolate(x, size=min_len, mode="linear", align_corners=False)
        z_a = self.patch_a(x)
        z_b = self.patch_b(x)
        if z_b.shape[-1] != z_a.shape[-1]:
            z_b = F.interpolate(z_b, size=z_a.shape[-1], mode="linear", align_corners=False)
        z = torch.cat([z_a, z_b], dim=1).transpose(1, 2)
        cls = self.cls_token.expand(z.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = self.pos_drop(z)
        z = self.encoder(z)
        z = self.norm(z[:, 0, :])
        return self.head(z)


class SpectralTransformerAttnPoolClassifier(nn.Module):
    """Transformer with learnable attention pooling instead of cls-token head."""

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
    ) -> None:
        super().__init__()
        self.patch_size = int(max(4, patch_size))
        self.patch = nn.Conv1d(
            1, d_model, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )
        self.pos_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(d_model)
        self.pool_query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.pool_query, mean=0.0, std=0.02)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.patch_size:
            x = F.interpolate(
                x, size=self.patch_size, mode="linear", align_corners=False
            )
        z = self.patch(x).transpose(1, 2)
        z = self.pos_drop(z)
        z = self.encoder(z)
        z = self.norm(z)
        logits = torch.matmul(z, self.pool_query) / math.sqrt(z.size(-1))
        attn = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(z * attn, dim=1)
        return self.head(pooled)


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


def build_torch_spectral_model(
    model_cfg: Any, n_features: int, n_classes: int
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
    if family == "ramannet_se":
        n_segments = int(getattr(model_cfg, "ramannet_segments", 64))
        segment_len = max(4, int(n_features // max(8, n_segments)))
        return RamanNetSE1D(
            n_classes=n_classes,
            n_segments=n_segments,
            segment_len=segment_len,
            embed_dim=int(getattr(model_cfg, "ramannet_embed_dim", 64)),
            dropout=float(getattr(model_cfg, "ramannet_dropout", 0.1)),
        )
    if family == "ramannet_multiscale":
        return RamanNetMultiScale1D(
            n_classes=n_classes,
            n_segments=int(getattr(model_cfg, "ramannet_segments", 64)),
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
        )
    if family == "spectral_transformer_patchmix":
        patch_a = int(getattr(model_cfg, "transformer_patch_size", 16))
        patch_b_raw = int(getattr(model_cfg, "transformer_patch_size_b", 0))
        patch_b = patch_b_raw if patch_b_raw > 0 else max(4, patch_a * 2)
        return SpectralTransformerPatchMixClassifier(
            n_classes=n_classes,
            patch_size=patch_a,
            patch_size_b=patch_b,
            d_model=int(getattr(model_cfg, "transformer_d_model", 128)),
            nhead=int(getattr(model_cfg, "transformer_nhead", 4)),
            num_layers=int(getattr(model_cfg, "transformer_num_layers", 3)),
            dim_ff=int(getattr(model_cfg, "transformer_dim_ff", 256)),
            dropout=float(getattr(model_cfg, "transformer_dropout", 0.1)),
        )
    if family == "spectral_transformer_attnpool":
        return SpectralTransformerAttnPoolClassifier(
            n_classes=n_classes,
            patch_size=int(getattr(model_cfg, "transformer_patch_size", 16)),
            d_model=int(getattr(model_cfg, "transformer_d_model", 128)),
            nhead=int(getattr(model_cfg, "transformer_nhead", 4)),
            num_layers=int(getattr(model_cfg, "transformer_num_layers", 3)),
            dim_ff=int(getattr(model_cfg, "transformer_dim_ff", 256)),
            dropout=float(getattr(model_cfg, "transformer_dropout", 0.1)),
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
