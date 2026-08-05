                                                              

                                                                           
                                                                            
   

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def overlapping_patches(x: Tensor, patch_size: int, stride: int) -> Tensor:
                                                                  
    if x.ndim != 3:
        raise ValueError(f"Expected [B, C, T], received {tuple(x.shape)}")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if x.shape[-1] < patch_size:
        raise ValueError(
            f"Input length {x.shape[-1]} is shorter than patch_size {patch_size}"
        )
    return x.unfold(dimension=-1, size=patch_size, step=stride).contiguous()


class DeterministicAdaptiveMeanPool1d(nn.Module):
                                                                        

    def __init__(self, output_size: int) -> None:
        super().__init__()
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        self.output_size = int(output_size)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] <= 0:
            raise ValueError("pool input must have shape [B, C, T] with T > 0")
        length = x.shape[-1]
        bins = []
        for index in range(self.output_size):
            start = math.floor(index * length / self.output_size)
            stop = math.ceil((index + 1) * length / self.output_size)
            bins.append(x[..., start:stop].mean(dim=-1))
        return torch.stack(bins, dim=-1)


class DualDomainPatchEmbedding(nn.Module):
                                                                                

    def __init__(
        self,
        num_channels: int,
        num_patches: int,
        patch_size: int,
        d_model: int,
        temporal_channels: int = 16,
        dropout: float = 0.1,
        spectral_mode: str = "amplitude",
    ) -> None:
        super().__init__()
        if spectral_mode not in {"amplitude", "log_power"}:
            raise ValueError("spectral_mode must be 'amplitude' or 'log_power'")
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.d_model = d_model
        self.spectral_mode = spectral_mode

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, temporal_channels, kernel_size=7, padding=3),
            nn.GroupNorm(4, temporal_channels),
            nn.GELU(),
            nn.Conv1d(temporal_channels, temporal_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, temporal_channels),
            nn.GELU(),
            DeterministicAdaptiveMeanPool1d(8),
        )
        self.temporal_projection = nn.Linear(temporal_channels * 8, d_model)
        self.spectral_projection = nn.Linear(patch_size // 2 + 1, d_model)
        self.fusion = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout))

        self.channel_position = nn.Parameter(
            torch.zeros(1, num_channels, 1, d_model)
        )
        self.patch_position = nn.Parameter(
            torch.zeros(1, 1, num_patches, d_model)
        )
        self.acpe = nn.Conv2d(
            d_model,
            d_model,
            kernel_size=(3, 3),
            padding=(1, 1),
            groups=d_model,
        )
        nn.init.normal_(self.channel_position, std=0.02)
        nn.init.normal_(self.patch_position, std=0.02)

    def forward(self, patches: Tensor) -> Tensor:
        if patches.ndim != 4:
            raise ValueError("patches must have shape [B, C, P, L]")
        batch, channels, patch_count, patch_size = patches.shape
        expected = (self.num_channels, self.num_patches, self.patch_size)
        actual = (channels, patch_count, patch_size)
        if actual != expected:
            raise ValueError(f"Expected patch shape {expected}, received {actual}")

        flat = patches.reshape(batch * channels * patch_count, 1, patch_size)
        temporal = self.temporal_conv(flat).flatten(start_dim=1)
        temporal = self.temporal_projection(temporal)

        spectrum = torch.fft.rfft(flat.squeeze(1), dim=-1, norm="ortho").abs()
        if self.spectral_mode == "log_power":
            spectrum = torch.log1p(spectrum.square())
        spectral = self.spectral_projection(spectrum)
        tokens = self.fusion(temporal + spectral)
        tokens = tokens.view(batch, channels, patch_count, self.d_model)
        tokens = tokens + self.channel_position + self.patch_position

        positional = self.acpe(tokens.permute(0, 3, 1, 2))
        return tokens + positional.permute(0, 2, 3, 1)


class CrissCrossBlock(nn.Module):
                                                               

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even for criss-cross attention")
        half = d_model // 2
        attention_heads = max(1, num_heads // 2)
        if half % attention_heads:
            raise ValueError("d_model / 2 must be divisible by num_heads / 2")

        self.norm_attention = nn.LayerNorm(d_model)
        self.spatial_attention = nn.MultiheadAttention(
            half, attention_heads, dropout=dropout, batch_first=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            half, attention_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        hidden = int(d_model * ffn_ratio)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, patches, d_model = x.shape
        normalized = self.norm_attention(x)
        spatial, temporal = normalized.split(d_model // 2, dim=-1)

        spatial = spatial.permute(0, 2, 1, 3).reshape(
            batch * patches, channels, d_model // 2
        )
        spatial = self.spatial_attention(spatial, spatial, spatial, need_weights=False)[0]
        spatial = spatial.view(batch, patches, channels, d_model // 2).permute(
            0, 2, 1, 3
        )

        temporal = temporal.reshape(batch * channels, patches, d_model // 2)
        temporal = self.temporal_attention(
            temporal, temporal, temporal, need_weights=False
        )[0]
        temporal = temporal.view(batch, channels, patches, d_model // 2)

        x = x + self.attention_dropout(torch.cat((spatial, temporal), dim=-1))
        return x + self.ffn(self.norm_ffn(x))


class DynamicAdjacency(nn.Module):
                                                                             

    def __init__(
        self,
        d_model: int,
        attention_dim: Optional[int] = None,
        topology_bias: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        attention_dim = attention_dim or d_model
        self.attention_dim = attention_dim
        self.query = nn.Linear(d_model, attention_dim, bias=False)
        self.key = nn.Linear(d_model, attention_dim, bias=False)
        if topology_bias is None:
            self.register_buffer("topology_bias", None)
        else:
            if topology_bias.ndim != 2 or topology_bias.shape[0] != topology_bias.shape[1]:
                raise ValueError("topology_bias must have shape [C, C]")
            self.register_buffer("topology_bias", topology_bias.float())

    def forward(self, h: Tensor) -> Tensor:
        channel_state = h.mean(dim=2)
        query = self.query(channel_state)
        key = self.key(channel_state)
        logits = torch.matmul(query, key.transpose(-1, -2))
        logits = logits / math.sqrt(self.attention_dim)
        if self.topology_bias is not None:
            if self.topology_bias.shape != logits.shape[-2:]:
                raise ValueError("topology_bias channel count does not match input")
            logits = logits + self.topology_bias
        return torch.softmax(logits, dim=-1)


class CGODEFunction(nn.Module):
                                                        

    def __init__(
        self,
        d_model: int,
        attention_dim: Optional[int] = None,
        rho_init: float = 0.1,
        topology_bias: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.adjacency = DynamicAdjacency(d_model, attention_dim, topology_bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        rho_init = max(float(rho_init), 1e-6)
        self.raw_rho = nn.Parameter(torch.tensor(math.log(math.expm1(rho_init))))
        self.nfe = 0
        self.last_adjacency: Optional[Tensor] = None
        self.adjacency_trace: List[Tensor] = []

    def reset_diagnostics(self) -> None:
        self.nfe = 0
        self.last_adjacency = None
        self.adjacency_trace = []

    def forward(self, tau: Tensor, h: Tensor) -> Tensor:
        del tau
        adjacency = self.adjacency(h)
        message = torch.einsum("bij,bjpd->bipd", adjacency, h)
        derivative = F.gelu(self.output_projection(message))
        derivative = derivative - F.softplus(self.raw_rho) * h
        self.nfe += 1
        self.last_adjacency = adjacency
        self.adjacency_trace.append(adjacency.detach())
        return derivative


class FixedStepRK4(nn.Module):
                                                             

    def __init__(self, t0: float = 0.0, t1: float = 1.0, step_size: float = 0.25):
        super().__init__()
        if t1 <= t0:
            raise ValueError("t1 must be greater than t0")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        steps_float = (t1 - t0) / step_size
        steps = int(round(steps_float))
        if not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("step_size must divide the integration interval exactly")
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.step_size = float(step_size)
        self.steps = steps

    def forward(self, function: nn.Module, initial_state: Tensor) -> Tensor:
        h = initial_state
        dt = self.step_size
        tau = initial_state.new_tensor(self.t0)
        for _ in range(self.steps):
            k1 = function(tau, h)
            k2 = function(tau + dt / 2, h + dt * k1 / 2)
            k3 = function(tau + dt / 2, h + dt * k2 / 2)
            k4 = function(tau + dt, h + dt * k3)
            h = h + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            tau = tau + dt
        return h


class CGODEBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        step_size: float = 0.25,
        rho_init: float = 0.1,
        attention_dim: Optional[int] = None,
        topology_bias: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.function = CGODEFunction(
            d_model=d_model,
            attention_dim=attention_dim,
            rho_init=rho_init,
            topology_bias=topology_bias,
        )
        self.solver = FixedStepRK4(t0=0.0, t1=1.0, step_size=step_size)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, h: Tensor) -> Tuple[Tensor, Dict[str, object]]:
        self.function.reset_diagnostics()
        solved = self.solver(self.function, h)
                                                                         
                                                                      
        output = self.output_norm(solved)
        trace = self.function.adjacency_trace
        metadata: Dict[str, object] = {
            "nfe": self.function.nfe,
            "adjacency": (
                self.function.last_adjacency.detach()
                if self.function.last_adjacency is not None
                else None
            ),
            "adjacency_trace": trace,
            "solver": "fixed_rk4",
            "interval": (self.solver.t0, self.solver.t1),
            "step_size": self.solver.step_size,
        }
        return output, metadata


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class STEM(nn.Module):
                                                                       

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_patches: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.position = SinusoidalPosition(d_model, max_patches)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * ffn_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequence: Tensor) -> Tensor:
        return self.norm(self.encoder(self.position(sequence)))


@dataclass
class ClassifierOutput:
    logits: Tensor
    features: Tensor
    graph: Dict[str, object]
    tokens: Tensor


class CGODECCTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        input_length: int,
        patch_size: int,
        patch_stride: int,
        d_model: int = 256,
        cct_layers: int = 12,
        num_heads: int = 8,
        stem_layers: int = 2,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        ode_step_size: float = 0.25,
        ode_rho_init: float = 0.1,
        topology_bias: Optional[Tensor] = None,
        spectral_mode: str = "amplitude",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.input_length = input_length
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_count = (input_length - patch_size) // patch_stride + 1
        self.d_model = d_model

        self.patch_embedding = DualDomainPatchEmbedding(
            num_channels=num_channels,
            num_patches=self.patch_count,
            patch_size=patch_size,
            d_model=d_model,
            dropout=dropout,
            spectral_mode=spectral_mode,
        )
        self.cct = nn.ModuleList(
            [CrissCrossBlock(d_model, num_heads, dropout=dropout) for _ in range(cct_layers)]
        )
        self.pre_graph_norm = nn.LayerNorm(d_model)
        self.graph_block = CGODEBlock(
            d_model=d_model,
            step_size=ode_step_size,
            rho_init=ode_rho_init,
            topology_bias=topology_bias,
        )
        self.stem = STEM(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=stem_layers,
            max_patches=self.patch_count,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, object]]:
        if x.shape[1:] != (self.num_channels, self.input_length):
            raise ValueError(
                "Expected input [B, "
                f"{self.num_channels}, {self.input_length}], received {tuple(x.shape)}"
            )
        patches = overlapping_patches(x, self.patch_size, self.patch_stride)
        tokens = self.patch_embedding(patches)
        for block in self.cct:
            tokens = block(tokens)
        tokens = self.pre_graph_norm(tokens)

        tokens, graph = self.graph_block(tokens)

        sequence = tokens.mean(dim=1)
        sequence = self.stem(sequence)
        features = sequence.mean(dim=1)
        return features, tokens, graph

    def forward(self, x: Tensor, return_details: bool = False):
        features, tokens, graph = self.forward_features(x)
        logits = self.classifier(features)
        if return_details:
            return ClassifierOutput(logits, features, graph, tokens)
        return logits
