                                                                  

                                                                               
                                                                               
                                                                              
                                                                               
                                               
   

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CondDiffEEGConfig:
                                                        

                                                                               
                                                             
       

    in_channels: int = 32
    time_points: int = 1280
    patch_size: int = 4
    embed_dim: int = 512
    depth: int = 8
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_classes: int = 2
    num_subjects: int = 32
    condition_dropout: float = 0.15
    multiscale_kernels: Tuple[int, ...] = (1, 3, 7, 15)
    input_skip_mode: str = "none"
    num_diffusion_steps: int = 1000

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.time_points <= 0:
            raise ValueError("in_channels and time_points must be positive")
        if self.patch_size <= 0 or self.time_points % self.patch_size != 0:
            raise ValueError("patch_size must evenly divide time_points")
        if self.embed_dim <= 0 or self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if not self.multiscale_kernels:
            raise ValueError("at least one multiscale kernel is required")
        if self.embed_dim % len(self.multiscale_kernels) != 0:
            raise ValueError("embed_dim must be divisible by the number of kernels")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.multiscale_kernels):
            raise ValueError("multiscale kernels must be positive odd integers")
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")
        if self.input_skip_mode not in {"none", "timestep_linear"}:
            raise ValueError("input_skip_mode must be 'none' or 'timestep_linear'")
        if self.num_diffusion_steps < 2:
            raise ValueError("num_diffusion_steps must be at least 2")


@dataclass
class DenoiserAux:
                                                                       

    decorrelation: Tensor
    condition_drop_mask: Tensor
    gate: Tensor
    emotion_condition: Tensor
    subject_condition: Tensor


def _validate_label_range(labels: Tensor, upper_bound: int, name: str) -> None:
    if labels.ndim != 1:
        raise ValueError("{} must have shape [batch]".format(name))
    if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= upper_bound):
        raise ValueError("{} contains an out-of-range index".format(name))


class TimestepEmbedder(nn.Module):
                                                                          

    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal_embedding(timesteps: Tensor, dimension: int, max_period: int = 10000) -> Tensor:
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half, 1)
        )
        arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=1)
        if dimension % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=1)
        return embedding

    def forward(self, timesteps: Tensor) -> Tensor:
        return self.mlp(self.sinusoidal_embedding(timesteps, self.frequency_size))


class JointConditioner(nn.Module):
                                                                      

    def __init__(self, config: CondDiffEEGConfig) -> None:
        super().__init__()
        self.num_classes = config.num_classes
        self.num_subjects = config.num_subjects
        self.dropout_probability = config.condition_dropout
        self.emotion_embedding = nn.Embedding(config.num_classes + 1, config.embed_dim)
        self.subject_embedding = nn.Embedding(config.num_subjects + 1, config.embed_dim)
        self.emotion_projection = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )
        self.subject_projection = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

    def _resolve_drop_mask(
        self,
        labels: Tensor,
        force_drop_mask: Optional[Tensor],
    ) -> Tensor:
        if force_drop_mask is not None:
            if force_drop_mask.shape != labels.shape:
                raise ValueError("force_drop_mask must have shape [batch]")
            return force_drop_mask.to(device=labels.device, dtype=torch.bool)
        if self.training and self.dropout_probability > 0.0:
            return torch.rand(labels.shape, device=labels.device) < self.dropout_probability
        return torch.zeros(labels.shape, device=labels.device, dtype=torch.bool)

    @staticmethod
    def _decorrelation(emotion: Tensor, subject: Tensor) -> Tensor:
        emotion_normalized = F.normalize(emotion, dim=-1, eps=1e-8)
        subject_normalized = F.normalize(subject, dim=-1, eps=1e-8)
        cross_similarity = emotion_normalized.transpose(0, 1) @ subject_normalized
        cross_similarity = cross_similarity / emotion.shape[0]
        return cross_similarity.square().sum()

    def forward(
        self,
        emotion_labels: Tensor,
        subject_labels: Tensor,
        force_drop_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, DenoiserAux]:
        _validate_label_range(emotion_labels, self.num_classes, "emotion_labels")
        _validate_label_range(subject_labels, self.num_subjects, "subject_labels")
        if emotion_labels.shape != subject_labels.shape:
            raise ValueError("emotion_labels and subject_labels must have the same shape")

        emotion_labels = emotion_labels.long()
        subject_labels = subject_labels.long()
        drop_mask = self._resolve_drop_mask(emotion_labels, force_drop_mask)

                                                                              
        raw_emotion = self.emotion_projection(self.emotion_embedding(emotion_labels))
        raw_subject = self.subject_projection(self.subject_embedding(subject_labels))

        used_emotion_labels = torch.where(drop_mask, self.num_classes, emotion_labels)
        used_subject_labels = torch.where(drop_mask, self.num_subjects, subject_labels)
        emotion = self.emotion_projection(self.emotion_embedding(used_emotion_labels))
        subject = self.subject_projection(self.subject_embedding(used_subject_labels))

        gate = torch.sigmoid(self.gate(torch.cat((emotion, subject), dim=-1)))
        semantic_condition = gate * emotion + (1.0 - gate) * subject
        aux = DenoiserAux(
            decorrelation=self._decorrelation(raw_emotion, raw_subject),
            condition_drop_mask=drop_mask,
            gate=gate,
            emotion_condition=raw_emotion,
            subject_condition=raw_subject,
        )
        return semantic_condition, aux


class MultiScalePatchEmbedding(nn.Module):
                                                                                

    def __init__(self, config: CondDiffEEGConfig) -> None:
        super().__init__()
        branch_size = config.embed_dim // len(config.multiscale_kernels)
        self.branches = nn.ModuleList(
            nn.Conv1d(
                config.in_channels,
                branch_size,
                kernel_size=kernel,
                padding=kernel // 2,
            )
            for kernel in config.multiscale_kernels
        )
        self.activation = nn.SiLU()
        self.patch_projection = nn.Conv1d(
            config.embed_dim,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        attention_hidden = max(config.embed_dim // 4, 1)
        self.channel_attention_norm = nn.LayerNorm(config.embed_dim)
        self.channel_attention = nn.Sequential(
            nn.Linear(config.embed_dim, attention_hidden, bias=False),
            nn.GELU(),
            nn.Linear(attention_hidden, config.embed_dim),
            nn.Sigmoid(),
        )
        self.refinement = nn.Sequential(
            nn.Conv1d(
                config.embed_dim,
                config.embed_dim,
                kernel_size=3,
                padding=1,
                groups=config.embed_dim,
            ),
            nn.SiLU(),
            nn.Conv1d(config.embed_dim, config.embed_dim, kernel_size=1),
        )

    def forward(self, signal: Tensor) -> Tensor:
        features = torch.cat([self.activation(branch(signal)) for branch in self.branches], dim=1)
        features = self.patch_projection(features)
        channel_summary = self.channel_attention_norm(features.mean(dim=-1))
        channel_weights = self.channel_attention(channel_summary).unsqueeze(-1)
        features = features * channel_weights
        features = features + self.refinement(features)
        return features.transpose(1, 2)


def _modulate(tokens: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return tokens * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNTransformerBlock(nn.Module):
                                                                             

    def __init__(self, config: CondDiffEEGConfig) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.embed_dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            config.embed_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(config.embed_dim, elementwise_affine=False, eps=1e-6)
        mlp_size = int(config.embed_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_size, config.embed_dim),
            nn.Dropout(config.dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.embed_dim, 6 * config.embed_dim),
        )

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.modulation(
            condition
        ).chunk(6, dim=-1)
        attention_input = _modulate(self.norm_attention(tokens), shift_attn, scale_attn)
        attention_output = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )[0]
        tokens = tokens + gate_attn.unsqueeze(1) * attention_output
        mlp_input = _modulate(self.norm_mlp(tokens), shift_mlp, scale_mlp)
        return tokens + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)


class CondDiffEEG(nn.Module):
                                                                        

                                                                           
                                                                                
                                                
       

    def __init__(self, config: Optional[CondDiffEEGConfig] = None, **overrides: object) -> None:
        super().__init__()
        if config is not None and overrides:
            raise ValueError("pass either config or keyword overrides, not both")
        self.config = config if config is not None else CondDiffEEGConfig(**overrides)
        config = self.config
        self.patch_count = config.time_points // config.patch_size

        self.patch_embedding = MultiScalePatchEmbedding(config)
        self.position_embedding = nn.Parameter(torch.empty(1, self.patch_count, config.embed_dim))
        self.timestep_embedding = TimestepEmbedder(config.embed_dim)
        self.conditioner = JointConditioner(config)
        self.blocks = nn.ModuleList(AdaLNTransformerBlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.embed_dim, 2 * config.embed_dim),
        )
        self.output_projection = nn.Linear(
            config.embed_dim,
            config.patch_size * config.in_channels,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.patch_embedding.channel_attention[2].weight)
        nn.init.constant_(self.patch_embedding.channel_attention[2].bias, 2.0)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.02)

    def _validate_input(
        self,
        signal: Tensor,
        timesteps: Tensor,
        emotion_labels: Tensor,
        subject_labels: Tensor,
    ) -> None:
        expected = (signal.shape[0], self.config.in_channels, self.config.time_points)
        if signal.shape != expected:
            raise ValueError("signal must have shape [B, {}, {}]".format(
                self.config.in_channels, self.config.time_points
            ))
        batch_shape = (signal.shape[0],)
        if timesteps.shape != batch_shape:
            raise ValueError("timesteps must have shape [batch]")
        if timesteps.numel() and (
            timesteps.min().item() < 0
            or timesteps.max().item() >= self.config.num_diffusion_steps
        ):
            raise ValueError("timesteps are outside the configured diffusion schedule")
        if emotion_labels.shape != batch_shape or subject_labels.shape != batch_shape:
            raise ValueError("condition labels must have shape [batch]")

    def forward(
        self,
        signal: Tensor,
        timesteps: Tensor,
        emotion_labels: Tensor,
        subject_labels: Tensor,
        *,
        force_drop_mask: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, DenoiserAux]]:
        self._validate_input(signal, timesteps, emotion_labels, subject_labels)
        tokens = self.patch_embedding(signal) + self.position_embedding
        time_condition = self.timestep_embedding(timesteps)
        semantic_condition, aux = self.conditioner(
            emotion_labels,
            subject_labels,
            force_drop_mask=force_drop_mask,
        )

                                                   
        condition = time_condition + semantic_condition
        for block in self.blocks:
            tokens = block(tokens, condition)

        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        tokens = _modulate(self.final_norm(tokens), shift, scale)
        patches = self.output_projection(tokens)
        batch_size = signal.shape[0]
        epsilon = patches.view(
            batch_size,
            self.patch_count,
            self.config.patch_size,
            self.config.in_channels,
        )
        epsilon = epsilon.permute(0, 3, 1, 2).contiguous()
        epsilon = epsilon.view(batch_size, self.config.in_channels, self.config.time_points)
        if self.config.input_skip_mode == "timestep_linear":
            skip_weight = timesteps.to(dtype=signal.dtype) / float(
                self.config.num_diffusion_steps - 1
            )
            epsilon = epsilon + skip_weight.view(-1, 1, 1) * signal
        if return_aux:
            return epsilon, aux
        return epsilon

    def forward_with_cfg(
        self,
        signal: Tensor,
        timesteps: Tensor,
        emotion_labels: Tensor,
        subject_labels: Tensor,
        cfg_scale: float,
    ) -> Tensor:
                                                                          

        batch_size = signal.shape[0]
        combined_signal = torch.cat((signal, signal), dim=0)
        combined_timesteps = torch.cat((timesteps, timesteps), dim=0)
        combined_emotion = torch.cat((emotion_labels, emotion_labels), dim=0)
        combined_subject = torch.cat((subject_labels, subject_labels), dim=0)
        drop_mask = torch.cat(
            (
                torch.zeros(batch_size, device=signal.device, dtype=torch.bool),
                torch.ones(batch_size, device=signal.device, dtype=torch.bool),
            ),
            dim=0,
        )
        output = self.forward(
            combined_signal,
            combined_timesteps,
            combined_emotion,
            combined_subject,
            force_drop_mask=drop_mask,
        )
        conditional, unconditional = output.chunk(2, dim=0)
        return unconditional + float(cfg_scale) * (conditional - unconditional)


__all__ = [
    "CondDiffEEG",
    "CondDiffEEGConfig",
    "DenoiserAux",
]
