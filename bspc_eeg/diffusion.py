                                                                            

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DiffusionConfig:
                                             

                                                                                
                                                     
       

    num_train_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    num_sample_steps: int = 100
    cfg_scale: float = 4.0
    clip_denoised: bool = True
    clip_range: Tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self) -> None:
        if self.num_train_steps < 2:
            raise ValueError("num_train_steps must be at least 2")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
        if not 1 <= self.num_sample_steps <= self.num_train_steps:
            raise ValueError("num_sample_steps must be in [1, num_train_steps]")
        if self.clip_range[0] >= self.clip_range[1]:
            raise ValueError("clip_range must be increasing")


def _extract(values: Tensor, timesteps: Tensor, reference: Tensor) -> Tensor:
    if timesteps.ndim != 1 or timesteps.shape[0] != reference.shape[0]:
        raise ValueError("timesteps must have shape [batch]")
    selected = values.to(device=timesteps.device)[timesteps.long()].to(dtype=reference.dtype)
    return selected.view(selected.shape[0], *([1] * (reference.ndim - 1)))


class GaussianDiffusion(nn.Module):
                                                          

                                                                              
                                                                                 
                                                    
       

    def __init__(self, config: Optional[DiffusionConfig] = None) -> None:
        super().__init__()
        self.config = config or DiffusionConfig()
        betas = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.num_train_steps,
            dtype=torch.float64,
        )
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alpha_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", alpha_cumprod.sqrt().float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alpha_cumprod).sqrt().float())
        self.register_buffer("sqrt_recip_alphas_cumprod", alpha_cumprod.rsqrt().float())
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            (1.0 / alpha_cumprod - 1.0).sqrt().float(),
        )

    @property
    def num_train_steps(self) -> int:
        return self.config.num_train_steps

    def _validate_timesteps(self, timesteps: Tensor) -> None:
        if timesteps.numel() and (
            timesteps.min().item() < 0 or timesteps.max().item() >= self.num_train_steps
        ):
            raise ValueError("timesteps are outside the diffusion schedule")

    def q_sample(
        self,
        clean_signal: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
                                                                               

        self._validate_timesteps(timesteps)
        if noise is None:
            noise = torch.randn_like(clean_signal)
        if noise.shape != clean_signal.shape:
            raise ValueError("noise and clean_signal must have the same shape")
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, clean_signal) * clean_signal
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_signal) * noise
        )

    def predict_x0(self, noisy_signal: Tensor, timesteps: Tensor, predicted_noise: Tensor) -> Tensor:
                                                                            

        self._validate_timesteps(timesteps)
        if noisy_signal.shape != predicted_noise.shape:
            raise ValueError("noisy_signal and predicted_noise must have the same shape")
        return (
            _extract(self.sqrt_recip_alphas_cumprod, timesteps, noisy_signal) * noisy_signal
            - _extract(self.sqrt_recipm1_alphas_cumprod, timesteps, noisy_signal)
            * predicted_noise
        )

    def predict_noise(self, noisy_signal: Tensor, timesteps: Tensor, predicted_x0: Tensor) -> Tensor:
                                                                   

        self._validate_timesteps(timesteps)
        return (
            noisy_signal
            - _extract(self.sqrt_alphas_cumprod, timesteps, noisy_signal) * predicted_x0
        ) / _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_signal).clamp_min(1e-12)

    def signal_to_noise_ratio(self, timesteps: Tensor) -> Tensor:
                                                                           

        self._validate_timesteps(timesteps)
        alpha = self.alphas_cumprod.to(timesteps.device)[timesteps.long()]
        return alpha / (1.0 - alpha).clamp_min(1e-12)

    def sampling_timesteps(self, num_sample_steps: Optional[int] = None) -> Tensor:
                                                                 

        step_count = num_sample_steps or self.config.num_sample_steps
        if not 1 <= step_count <= self.num_train_steps:
            raise ValueError("num_sample_steps must be in [1, num_train_steps]")
        timesteps = torch.linspace(
            self.num_train_steps - 1,
            0,
            step_count,
            dtype=torch.float64,
        ).round().long()
        return torch.unique_consecutive(timesteps)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Sequence[int],
        emotion_labels: Tensor,
        subject_labels: Tensor,
        *,
        noise: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        num_sample_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        clip_denoised: Optional[bool] = None,
    ) -> Tensor:
                                                         

                                                                             
                                                                           
                      
           

        shape = tuple(int(value) for value in shape)
        if len(shape) != 3 or shape[0] != emotion_labels.shape[0]:
            raise ValueError("shape must be [B, C, T] and match the labels")
        if emotion_labels.shape != subject_labels.shape:
            raise ValueError("emotion_labels and subject_labels must have the same shape")
        try:
            device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("model must contain parameters") from error

        emotion_labels = emotion_labels.to(device=device, dtype=torch.long)
        subject_labels = subject_labels.to(device=device, dtype=torch.long)
        if noise is None:
            sample = torch.randn(shape, device=device, generator=generator)
        else:
            if tuple(noise.shape) != shape:
                raise ValueError("noise does not match the requested shape")
            sample = noise.to(device=device)

        scale = self.config.cfg_scale if cfg_scale is None else float(cfg_scale)
        should_clip = self.config.clip_denoised if clip_denoised is None else clip_denoised
        schedule = self.sampling_timesteps(num_sample_steps).to(device)
        was_training = model.training
        model.eval()
        try:
            for index, timestep_value in enumerate(schedule):
                timesteps = torch.full(
                    (shape[0],),
                    int(timestep_value.item()),
                    device=device,
                    dtype=torch.long,
                )
                predicted_noise = model.forward_with_cfg(
                    sample,
                    timesteps,
                    emotion_labels,
                    subject_labels,
                    scale,
                )
                predicted_x0 = self.predict_x0(sample, timesteps, predicted_noise)
                if should_clip:
                    predicted_x0 = predicted_x0.clamp(*self.config.clip_range)
                    predicted_noise = self.predict_noise(sample, timesteps, predicted_x0)

                if index + 1 == schedule.numel():
                    alpha_previous = torch.ones(
                        (shape[0], 1, 1),
                        device=device,
                        dtype=sample.dtype,
                    )
                else:
                    previous = torch.full_like(timesteps, int(schedule[index + 1].item()))
                    alpha_previous = _extract(self.alphas_cumprod, previous, sample)
                sample = (
                    alpha_previous.sqrt() * predicted_x0
                    + (1.0 - alpha_previous).clamp_min(0.0).sqrt() * predicted_noise
                )
        finally:
            model.train(was_training)
        return sample.detach()


__all__ = ["DiffusionConfig", "GaussianDiffusion"]
