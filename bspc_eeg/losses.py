                                                           

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class SpectralLossConfig:
                                                                     

                                                                             
                                                         
       

    sample_rate: float = 128.0
    bands: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 4.0, 0.5),
        (4.0, 8.0, 1.0),
        (8.0, 13.0, 2.5),
        (13.0, 30.0, 2.0),
        (30.0, 45.0, 0.5),
    )
    amplitude_weight: float = 1.0
    phase_weight: float = 1.5
    phase_mode: str = "amplitude_weighted"
    relative_amplitude_threshold: float = 0.05
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.phase_mode not in {"none", "unweighted", "amplitude_weighted"}:
            raise ValueError("invalid phase_mode")
        if not 0.0 <= self.relative_amplitude_threshold <= 1.0:
            raise ValueError("relative_amplitude_threshold must be in [0, 1]")
        if self.amplitude_weight < 0.0 or self.phase_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if not self.bands:
            raise ValueError("at least one frequency band is required")
        for low, high, weight in self.bands:
            if low < 0.0 or high <= low or weight < 0.0:
                raise ValueError("each band must have 0 <= low < high and weight >= 0")


@dataclass
class SpectralLossOutput:
                                                               

    total: Tensor
    amplitude: Tensor
    phase: Tensor
    active_phase_fraction: Tensor


def _frequency_weights(
    time_points: int,
    config: SpectralLossConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    frequencies = torch.fft.rfftfreq(time_points, d=1.0 / config.sample_rate).to(device=device)
    weights = torch.zeros_like(frequencies, dtype=dtype)
    for low, high, band_weight in config.bands:
        in_band = (frequencies >= low) & (frequencies < high)
        weights = weights + in_band.to(dtype=dtype) * float(band_weight)
    if not torch.any(weights > 0):
        raise ValueError("configured bands contain no FFT bins for this signal length")
    return weights


def spectral_reconstruction_loss(
    reference: Tensor,
    prediction: Tensor,
    config: Optional[SpectralLossConfig] = None,
) -> SpectralLossOutput:
                                                            

                                                                              
                                                                                

                                           
                                                  
       

    config = config or SpectralLossConfig()
    if reference.shape != prediction.shape or reference.ndim != 3:
        raise ValueError("reference and prediction must share shape [B, C, T]")
    if reference.shape[-1] < 2:
        raise ValueError("signals must contain at least two time points")

                                                       
    reference_fft = torch.fft.rfft(reference.detach(), dim=-1)
    prediction_fft = torch.fft.rfft(prediction, dim=-1)
    reference_amplitude = reference_fft.abs()
    prediction_amplitude = prediction_fft.abs()
    bin_weights = _frequency_weights(
        reference.shape[-1],
        config,
        prediction.device,
        prediction.dtype,
    ).view(1, 1, -1)

    amplitude_error = (
        torch.log1p(prediction_amplitude) - torch.log1p(reference_amplitude)
    ).abs()
    amplitude_denominator = bin_weights.sum().clamp_min(config.epsilon)
    amplitude_loss = (amplitude_error * bin_weights).sum(dim=-1) / amplitude_denominator
    amplitude_loss = amplitude_loss.mean()

    zero = prediction.sum() * 0.0
    if config.phase_mode == "none" or config.phase_weight == 0.0:
        phase_loss = zero
        active_fraction = zero.detach()
    else:
        phase_delta = torch.angle(prediction_fft) - torch.angle(reference_fft)
        circular_error = 1.0 - torch.cos(phase_delta)
        eligible = bin_weights > 0

        if config.phase_mode == "unweighted":
            phase_weights = bin_weights.expand_as(circular_error)
            phase_denominator = phase_weights.sum(dim=-1).clamp_min(config.epsilon)
            phase_loss = ((phase_weights * circular_error).sum(dim=-1) / phase_denominator).mean()
            active_fraction = eligible.to(prediction.dtype).mean()
        else:
            eligible_amplitude = torch.where(
                eligible, reference_amplitude, torch.zeros_like(reference_amplitude)
            )
            channel_maximum = eligible_amplitude.amax(dim=-1, keepdim=True)
            amplitude_mask = reference_amplitude >= (
                config.relative_amplitude_threshold * channel_maximum
            )
            amplitude_mask = amplitude_mask & eligible
            reference_power = reference_amplitude.square()
            phase_weights = (
                amplitude_mask.to(reference_power.dtype) * reference_power * bin_weights
            ).detach()
            phase_denominator = phase_weights.sum(dim=-1, keepdim=True)
            normalized_weights = phase_weights / phase_denominator.clamp_min(config.epsilon)
            per_channel = (normalized_weights * circular_error).sum(dim=-1)
            has_active_bin = phase_denominator.squeeze(-1) > config.epsilon
            if torch.any(has_active_bin):
                phase_loss = per_channel[has_active_bin].mean()
            else:
                phase_loss = zero
            eligible_count = eligible.expand_as(amplitude_mask).sum().clamp_min(1)
            active_fraction = amplitude_mask.sum().to(prediction.dtype) / eligible_count

    total = config.amplitude_weight * amplitude_loss + config.phase_weight * phase_loss
    return SpectralLossOutput(
        total=total,
        amplitude=amplitude_loss,
        phase=phase_loss,
        active_phase_fraction=active_fraction,
    )


__all__ = [
    "SpectralLossConfig",
    "SpectralLossOutput",
    "spectral_reconstruction_loss",
]
