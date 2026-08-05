                                                               

                                                                

                                                                          
                                                                           
                                                     
   

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from bspc_eeg.diffusion import GaussianDiffusion
from bspc_eeg.losses import (
    SpectralLossConfig,
    SpectralLossOutput,
    spectral_reconstruction_loss,
)
from bspc_eeg.models.cond_diff_eeg import CondDiffEEG


REAL_ONLY = "real_only"
OFFLINE = "offline"
CLASSIFICATION_AWARE = "classification_aware"
VALID_MODES = {REAL_ONLY, OFFLINE, CLASSIFICATION_AWARE}


@dataclass(frozen=True)
class JointTrainingConfig:
                                                                            

    mode: str = CLASSIFICATION_AWARE
    noise_weight: float = 1.0
    reconstruction_weight: float = 0.5
    spectral_weight: float = 0.1
    decorrelation_weight: float = 0.05
    guidance_weight: float = 0.1
    synthetic_classifier_weight: float = 1.0
    guidance_warmup_steps: int = 0
    guidance_max_timestep: int = 200
    spectral_clarity_weighting: bool = True
    spectral_max_timestep: Optional[int] = None
    reconstruction_snr_weighting: bool = True
    noise_min_snr_gamma: Optional[float] = None
    gradient_clip_norm: Optional[float] = 1.0
    verify_guidance_gradients: bool = True

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError("mode must be one of {}".format(sorted(VALID_MODES)))
        for name in (
            "noise_weight",
            "reconstruction_weight",
            "spectral_weight",
            "decorrelation_weight",
            "guidance_weight",
            "synthetic_classifier_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError("{} must be non-negative".format(name))
        if self.guidance_warmup_steps < 0 or self.guidance_max_timestep < 0:
            raise ValueError("guidance step settings must be non-negative")
        if self.spectral_max_timestep is not None and self.spectral_max_timestep < 0:
            raise ValueError("spectral_max_timestep must be non-negative or None")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if self.noise_min_snr_gamma is not None and self.noise_min_snr_gamma <= 0.0:
            raise ValueError("noise_min_snr_gamma must be positive or None")


@dataclass
class GeneratorStepOutput:
                                                                          

    total_loss: Tensor
    noise_loss: Tensor
    noise_weight_mean: Tensor
    reconstruction_loss: Tensor
    reconstruction_weight_mean: Tensor
    amplitude_loss: Tensor
    phase_loss: Tensor
    decorrelation_loss: Tensor
    guidance_loss: Tensor
    guidance_grad_norm: Tensor
    generator_grad_norm: Tensor
    spectral_clarity: Tensor
    spectral_active_fraction: Tensor
    active_phase_fraction: Tensor
    x0_hat: Tensor
    guidance_active: bool


@dataclass
class ClassifierStepOutput:
                                                                           

    total_loss: Tensor
    real_loss: Tensor
    synthetic_loss: Tensor
    used_synthetic: bool


def _clear_parameter_grads(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.grad = None


def _gradient_norm(gradients: Iterator[Optional[Tensor]], device: torch.device) -> Tensor:
    squared_norm = torch.zeros((), device=device)
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + gradient.detach().float().square().sum()
    return squared_norm.sqrt()


@contextmanager
def frozen_for_input_gradients(module: nn.Module) -> Iterator[None]:
                                                                               

                                                                              
                                                                                
                                           
       

    requirements = [parameter.requires_grad for parameter in module.parameters()]
    was_training = module.training
    _clear_parameter_grads(module)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    try:
        yield
    finally:
        for parameter, requirement in zip(module.parameters(), requirements):
            parameter.requires_grad_(requirement)
        module.train(was_training)


class JointTrainer:
                                                      

         
                                                               
                                                                                
                                                    
                                                                                    
                                                               
                                                                                  
                                                  
       

    def __init__(
        self,
        generator: CondDiffEEG,
        classifier: nn.Module,
        diffusion: GaussianDiffusion,
        *,
        generator_optimizer: Optional[torch.optim.Optimizer],
        classifier_optimizer: torch.optim.Optimizer,
        config: Optional[JointTrainingConfig] = None,
        spectral_config: Optional[SpectralLossConfig] = None,
        classifier_input_transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        self.generator = generator
        self.classifier = classifier
        self.diffusion = diffusion
        self.generator_optimizer = generator_optimizer
        self.classifier_optimizer = classifier_optimizer
        self.config = config or JointTrainingConfig()
        self.spectral_config = spectral_config or SpectralLossConfig()
        self.classifier_input_transform = classifier_input_transform or (lambda value: value)
        self._guidance_gradient_verified = False
        if self.config.guidance_max_timestep >= diffusion.num_train_steps:
            raise ValueError("guidance_max_timestep must be smaller than num_train_steps")
        if (
            self.config.spectral_max_timestep is not None
            and self.config.spectral_max_timestep >= diffusion.num_train_steps
        ):
            raise ValueError("spectral_max_timestep must be smaller than num_train_steps")

    @property
    def mode(self) -> str:
        return self.config.mode

    def _sample_timesteps(self, batch_size: int, maximum: int, device: torch.device) -> Tensor:
        return torch.randint(0, maximum + 1, (batch_size,), device=device)

    def _guidance_is_active(self, global_step: int) -> bool:
        return (
            self.mode == CLASSIFICATION_AWARE
            and self.config.guidance_weight > 0.0
            and global_step >= self.config.guidance_warmup_steps
        )

    def generator_step(
        self,
        clean_signal: Tensor,
        emotion_labels: Tensor,
        subject_labels: Tensor,
        *,
        global_step: int = 0,
        diffusion_timesteps: Optional[Tensor] = None,
        diffusion_noise: Optional[Tensor] = None,
        guidance_timesteps: Optional[Tensor] = None,
        guidance_noise: Optional[Tensor] = None,
    ) -> GeneratorStepOutput:
                                                                                   

        if self.mode == REAL_ONLY:
            raise RuntimeError("the real_only ablation does not train or use a generator")
        if self.generator_optimizer is None:
            raise RuntimeError("generator_optimizer is required for generator_step")
        if global_step < 0:
            raise ValueError("global_step must be non-negative")

        device = clean_signal.device
        batch_size = clean_signal.shape[0]
        emotion_labels = emotion_labels.to(device=device, dtype=torch.long)
        subject_labels = subject_labels.to(device=device, dtype=torch.long)
        self.generator.train()
        self.generator_optimizer.zero_grad(set_to_none=True)
        _clear_parameter_grads(self.classifier)

        if diffusion_timesteps is None:
            diffusion_timesteps = self._sample_timesteps(
                batch_size,
                self.diffusion.num_train_steps - 1,
                device,
            )
        else:
            diffusion_timesteps = diffusion_timesteps.to(device=device, dtype=torch.long)
        if diffusion_noise is None:
            diffusion_noise = torch.randn_like(clean_signal)
        noisy_signal = self.diffusion.q_sample(
            clean_signal,
            diffusion_timesteps,
            diffusion_noise,
        )
        predicted_noise, aux = self.generator(
            noisy_signal,
            diffusion_timesteps,
            emotion_labels,
            subject_labels,
            return_aux=True,
        )
        noise_error = (predicted_noise - diffusion_noise).square().mean(dim=(1, 2))
        if self.config.noise_min_snr_gamma is None:
            noise_weights = torch.ones_like(noise_error)
        else:
            snr = self.diffusion.signal_to_noise_ratio(diffusion_timesteps).to(
                dtype=noise_error.dtype
            )
            gamma = torch.full_like(snr, self.config.noise_min_snr_gamma)
            noise_weights = torch.minimum(snr, gamma) / snr.clamp_min(1e-12)
        noise_loss = (noise_weights * noise_error).mean()
        diffusion_x0_hat = self.diffusion.predict_x0(
            noisy_signal,
            diffusion_timesteps,
            predicted_noise,
        )
        reconstruction_error = (diffusion_x0_hat - clean_signal).square().mean(
            dim=(1, 2)
        )
        if self.config.reconstruction_snr_weighting:
            reconstruction_weights = self.diffusion.alphas_cumprod[
                diffusion_timesteps
            ].to(device=device, dtype=reconstruction_error.dtype)
        else:
            reconstruction_weights = torch.ones_like(reconstruction_error)
        reconstruction_loss = (reconstruction_weights * reconstruction_error).mean()
        if self.config.spectral_max_timestep is not None:
            spectral_mask = diffusion_timesteps <= self.config.spectral_max_timestep
            spectral_active_fraction = spectral_mask.to(dtype=clean_signal.dtype).mean()
            if torch.any(spectral_mask):
                clarity = torch.ones((), device=device)
            else:
                clarity = torch.zeros((), device=device)
        else:
            spectral_mask = torch.ones_like(diffusion_timesteps, dtype=torch.bool)
            spectral_active_fraction = torch.ones((), device=device)
            if self.config.spectral_clarity_weighting:
                clarity = (
                    1.0
                    - diffusion_timesteps.float()
                    / float(max(self.diffusion.num_train_steps - 1, 1))
                ).mean()
            else:
                clarity = torch.ones((), device=device)
        if torch.any(spectral_mask):
            spectral = spectral_reconstruction_loss(
                clean_signal[spectral_mask],
                diffusion_x0_hat[spectral_mask],
                self.spectral_config,
            )
        else:
            zero_spectral = predicted_noise.sum() * 0.0
            spectral = SpectralLossOutput(
                total=zero_spectral,
                amplitude=zero_spectral,
                phase=zero_spectral,
                active_phase_fraction=zero_spectral.detach(),
            )

        guidance_active = self._guidance_is_active(global_step)
        zero = predicted_noise.sum() * 0.0
        guidance_loss = zero
        guidance_grad_norm = zero.detach()
        guided_x0_hat = diffusion_x0_hat

        if guidance_active:
            if guidance_timesteps is None:
                guidance_timesteps = self._sample_timesteps(
                    batch_size,
                    self.config.guidance_max_timestep,
                    device,
                )
            else:
                guidance_timesteps = guidance_timesteps.to(device=device, dtype=torch.long)
                if guidance_timesteps.numel() and (
                    guidance_timesteps.max().item() > self.config.guidance_max_timestep
                ):
                    raise ValueError("guidance_timesteps exceed guidance_max_timestep")
            if guidance_noise is None:
                guidance_noise = torch.randn_like(clean_signal)
            guidance_input = self.diffusion.q_sample(
                clean_signal,
                guidance_timesteps,
                guidance_noise,
            )
            all_conditions_present = torch.zeros(
                batch_size,
                device=device,
                dtype=torch.bool,
            )
            guidance_prediction = self.generator(
                guidance_input,
                guidance_timesteps,
                emotion_labels,
                subject_labels,
                force_drop_mask=all_conditions_present,
            )
            guided_x0_hat = self.diffusion.predict_x0(
                guidance_input,
                guidance_timesteps,
                guidance_prediction,
            )
            with frozen_for_input_gradients(self.classifier):
                classifier_estimate = guided_x0_hat
                if self.diffusion.config.clip_denoised:
                    classifier_estimate = classifier_estimate.clamp(
                        *self.diffusion.config.clip_range
                    )
                classifier_input = self.classifier_input_transform(classifier_estimate)
                logits = self.classifier(classifier_input)
                guidance_loss = F.cross_entropy(logits, emotion_labels)

            if (
                self.config.verify_guidance_gradients
                and not self._guidance_gradient_verified
            ):
                parameters = [
                    parameter for parameter in self.generator.parameters() if parameter.requires_grad
                ]
                guidance_gradients = torch.autograd.grad(
                    guidance_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                guidance_grad_norm = _gradient_norm(iter(guidance_gradients), device)
                if not torch.isfinite(guidance_grad_norm) or guidance_grad_norm.item() <= 0.0:
                    raise RuntimeError("guidance loss produced no finite generator gradient")
                self._guidance_gradient_verified = True

        total_loss = (
            self.config.noise_weight * noise_loss
            + self.config.reconstruction_weight * reconstruction_loss
            + self.config.spectral_weight * clarity * spectral.total
            + self.config.decorrelation_weight * aux.decorrelation
            + self.config.guidance_weight * guidance_loss
        )
        total_loss.backward()

        if any(parameter.grad is not None for parameter in self.classifier.parameters()):
            raise RuntimeError("classifier parameters received gradients during generator_step")
        generator_grad_norm = _gradient_norm(
            (parameter.grad for parameter in self.generator.parameters()),
            device,
        )
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.config.gradient_clip_norm,
            )
        self.generator_optimizer.step()

        return GeneratorStepOutput(
            total_loss=total_loss.detach(),
            noise_loss=noise_loss.detach(),
            noise_weight_mean=noise_weights.mean().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            reconstruction_weight_mean=reconstruction_weights.mean().detach(),
            amplitude_loss=spectral.amplitude.detach(),
            phase_loss=spectral.phase.detach(),
            decorrelation_loss=aux.decorrelation.detach(),
            guidance_loss=guidance_loss.detach(),
            guidance_grad_norm=guidance_grad_norm.detach(),
            generator_grad_norm=generator_grad_norm.detach(),
            spectral_clarity=clarity.detach(),
            spectral_active_fraction=spectral_active_fraction.detach(),
            active_phase_fraction=spectral.active_phase_fraction.detach(),
            x0_hat=guided_x0_hat.detach(),
            guidance_active=guidance_active,
        )

    def classifier_step(
        self,
        real_signal: Tensor,
        real_labels: Tensor,
        *,
        synthetic_signal: Optional[Tensor] = None,
        synthetic_labels: Optional[Tensor] = None,
    ) -> ClassifierStepOutput:
                                                                                   

        device = real_signal.device
        real_labels = real_labels.to(device=device, dtype=torch.long)
        self.classifier.train()
        self.classifier_optimizer.zero_grad(set_to_none=True)
        _clear_parameter_grads(self.generator)

        real_logits = self.classifier(self.classifier_input_transform(real_signal))
        real_loss = F.cross_entropy(real_logits, real_labels)
        zero = real_loss * 0.0
        synthetic_loss = zero
        used_synthetic = False

        if self.mode != REAL_ONLY and synthetic_signal is not None:
            if synthetic_labels is None:
                raise ValueError("synthetic_labels are required with synthetic_signal")
            detached_synthetic = synthetic_signal.detach().to(device=device)
            synthetic_labels = synthetic_labels.to(device=device, dtype=torch.long)
            synthetic_logits = self.classifier(
                self.classifier_input_transform(detached_synthetic)
            )
            synthetic_loss = F.cross_entropy(synthetic_logits, synthetic_labels)
            used_synthetic = True
        elif synthetic_labels is not None and synthetic_signal is None:
            raise ValueError("synthetic_signal is required with synthetic_labels")

        total_loss = real_loss + self.config.synthetic_classifier_weight * synthetic_loss
        total_loss.backward()
        self.classifier_optimizer.step()

        return ClassifierStepOutput(
            total_loss=total_loss.detach(),
            real_loss=real_loss.detach(),
            synthetic_loss=synthetic_loss.detach(),
            used_synthetic=used_synthetic,
        )

    def sample_synthetic(
        self,
        emotion_labels: Tensor,
        subject_labels: Tensor,
        *,
        noise: Optional[Tensor] = None,
        num_sample_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
    ) -> Tensor:
                                                                                 

        if self.mode == REAL_ONLY:
            raise RuntimeError("the real_only ablation does not generate synthetic data")
        shape = (
            emotion_labels.shape[0],
            self.generator.config.in_channels,
            self.generator.config.time_points,
        )
        return self.diffusion.ddim_sample(
            self.generator,
            shape,
            emotion_labels,
            subject_labels,
            noise=noise,
            num_sample_steps=num_sample_steps,
            cfg_scale=cfg_scale,
        )


__all__ = [
    "CLASSIFICATION_AWARE",
    "OFFLINE",
    "REAL_ONLY",
    "ClassifierStepOutput",
    "GeneratorStepOutput",
    "JointTrainer",
    "JointTrainingConfig",
    "frozen_for_input_gradients",
]
