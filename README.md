# CondDiffEEG and CGODE-CCT Core Models

This repository contains the model-core implementation accompanying the paper on classification-aware conditional diffusion augmentation and CGODE-CCT EEG emotion recognition.

The release includes the conditional diffusion denoiser, the CGODE-CCT classifier, the parameter-efficient graph baseline, diffusion utilities, spectral reconstruction losses, and the alternating generator-classifier optimization pathway.

Dataset preparation, subject partitions, experiment orchestration, trained checkpoints, raw EEG recordings, generated samples, figures, logs, and local workstation paths are not included.

## Installation

```text
python -m pip install -r requirements.txt
```

## Core imports

```python
from bspc_eeg.diffusion import DiffusionConfig, GaussianDiffusion
from bspc_eeg.models import CGODECCTClassifier, CondDiffEEG, CondDiffEEGConfig, PatchGCNClassifier
from bspc_eeg.training import JointTrainer, JointTrainingConfig
```

## Minimal model construction

```python
import torch

from bspc_eeg.models import CGODECCTClassifier, CondDiffEEG, CondDiffEEGConfig

generator = CondDiffEEG(
    CondDiffEEGConfig(
        in_channels=32,
        time_points=1280,
        patch_size=4,
        embed_dim=512,
        depth=8,
        num_heads=16,
        num_classes=2,
        num_subjects=32,
    )
)

classifier = CGODECCTClassifier(
    num_classes=2,
    num_channels=32,
    input_length=1280,
    patch_size=128,
    patch_stride=64,
    d_model=256,
    cct_layers=12,
    num_heads=8,
    stem_layers=2,
    graph_mode="cgode",
)

signal = torch.randn(2, 32, 1280)
timesteps = torch.randint(0, 1000, (2,))
emotion = torch.tensor([0, 1])
subject = torch.tensor([0, 1])

noise_prediction = generator(signal, timesteps, emotion, subject)
logits = classifier(signal)
```

## Optimization pathway

`JointTrainer` implements the two update paths used by the paper. Detached multi-step samples update the classifier, while differentiable one-step clean estimates pass through a temporarily fixed classifier and provide task-aware gradients to the generator.

## License

Released under the MIT License. See `LICENSE`.
