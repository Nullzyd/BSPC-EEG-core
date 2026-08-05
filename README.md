# CondDiffEEG and CGODE-CCT Core Models

This repository releases only the core PyTorch model definitions used in the paper.

## Contents

- `bspc_eeg/models/cond_diff_eeg.py`: CondDiffEEG conditional diffusion denoiser, gated emotion-subject conditioning, timestep modulation, multi-scale temporal encoding, and DiT blocks.
- `bspc_eeg/models/cgode_cct.py`: CGODE-CCT classifier, dual-domain patch embedding, criss-cross attention, dynamic graph ODE with fixed-step RK4 integration, and STEM refinement.

Dataset processing, diffusion schedules, samplers, reconstruction losses, joint-training orchestration, classifier baselines, checkpoints, generated EEG, figures, logs, and local paths are not included.

## Model settings

The released defaults follow the revised manuscript. CondDiffEEG uses an embedding dimension of 512, 8 DiT blocks, 16 attention heads, an FFN expansion ratio of 4, overlapping generator patches with size 4 and stride 2, multi-scale kernels `(1, 3, 7, 15)`, and condition dropout 0.15. CGODE-CCT uses a hidden dimension of 256, 12 criss-cross blocks, 8 attention heads, 2 STEM layers, dropout 0.1, classification-head dropout 0.3, and fixed-step RK4 integration over `[0, 1]` with step size 0.25 and nonnegative decay initialization 0.1. Dataset-specific classifier patch settings are `128/64` for DEAP and DREAMER and `200/100` for SEED.

## Core imports

```python
from bspc_eeg.models.cond_diff_eeg import CondDiffEEG, CondDiffEEGConfig
from bspc_eeg.models.cgode_cct import CGODECCTClassifier
```

The model interfaces accept standardized EEG tensors with shape `[batch, channels, time]`. The generator additionally receives diffusion timesteps, emotion labels, and subject identifiers.
