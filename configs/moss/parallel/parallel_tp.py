"""Parallel layer for scripts/train/moss.py: tensor parallel over two devices (Linux;
nccl / hccl by the accelerator). Each rank holds a slice of every block's q / k / v /
o / up-gate / down projection and runs the same tokens, so the batch and the data plan
are one rank's. Replaces the VRAM layer's profile: resident weights, per-step optimiser.
See docs/guides/configs.md and docs/guides/multi-gpu.md (tensor parallel)."""

PARALLEL = "tp"
DEVICES = (0, 1)
SAMPLE_EVERY = 0
VALIDATE_EVERY = 0
MEMORY_PROFILE = "ckpt-resident"
MEMORY_OVERRIDES = {}
