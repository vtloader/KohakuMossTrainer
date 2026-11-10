"""Device parameters for model-free tests: CPU always, CUDA / XPU / NPU behind their
markers."""

import pytest

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.cuda),
    pytest.param("xpu", marks=pytest.mark.xpu),
    pytest.param("npu", marks=pytest.mark.npu),
]
