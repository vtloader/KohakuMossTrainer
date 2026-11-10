"""`compile` backend: torch.compile of each op's eager reference (and of the eager
compositions of the fused linear ops), compiled once per op with dynamic shapes.

Registered for every block op; kernel backends also choose against it per shape
(`choice.py`). Its probe on a GPU is Triton's; on NPU it is the `npu` graph backend
(torchair); on CPU it needs the host C++ compiler.
See docs/internals/kernel-dev.md#compile-backend.
"""

import torch

from kohakumoss_kernels import status
from kohakumoss_kernels.backends.compile.ops import COMPILED, compiled_op
from kohakumoss_kernels.hardware.toolchain import ensure_toolchain, installed
from kohakumoss_kernels.registry import register

__all__ = ["COMPILED", "compiled_op"]


def check(device: torch.device) -> None:
    """Raise when inductor cannot generate code for `device` here."""
    if device.type == "cpu":
        toolchain = ensure_toolchain()
        if not toolchain.ok:
            raise RuntimeError(toolchain.detail)
        return
    if device.type == "npu":
        _check_npu()
        return
    if not installed("triton"):
        raise RuntimeError(f"inductor on {device.type} needs triton")
    why = status.probe("triton", device)
    if why is not None:
        raise RuntimeError(why)


def _check_npu() -> None:
    """Raise unless torch_npu's `npu` dynamo backend (torchair) compiles one tiny graph
    here. The probe compiles, so a torchair that imports but cannot lower fails too;
    torch_npu caches a failed `import torchair` and re-raises it, hence one probe per
    process (`status` caches the answer)."""
    try:
        import torch_npu  # noqa: F401
    except ImportError as err:
        raise RuntimeError(f"inductor on npu needs torch-npu: {err}") from err

    def f(a, b):
        return (a @ b).relu().sum()

    from kohakumoss_kernels import compile_setup

    compile_setup.apply()
    x = torch.randn(16, 16, device="npu", dtype=torch.float16)
    try:
        out = torch.compile(_fresh_for_probe(f), backend="npu")(x, x)
        out.item()
    except Exception as err:  # noqa: BLE001 - any failure means the backend cannot run
        raise RuntimeError(
            "torchair (torch.compiler backend 'npu') cannot compile on this host: "
            f"{type(err).__name__}: {err}"[:2000]
        ) from err


def _fresh_for_probe(fn):
    """A copy of `fn` with its own code object, so the probe's Dynamo cache is its
    own."""
    code = fn.__code__.replace(co_name=f"{fn.__code__.co_name}_probe")
    import types

    return types.FunctionType(
        code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__
    )


status.register_check("compile", check)
for _name in COMPILED:
    register(_name, "compile", compiled_op(_name))
