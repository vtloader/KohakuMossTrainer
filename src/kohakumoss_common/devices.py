"""Device-type facts every package shares (no torch import): the accelerator types with
side streams, events and pinned host memory."""

STREAMED = ("cuda", "xpu", "npu")
ORDER = ("cuda", "xpu", "npu")


def streamed_type(device_type: str) -> bool:
    """`device_type` ("cuda", "xpu", "npu", "cpu", ...) has side streams, events and
    pinned host memory."""
    return device_type in STREAMED


def accel_type(device_type: str) -> bool:
    """`device_type` is an accelerator (`STREAMED`)."""
    return device_type in STREAMED
