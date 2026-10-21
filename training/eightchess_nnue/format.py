"""Binary format read/write for the 8Chess NNUE `Parameters` image.

The engine deserializes the network with a raw `transmute` of the file
contents into a `#[repr(C)]` struct (see src/nnue.rs). This module is the
canonical specification of that layout: every offset is asserted, and
exported files are guaranteed byte-compatible with `EvalFile`.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    INPUT_BUCKETS,
    L1_SIZE,
    L2_SIZE,
    L3_SIZE,
    NETWORK_SIZE,
    OUTPUT_BUCKETS,
    PIECE_FEATURES,
    THREAT_FEATURES,
)

# (name, dtype, shape, byte offset) -- offsets must match src/nnue.rs exactly.
_LAYOUT = [
    ("ft_threat_weights", np.int8, (THREAT_FEATURES, L1_SIZE), 0),
    ("ft_piece_weights", np.int16, (PIECE_FEATURES, L1_SIZE), 68_468_736),
    ("ft_biases", np.int16, (L1_SIZE,), 84_197_376),
    ("l1_weights", np.int8, (OUTPUT_BUCKETS, L2_SIZE * L1_SIZE), 84_199_424),
    ("l1_biases", np.float32, (OUTPUT_BUCKETS, L2_SIZE), 84_330_496),
    ("l2_weights", np.float32, (OUTPUT_BUCKETS, L2_SIZE, L3_SIZE), 84_331_008),
    ("l2_biases", np.float32, (OUTPUT_BUCKETS, L3_SIZE), 84_347_392),
    ("l3_weights", np.float32, (OUTPUT_BUCKETS, L3_SIZE), 84_348_416),
    ("l3_biases", np.float32, (OUTPUT_BUCKETS,), 84_349_440),
]


def validate_layout() -> None:
    end = 0
    for name, dtype, shape, offset in _LAYOUT:
        assert offset == end or name == _LAYOUT[0][0], f"gap before {name}: {end} != {offset}"
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        # fields are 64-byte aligned; every field except the last is a multiple
        assert offset % 64 == 0, name
        end = offset + size
    assert end <= NETWORK_SIZE
    assert NETWORK_SIZE - end < 64  # trailing padding only
    assert NETWORK_SIZE == 84_349_504


validate_layout()


def read_network(path: str) -> dict[str, np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != NETWORK_SIZE:
        raise ValueError(f"invalid network size: expected {NETWORK_SIZE}, got {data.size}")

    net = {}
    for name, dtype, shape, offset in _LAYOUT:
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        raw = data[offset : offset + size]
        net[name] = raw.view(dtype).reshape(shape).copy()
    return net


def build_network(net: dict[str, np.ndarray]) -> bytes:
    buf = np.zeros(NETWORK_SIZE, dtype=np.uint8)

    for name, dtype, shape, offset in _LAYOUT:
        arr = np.ascontiguousarray(net[name], dtype=dtype)
        if arr.shape != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        buf[offset : offset + size] = arr.view(np.uint8).reshape(-1)

    return buf.tobytes()


def write_network(net: dict[str, np.ndarray], path: str) -> None:
    with open(path, "wb") as f:
        f.write(build_network(net))


def zero_network() -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, dtype=dtype)
        for name, dtype, shape, _ in _LAYOUT
    }
