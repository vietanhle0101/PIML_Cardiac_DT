"""
Load openCARP 2D Mitchell-Schaeffer voltage data for PyTorch PINN experiments.

The openCARP basic tissue example creates a thin 3D slab. For a 2D PINN, this
loader selects one z-slice, reads vm.igb, and returns point samples:

    coords = [x, y, t]
    values = vm

Coordinates are available both in physical units and normalized to [-1, 1].
"""

from __future__ import annotations

import argparse
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class OpenCARPVoltageData:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    vm: np.ndarray
    coords: np.ndarray
    coords_norm: np.ndarray
    values: np.ndarray
    bounds: dict


def read_array_igb(igb_path: str | Path) -> np.ndarray:
    """Read an openCARP .igb file into an array shaped (time, nodes)."""
    igb_path = Path(igb_path)
    data = []

    with igb_path.open("rb") as file:
        header = file.read(1024)
        words = header.split()
        dims = []
        for i in range(4):
            decoded = words[i].decode("utf-8")
            dims.append(int(re.split(r"(\d+)", decoded)[1]))

        n_nodes = dims[0] * dims[1] * dims[2]
        n_frames = os.path.getsize(igb_path) // 4 // n_nodes

        for _ in range(n_frames):
            frame = struct.unpack("f" * n_nodes, file.read(4 * n_nodes))
            data.append(frame)

    return np.asarray(data, dtype=np.float32)


def read_pts(base_path: str | Path) -> np.ndarray:
    """Read a CARP .pts file. Pass path with or without .pts suffix."""
    base_path = Path(base_path)
    pts_path = base_path if base_path.suffix == ".pts" else base_path.with_suffix(".pts")

    with pts_path.open() as file:
        count = int(file.readline().split()[0])
        points = np.empty((count, 3), dtype=np.float32)
        for i in range(count):
            points[i] = [float(v) for v in file.readline().split()[:3]]

    return points


def normalize_to_minus_one_one(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=np.float32)
    return (2.0 * (values - vmin) / (vmax - vmin) - 1.0).astype(np.float32)


def load_opencarp_2d_voltage(
    vm_path: str | Path,
    pts_path: str | Path,
    dt: float = 1.0,
    z_slice: str | float = "middle",
) -> OpenCARPVoltageData:
    """Load vm.igb and mesh points, selecting one z-slice from a thin slab."""
    vm_all = read_array_igb(vm_path)
    points_um = read_pts(pts_path)

    if vm_all.shape[1] != points_um.shape[0]:
        raise ValueError(
            f"vm node count ({vm_all.shape[1]}) does not match pts count ({points_um.shape[0]})."
        )

    points_mm = (points_um - np.min(points_um, axis=0)) / 1000.0
    z_values = np.unique(points_mm[:, 2])

    if z_values.shape[0] > 1:
        if z_slice == "middle":
            z_target = z_values[z_values.shape[0] // 2]
        else:
            z_target = float(z_slice)
        z_mask = np.isclose(points_mm[:, 2], z_target)
    else:
        z_target = z_values[0]
        z_mask = np.ones(points_mm.shape[0], dtype=bool)

    xy = points_mm[z_mask, :2]
    vm_slice = vm_all[:, z_mask]

    x = np.unique(xy[:, 0]).astype(np.float32)
    y = np.unique(xy[:, 1]).astype(np.float32)
    t = (np.arange(vm_slice.shape[0], dtype=np.float32) * dt).astype(np.float32)

    expected_nodes = x.shape[0] * y.shape[0]
    if expected_nodes != xy.shape[0]:
        raise ValueError(
            "Selected z-slice is not a complete rectangular x-y grid: "
            f"expected {expected_nodes} nodes from unique x/y, found {xy.shape[0]}."
        )

    vm = vm_slice.T.reshape(x.shape[0], y.shape[0], t.shape[0]).astype(np.float32)

    x_grid, y_grid, t_grid = np.meshgrid(x, y, t, indexing="ij")
    coords = np.column_stack(
        [x_grid.reshape(-1), y_grid.reshape(-1), t_grid.reshape(-1)]
    ).astype(np.float32)
    values = vm.reshape(-1, 1).astype(np.float32)

    bounds = {
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "t_min": float(t.min()),
        "t_max": float(t.max()),
        "z_selected": float(z_target),
    }

    coords_norm = np.column_stack(
        [
            normalize_to_minus_one_one(coords[:, 0], bounds["x_min"], bounds["x_max"]),
            normalize_to_minus_one_one(coords[:, 1], bounds["y_min"], bounds["y_max"]),
            normalize_to_minus_one_one(coords[:, 2], bounds["t_min"], bounds["t_max"]),
        ]
    ).astype(np.float32)

    return OpenCARPVoltageData(
        x=x,
        y=y,
        t=t,
        vm=vm,
        coords=coords,
        coords_norm=coords_norm,
        values=values,
        bounds=bounds,
    )


def train_test_split_points(
    coords: np.ndarray,
    values: np.ndarray,
    train_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_points = coords.shape[0]
    n_train = int(train_fraction * n_points)
    indices = rng.permutation(n_points)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return coords[train_idx], values[train_idx], coords[test_idx], values[test_idx]



