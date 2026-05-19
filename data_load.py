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
    mesh_type: str = "rectangular"
    points: np.ndarray | None = None
    triangles: np.ndarray | None = None


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


def read_elem_triangles(base_path: str | Path) -> np.ndarray:
    """Read triangle connectivity from a CARP .elem file."""
    base_path = Path(base_path)
    elem_path = base_path if base_path.suffix == ".elem" else base_path.with_suffix(".elem")

    triangles = []
    with elem_path.open() as file:
        expected_count = int(file.readline().split()[0])
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] != "Tr":
                raise ValueError(f"Only triangular 2D elements are supported, found {parts[0]}.")
            triangles.append([int(parts[1]), int(parts[2]), int(parts[3])])

    triangles = np.asarray(triangles, dtype=np.int32)
    if triangles.shape[0] != expected_count:
        raise ValueError(
            f"Triangle count mismatch in {elem_path}: expected {expected_count}, "
            f"read {triangles.shape[0]}."
        )
    return triangles


def normalize_to_minus_one_one(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=np.float32)
    return (2.0 * (values - vmin) / (vmax - vmin) - 1.0).astype(np.float32)


def normalize_to_zero_one(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


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

    # openCARP point ordering is row-major in y-x for each z-slice: x changes
    # fastest within each y row. Reshape as (y, x, t), then transpose to the
    # project convention (x, y, t).
    vm = (
        vm_slice.T.reshape(y.shape[0], x.shape[0], t.shape[0])
        .transpose(1, 0, 2)
        .astype(np.float32)
    )

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
        mesh_type="rectangular",
    )


def load_opencarp_unstructured_2d_voltage(
    vm_path: str | Path,
    pts_path: str | Path,
    elem_path: str | Path | None = None,
    dt: float = 1.0,
    normalize_vm: bool = True,
    vm_min: float | None = None,
    vm_max: float | None = None,
) -> OpenCARPVoltageData:
    """Load openCARP voltage data on an unstructured 2D triangular mesh.

    Returns point-time samples:

        coords = [x, y, t]
        values = normalized vm

    The raw mesh coordinates are shifted to start at zero and converted from
    micrometers to millimeters, matching the rectangular loader's convention.
    """
    vm_all = read_array_igb(vm_path)
    points_um = read_pts(pts_path)

    if vm_all.shape[1] != points_um.shape[0]:
        raise ValueError(
            f"vm node count ({vm_all.shape[1]}) does not match pts count ({points_um.shape[0]})."
        )

    points_mm = ((points_um - np.min(points_um, axis=0)) / 1000.0).astype(np.float32)
    xy = points_mm[:, :2]
    t = (np.arange(vm_all.shape[0], dtype=np.float32) * dt).astype(np.float32)

    raw_vm_min = float(vm_all.min())
    raw_vm_max = float(vm_all.max())
    scale_min = raw_vm_min if vm_min is None else float(vm_min)
    scale_max = raw_vm_max if vm_max is None else float(vm_max)
    if normalize_vm:
        vm_nodes_time = normalize_to_zero_one(vm_all.T, scale_min, scale_max)
    else:
        vm_nodes_time = vm_all.T.astype(np.float32)

    coords = np.column_stack(
        [
            np.repeat(xy[:, 0], t.shape[0]),
            np.repeat(xy[:, 1], t.shape[0]),
            np.tile(t, xy.shape[0]),
        ]
    ).astype(np.float32)
    values = vm_nodes_time.reshape(-1, 1).astype(np.float32)

    bounds = {
        "x_min": float(xy[:, 0].min()),
        "x_max": float(xy[:, 0].max()),
        "y_min": float(xy[:, 1].min()),
        "y_max": float(xy[:, 1].max()),
        "t_min": float(t.min()),
        "t_max": float(t.max()),
        "z_selected": float(points_mm[:, 2].mean()),
        "vm_raw_min": raw_vm_min,
        "vm_raw_max": raw_vm_max,
        "vm_scale_min": scale_min,
        "vm_scale_max": scale_max,
        "vm_normalized": bool(normalize_vm),
    }

    coords_norm = np.column_stack(
        [
            normalize_to_minus_one_one(coords[:, 0], bounds["x_min"], bounds["x_max"]),
            normalize_to_minus_one_one(coords[:, 1], bounds["y_min"], bounds["y_max"]),
            normalize_to_minus_one_one(coords[:, 2], bounds["t_min"], bounds["t_max"]),
        ]
    ).astype(np.float32)

    triangles = read_elem_triangles(elem_path) if elem_path is not None else None

    return OpenCARPVoltageData(
        x=np.unique(xy[:, 0]).astype(np.float32),
        y=np.unique(xy[:, 1]).astype(np.float32),
        t=t,
        vm=vm_nodes_time,
        coords=coords,
        coords_norm=coords_norm,
        values=values,
        bounds=bounds,
        mesh_type="unstructured",
        points=xy,
        triangles=triangles,
    )


def load_ms_npz_2d_voltage(npz_path: str | Path) -> OpenCARPVoltageData:
    """Load generated Mitchell-Schaeffer 2D data from a compressed .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    t = data["t"].astype(np.float32)
    u = data["u"].astype(np.float32)

    if u.shape != (t.shape[0], x.shape[0], y.shape[0]):
        raise ValueError(
            "Expected u shape (time, x, y), got "
            f"{u.shape} for t={t.shape}, x={x.shape}, y={y.shape}."
        )

    vm = u.transpose(1, 2, 0).astype(np.float32)
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
        "z_selected": 0.0,
        "vm_raw_min": float(u.min()),
        "vm_raw_max": float(u.max()),
        "vm_scale_min": 0.0,
        "vm_scale_max": 1.0,
        "vm_normalized": True,
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
        mesh_type="rectangular",
    )


def subset_time_window(
    data: OpenCARPVoltageData,
    t_min: float | None = None,
    t_max: float | None = None,
) -> OpenCARPVoltageData:
    """Return a copy of loaded data restricted to a physical time window."""
    time_mask = np.ones(data.t.shape, dtype=bool)
    if t_min is not None:
        time_mask &= data.t >= t_min
    if t_max is not None:
        time_mask &= data.t <= t_max

    if not np.any(time_mask):
        raise ValueError(f"No time frames found in requested window [{t_min}, {t_max}].")

    t = data.t[time_mask].astype(np.float32)

    if data.mesh_type == "unstructured":
        if data.points is None:
            raise ValueError("Unstructured data is missing node coordinates.")
        vm = data.vm[:, time_mask].astype(np.float32)
        coords = np.column_stack(
            [
                np.repeat(data.points[:, 0], t.shape[0]),
                np.repeat(data.points[:, 1], t.shape[0]),
                np.tile(t, data.points.shape[0]),
            ]
        ).astype(np.float32)
        values = vm.reshape(-1, 1).astype(np.float32)
    else:
        vm = data.vm[:, :, time_mask].astype(np.float32)
        x_grid, y_grid, t_grid = np.meshgrid(data.x, data.y, t, indexing="ij")
        coords = np.column_stack(
            [x_grid.reshape(-1), y_grid.reshape(-1), t_grid.reshape(-1)]
        ).astype(np.float32)
        values = vm.reshape(-1, 1).astype(np.float32)

    bounds = data.bounds.copy()
    bounds["t_min"] = float(t.min())
    bounds["t_max"] = float(t.max())

    coords_norm = np.column_stack(
        [
            normalize_to_minus_one_one(coords[:, 0], bounds["x_min"], bounds["x_max"]),
            normalize_to_minus_one_one(coords[:, 1], bounds["y_min"], bounds["y_max"]),
            normalize_to_minus_one_one(coords[:, 2], bounds["t_min"], bounds["t_max"]),
        ]
    ).astype(np.float32)

    return OpenCARPVoltageData(
        x=data.x,
        y=data.y,
        t=t,
        vm=vm,
        coords=coords,
        coords_norm=coords_norm,
        values=values,
        bounds=bounds,
        mesh_type=data.mesh_type,
        points=data.points,
        triangles=data.triangles,
    )


def train_test_split_points(
    coords: np.ndarray,
    values: np.ndarray,
    train_fraction: float = 0.25,
    seed: int = 42,
    max_train_points: int | None = None,
    max_test_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_points = coords.shape[0]
    n_train = int(train_fraction * n_points)
    indices = rng.permutation(n_points)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    if max_train_points is not None and train_idx.shape[0] > max_train_points:
        train_idx = rng.choice(train_idx, size=max_train_points, replace=False)
    if max_test_points is not None and test_idx.shape[0] > max_test_points:
        test_idx = rng.choice(test_idx, size=max_test_points, replace=False)

    return coords[train_idx], values[train_idx], coords[test_idx], values[test_idx]
