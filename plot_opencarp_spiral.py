import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from data_load import read_array_igb, read_pts

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplconfig").resolve()))
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def read_elem_triangles(elem_path):
    elem_path = Path(elem_path)
    triangles = []
    with elem_path.open() as file:
        n_elem = int(file.readline().split()[0])
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] != "Tr":
                continue
            triangles.append([int(parts[1]), int(parts[2]), int(parts[3])])

    triangles = np.asarray(triangles, dtype=np.int32)
    if triangles.shape[0] != n_elem:
        print(f"warning: expected {n_elem} triangles, read {triangles.shape[0]}")
    return triangles


def make_triangulation(pts_path, elem_path):
    points_um = read_pts(pts_path)
    points_mm = points_um / 1000.0
    triangles = read_elem_triangles(elem_path)
    return points_mm, mtri.Triangulation(points_mm[:, 0], points_mm[:, 1], triangles)


def draw_voltage_frame(
    triangulation,
    vm_frame,
    time_ms,
    output_path,
    vmin=-85.0,
    vmax=35.0,
    title=None,
):
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    image = ax.tripcolor(
        triangulation,
        vm_frame,
        shading="gouraud",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title or f"openCARP reentry Vm, t={time_ms:.0f} ms")
    ax.set_xlim(float(triangulation.x.min()), float(triangulation.x.max()))
    ax.set_ylim(float(triangulation.y.min()), float(triangulation.y.max()))
    colorbar = fig.colorbar(image, ax=ax, label="Vm (mV)")
    colorbar.set_ticks([-80, -40, 0, 30])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_snapshot_grid(triangulation, vm, frame_indices, output_path, vmin=-85.0, vmax=35.0):
    n_cols = min(3, len(frame_indices))
    n_rows = int(np.ceil(len(frame_indices) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 4.1 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    image = None
    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, frame_index in zip(axes.ravel(), frame_indices):
        ax.set_visible(True)
        image = ax.tripcolor(
            triangulation,
            vm[frame_index],
            shading="gouraud",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect("equal")
        ax.set_title(f"t={frame_index} ms")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_xlim(float(triangulation.x.min()), float(triangulation.x.max()))
        ax.set_ylim(float(triangulation.y.min()), float(triangulation.y.max()))

    fig.colorbar(image, ax=axes.ravel().tolist(), label="Vm (mV)")
    fig.suptitle("openCARP RP_E reentry / spiral-wave snapshots")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_gif(frame_paths, output_path, duration_ms):
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    for image in images:
        image.close()


def main():
    parser = argparse.ArgumentParser(description="Plot openCARP RP_E spiral/reentry voltage data.")
    parser.add_argument("--data-dir", default="Data/RP_E/Spiral")
    parser.add_argument("--output-dir", default="outputs/opencarp_spiral")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--num-gif-frames", type=int, default=90)
    parser.add_argument("--gif-duration-ms", type=int, default=70)
    parser.add_argument("--vmin", type=float, default=-85.0)
    parser.add_argument("--vmax", type=float, default=35.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    vm = read_array_igb(data_dir / "vm.igb")
    points, triangulation = make_triangulation(data_dir / "block.pts", data_dir / "block.elem")

    end_frame = args.end_frame if args.end_frame is not None else vm.shape[0] - 1
    end_frame = min(end_frame, vm.shape[0] - 1)
    start_frame = max(args.start_frame, 0)
    if start_frame >= end_frame:
        raise ValueError("start-frame must be smaller than end-frame")

    gif_indices = np.linspace(start_frame, end_frame, args.num_gif_frames, dtype=int)
    snapshot_indices = np.linspace(start_frame, end_frame, 6, dtype=int).tolist()

    print("openCARP spiral/reentry dataset")
    print(f"points: {points.shape[0]}")
    print(f"triangles: {triangulation.triangles.shape[0]}")
    print(f"vm shape: {vm.shape}")
    print(f"vm range: [{float(vm.min()):.3f}, {float(vm.max()):.3f}] mV")
    print(f"frames: {start_frame}..{end_frame}")

    frame_paths = []
    for frame_index in gif_indices:
        frame_path = frame_dir / f"vm_{frame_index:04d}.png"
        draw_voltage_frame(
            triangulation,
            vm[frame_index],
            frame_index,
            frame_path,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        frame_paths.append(frame_path)

    plot_snapshot_grid(
        triangulation,
        vm,
        snapshot_indices,
        output_dir / "spiral_wave_snapshots.png",
        vmin=args.vmin,
        vmax=args.vmax,
    )
    draw_voltage_frame(
        triangulation,
        vm[snapshot_indices[len(snapshot_indices) // 2]],
        snapshot_indices[len(snapshot_indices) // 2],
        output_dir / "spiral_wave_single_frame.png",
        vmin=args.vmin,
        vmax=args.vmax,
    )
    save_gif(frame_paths, output_dir / "spiral_wave_vm.gif", args.gif_duration_ms)

    print(f"saved snapshots: {output_dir / 'spiral_wave_snapshots.png'}")
    print(f"saved single frame: {output_dir / 'spiral_wave_single_frame.png'}")
    print(f"saved gif: {output_dir / 'spiral_wave_vm.gif'}")


if __name__ == "__main__":
    main()
