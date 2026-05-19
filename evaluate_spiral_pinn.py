import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from data_load import load_opencarp_unstructured_2d_voltage, subset_time_window
from model import MLP_Net

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplconfig").resolve()))
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def load_config(config_path):
    with Path(config_path).open() as file:
        return yaml.safe_load(file)


def config_path(base_path, file_name):
    path = Path(file_name)
    if path.is_absolute():
        return str(path)
    return str(Path(base_path) / path)


def predict_vm(model, coords_norm, device, batch_size):
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_norm.shape[0], batch_size):
            batch = torch.from_numpy(coords_norm[start : start + batch_size]).to(device)
            predictions.append(model(batch)[:, 0:1].cpu().numpy())
    return np.vstack(predictions)


def denormalize_vm(vm_norm, bounds):
    if not bounds.get("vm_normalized", False):
        return vm_norm
    return vm_norm * (bounds["vm_scale_max"] - bounds["vm_scale_min"]) + bounds["vm_scale_min"]


def make_triangulation(data):
    if data.points is None or data.triangles is None:
        raise ValueError("Unstructured data requires points and triangles for plotting.")
    return mtri.Triangulation(data.points[:, 0], data.points[:, 1], data.triangles)


def plot_true_pred_error(triangulation, true_vm, pred_vm, time_indices, output_path):
    error = pred_vm - true_vm
    vmin = min(float(true_vm.min()), float(pred_vm.min()))
    vmax = max(float(true_vm.max()), float(pred_vm.max()))
    err_abs = max(float(np.abs(error).max()), 1e-12)

    fig, axes = plt.subplots(
        len(time_indices),
        3,
        figsize=(12.4, 3.7 * len(time_indices)),
        constrained_layout=True,
        squeeze=False,
    )

    for row, time_idx in enumerate(time_indices):
        panels = [
            (true_vm[:, time_idx], "True Vm", "turbo", vmin, vmax),
            (pred_vm[:, time_idx], "Predicted Vm", "turbo", vmin, vmax),
            (error[:, time_idx], "Error", "coolwarm", -err_abs, err_abs),
        ]
        for col, (values, title, cmap, panel_vmin, panel_vmax) in enumerate(panels):
            ax = axes[row, col]
            image = ax.tripcolor(
                triangulation,
                values,
                shading="gouraud",
                cmap=cmap,
                vmin=panel_vmin,
                vmax=panel_vmax,
            )
            ax.set_aspect("equal")
            ax.set_title(f"{title}, t={time_idx} ms")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_xlim(float(triangulation.x.min()), float(triangulation.x.max()))
            ax.set_ylim(float(triangulation.y.min()), float(triangulation.y.max()))
            fig.colorbar(image, ax=ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_frame(triangulation, pred_vm_frame, time_idx, output_path, vmin, vmax):
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    image = ax.tripcolor(
        triangulation,
        pred_vm_frame,
        shading="gouraud",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_title(f"PINN predicted Vm, t={time_idx} ms")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(float(triangulation.x.min()), float(triangulation.x.max()))
    ax.set_ylim(float(triangulation.y.min()), float(triangulation.y.max()))
    fig.colorbar(image, ax=ax, label="Vm (mV)")
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
    parser = argparse.ArgumentParser(description="Evaluate a spiral PINN checkpoint on an unstructured mesh.")
    parser.add_argument("--config", default="config_spiral_ms.yaml")
    parser.add_argument("--checkpoint", default="outputs/mlp_spiral_data_only.pt")
    parser.add_argument("--output-dir", default="outputs/spiral_pinn_eval")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--time-indices", type=int, nargs="+", default=None)
    parser.add_argument("--gif-start-frame", type=int, default=0)
    parser.add_argument("--gif-end-frame", type=int, default=None)
    parser.add_argument("--num-gif-frames", type=int, default=80)
    parser.add_argument("--gif-duration-ms", type=int, default=70)
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config.get("data", {})
    core_name = data_config.get("core_name", "")
    vm_path = config_path(core_name, data_config["v_file_name"])
    pts_path = config_path(core_name, data_config["pt_file_name"])
    elem_path = config_path(core_name, data_config["elem_file_name"])

    data = load_opencarp_unstructured_2d_voltage(
        vm_path,
        pts_path,
        elem_path=elem_path,
        dt=float(data_config.get("dt", 1.0)),
        normalize_vm=bool(data_config.get("normalize_vm", True)),
        vm_min=float(data_config["vm_min"]) if data_config.get("vm_min") is not None else None,
        vm_max=float(data_config["vm_max"]) if data_config.get("vm_max") is not None else None,
    )
    data = subset_time_window(
        data,
        t_min=float(data_config["time_min"]) if data_config.get("time_min") is not None else None,
        t_max=float(data_config["time_max"]) if data_config.get("time_max") is not None else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    hidden_widths = checkpoint.get("hidden_widths", config.get("model", {}).get("hidden_widths", [128, 128, 128, 128]))
    model = MLP_Net(hidden_widths=hidden_widths).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_norm_flat = predict_vm(model, data.coords_norm, device, args.batch_size)
    pred_norm = pred_norm_flat.reshape(data.vm.shape)
    true_mv = denormalize_vm(data.vm, data.bounds)
    pred_mv = denormalize_vm(pred_norm, data.bounds)

    rmse = float(np.sqrt(np.mean((pred_mv - true_mv) ** 2)))
    mae = float(np.mean(np.abs(pred_mv - true_mv)))
    max_error = float(np.max(np.abs(pred_mv - true_mv)))

    output_dir = Path(args.output_dir)
    triangulation = make_triangulation(data)

    if args.time_indices is None:
        args.time_indices = np.linspace(0, data.t.shape[0] - 1, 6, dtype=int).tolist()

    plot_true_pred_error(
        triangulation,
        true_mv,
        pred_mv,
        args.time_indices,
        output_dir / "true_pred_error_snapshots.png",
    )

    gif_start = max(args.gif_start_frame, 0)
    gif_end = data.t.shape[0] - 1 if args.gif_end_frame is None else min(args.gif_end_frame, data.t.shape[0] - 1)
    gif_indices = np.linspace(gif_start, gif_end, args.num_gif_frames, dtype=int)
    frame_dir = output_dir / "prediction_frames"
    frame_paths = []
    vmin = float(true_mv.min())
    vmax = float(true_mv.max())
    for time_idx in gif_indices:
        frame_path = frame_dir / f"pred_vm_{time_idx:04d}.png"
        plot_prediction_frame(triangulation, pred_mv[:, time_idx], time_idx, frame_path, vmin, vmax)
        frame_paths.append(frame_path)
    save_gif(frame_paths, output_dir / "pinn_prediction_vm.gif", args.gif_duration_ms)

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(f"true/pred shape: {true_mv.shape} / {pred_mv.shape}")
    print(f"RMSE: {rmse:.6f} mV")
    print(f"MAE: {mae:.6f} mV")
    print(f"max abs error: {max_error:.6f} mV")
    print(f"saved snapshots: {output_dir / 'true_pred_error_snapshots.png'}")
    print(f"saved prediction gif: {output_dir / 'pinn_prediction_vm.gif'}")


if __name__ == "__main__":
    main()
