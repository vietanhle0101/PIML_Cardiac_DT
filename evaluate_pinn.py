import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from data_load import load_opencarp_2d_voltage, subset_time_window
from model import MLP_Net

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplconfig").resolve()))
import matplotlib.pyplot as plt


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
            vm = model(batch)[:, 0:1]
            predictions.append(vm.cpu().numpy())
    return np.vstack(predictions)


def plot_time_slices(data, vm_pred, time_indices, output_path):
    vm_true = data.vm
    error = vm_pred - vm_true
    vmax = max(float(vm_true.max()), float(vm_pred.max()))
    vmin = min(float(vm_true.min()), float(vm_pred.min()))
    err_abs = max(float(np.abs(error).max()), 1e-12)

    fig, axes = plt.subplots(
        len(time_indices),
        3,
        figsize=(11, 3.2 * len(time_indices)),
        constrained_layout=True,
    )
    if len(time_indices) == 1:
        axes = axes.reshape(1, 3)

    extent = [float(data.x.min()), float(data.x.max()), float(data.y.min()), float(data.y.max())]

    for row, time_idx in enumerate(time_indices):
        true_slice = vm_true[:, :, time_idx].T
        pred_slice = vm_pred[:, :, time_idx].T
        err_slice = error[:, :, time_idx].T

        images = [
            axes[row, 0].imshow(true_slice, origin="lower", extent=extent, vmin=vmin, vmax=vmax, aspect="auto"),
            axes[row, 1].imshow(pred_slice, origin="lower", extent=extent, vmin=vmin, vmax=vmax, aspect="auto"),
            axes[row, 2].imshow(err_slice, origin="lower", extent=extent, vmin=-err_abs, vmax=err_abs, cmap="coolwarm", aspect="auto"),
        ]

        axes[row, 0].set_title(f"True vm, t={data.t[time_idx]:.1f}")
        axes[row, 1].set_title(f"Predicted vm, t={data.t[time_idx]:.1f}")
        axes[row, 2].set_title("Error")

        for col in range(3):
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("y")
            fig.colorbar(images[col], ax=axes[row, col])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_ms.yaml")
    parser.add_argument("--checkpoint", default="outputs/mlp_pinn.pt")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--output", default="outputs/pinn_time_slices.png")
    parser.add_argument("--time-indices", type=int, nargs="+", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config.get("data", {})
    core_name = data_config.get("core_name", "")
    vm_path = config_path(core_name, data_config["v_file_name"])
    pts_path = config_path(core_name, data_config["pt_file_name"])
    dt = data_config.get("dt", 1.0)
    z_slice = data_config.get("z_slice", "middle")
    time_min = data_config.get("time_min")
    time_max = data_config.get("time_max")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    hidden_widths = checkpoint.get("hidden_widths", [64, 64, 64, 64, 64])
    model = MLP_Net(hidden_widths=hidden_widths).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    data = load_opencarp_2d_voltage(vm_path, pts_path, dt=dt, z_slice=z_slice)
    data = subset_time_window(data, t_min=time_min, t_max=time_max)
    vm_pred_flat = predict_vm(model, data.coords_norm, device, args.batch_size)
    vm_pred = vm_pred_flat.reshape(data.vm.shape)

    rmse = np.sqrt(np.mean((vm_pred - data.vm) ** 2))
    mae = np.mean(np.abs(vm_pred - data.vm))
    max_error = np.max(np.abs(vm_pred - data.vm))

    if args.time_indices is None:
        args.time_indices = np.linspace(0, data.t.shape[0] - 1, 9, dtype=int).tolist()

    plot_time_slices(data, vm_pred, args.time_indices, Path(args.output))

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(f"vm_pred shape: {vm_pred.shape}")
    print(f"vm_true shape: {data.vm.shape}")
    print(f"RMSE: {rmse:.6e}")
    print(f"MAE: {mae:.6e}")
    print(f"max abs error: {max_error:.6e}")
    print(f"saved plot: {args.output}")


if __name__ == "__main__":
    main()
