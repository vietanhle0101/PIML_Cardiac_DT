import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from data_load import load_ms_npz_2d_voltage, subset_time_window
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


def predict_frames(model, coords_norm, frame_size, frame_indices, device, batch_size):
    predictions = {}
    model.eval()
    with torch.no_grad():
        for frame_idx in frame_indices:
            start = frame_idx * frame_size
            end = start + frame_size
            frame_coords = coords_norm[start:end]
            chunks = []
            for batch_start in range(0, frame_coords.shape[0], batch_size):
                batch = torch.from_numpy(frame_coords[batch_start : batch_start + batch_size]).to(device)
                chunks.append(model(batch)[:, 0].cpu().numpy())
            predictions[int(frame_idx)] = np.concatenate(chunks).reshape(frame_size)
    return predictions


def save_comparison_frame(data, pred_frame, frame_idx, output_path, vmin, vmax):
    true_frame = data.vm[:, :, frame_idx]
    pred_frame = pred_frame.reshape(data.x.shape[0], data.y.shape[0])
    extent = [float(data.x.min()), float(data.x.max()), float(data.y.min()), float(data.y.max())]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), constrained_layout=True)
    panels = [
        (true_frame, "Generated MS data"),
        (pred_frame, "PINN prediction"),
    ]

    for ax, (values, title) in zip(axes, panels):
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=extent,
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )
        ax.set_title(f"{title}, t={data.t[frame_idx]:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label="u")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_gif(frame_paths, output_path, duration_ms):
    first_size = Image.open(frame_paths[0]).size
    images = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        if image.size != first_size:
            fixed = Image.new("RGB", first_size, "white")
            fixed.paste(image, (0, 0))
            image = fixed
        images.append(image.convert("P", palette=Image.ADAPTIVE))

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    for image in images:
        image.close()


def main():
    parser = argparse.ArgumentParser(description="Create a synchronized MS-vs-PINN spiral comparison GIF.")
    parser.add_argument("--config", default="config_ms_spiral_generated.yaml")
    parser.add_argument("--checkpoint", default="outputs/mlp_ms_spiral_pinn.pt")
    parser.add_argument("--output-gif", default="outputs/ms_spiral_side_by_side_comparison.gif")
    parser.add_argument("--frames-dir", default="outputs/ms_spiral_side_by_side_frames")
    parser.add_argument("--num-frames", type=int, default=80)
    parser.add_argument("--duration-ms", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config.get("data", {})
    core_name = data_config.get("core_name", "")
    npz_path = config_path(core_name, data_config["npz_file_name"])
    data = load_ms_npz_2d_voltage(npz_path)
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

    frame_indices = np.linspace(0, data.t.shape[0] - 1, args.num_frames, dtype=int)
    frame_size = data.x.shape[0] * data.y.shape[0]
    pred_frames = predict_frames(
        model,
        data.coords_norm,
        frame_size,
        frame_indices,
        device,
        args.batch_size,
    )

    frames_dir = Path(args.frames_dir)
    frame_paths = []
    vmin = float(data.vm.min())
    vmax = float(data.vm.max())
    for output_idx, frame_idx in enumerate(frame_indices):
        frame_path = frames_dir / f"comparison_{output_idx:04d}.png"
        save_comparison_frame(data, pred_frames[int(frame_idx)], int(frame_idx), frame_path, vmin, vmax)
        frame_paths.append(frame_path)

    save_gif(frame_paths, Path(args.output_gif), args.duration_ms)

    print(f"saved gif: {args.output_gif}")
    print(f"saved frames: {args.frames_dir}")
    print(f"frames: {len(frame_paths)}")
    print(f"time range: {float(data.t[frame_indices[0]]):.1f} to {float(data.t[frame_indices[-1]]):.1f}")


if __name__ == "__main__":
    main()
