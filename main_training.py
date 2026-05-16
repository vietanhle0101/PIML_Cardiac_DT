import argparse
from types import SimpleNamespace
from pathlib import Path

import torch
import yaml

from data_load import load_opencarp_2d_voltage, subset_time_window, train_test_split_points
from model import MLP_Net
from physics import ms_params_from_config
from training import train_pinn


def load_config(config_path):
    with Path(config_path).open() as file:
        return yaml.safe_load(file)


def config_path(base_path, file_name):
    path = Path(file_name)
    if path.is_absolute():
        return str(path)
    return str(Path(base_path) / path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_ms.yaml")
    parser.add_argument("--checkpoint", default="outputs/mlp_pinn.pt")
    parser.add_argument("--data-weight", type=float, default=None)
    parser.add_argument("--physics-weight", type=float, default=None)
    parser.add_argument("--boundary-weight", type=float, default=None)
    parser.add_argument("--initial-weight", type=float, default=None)
    cli_args = parser.parse_args()

    config = load_config(cli_args.config)
    data_config = config.get("data", {})
    core_name = data_config.get("core_name", "")
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    seed = training_config.get("seed", 42)
    torch.manual_seed(seed)
    args = SimpleNamespace(
        vm=config_path(core_name, data_config["v_file_name"]),
        pts=config_path(core_name, data_config["pt_file_name"]),
        dt=data_config.get("dt", 1.0),
        z_slice=data_config.get("z_slice", "middle"),
        time_min=data_config.get("time_min"),
        time_max=data_config.get("time_max"),
        hidden_widths=model_config.get("hidden_widths", [64, 64, 64, 64, 64]),
        num_domain=training_config.get("num_domain", 256),
        num_boundary=training_config.get("num_boundary", 128),
        num_initial=training_config.get("num_initial", training_config.get("num_boundary", 128)),
        train_fraction=training_config.get("train_fraction", 0.25),
        batch_size=training_config.get("batch_size", 4096),
        epochs=training_config.get("epochs", 200),
        lr=training_config.get("lr", 1e-3),
        lr_factor=training_config.get("lr_factor", 0.5),
        lr_patience=training_config.get("lr_patience", 10),
        min_lr=training_config.get("min_lr", 1e-6),
        early_stop_patience=training_config.get("early_stop_patience", 30),
        early_stop_min_delta=training_config.get("early_stop_min_delta", 0.0),
        log_every=training_config.get("log_every", 10),
        data_weight=cli_args.data_weight if cli_args.data_weight is not None else training_config.get("data_weight", 1.0),
        physics_weight=cli_args.physics_weight if cli_args.physics_weight is not None else training_config.get("physics_weight", 1.0),
        boundary_weight=cli_args.boundary_weight if cli_args.boundary_weight is not None else training_config.get("boundary_weight", 1.0),
        initial_weight=cli_args.initial_weight if cli_args.initial_weight is not None else training_config.get("initial_weight", 1.0),
        checkpoint=cli_args.checkpoint,
        seed=seed,
    )
    physics_params = ms_params_from_config(config)

    data = load_opencarp_2d_voltage(args.vm, args.pts, dt=args.dt, z_slice=args.z_slice)
    data = subset_time_window(data, t_min=args.time_min, t_max=args.time_max)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_Net(
        hidden_widths=args.hidden_widths,
    ).to(device)

    train_x, train_vm, test_x, test_vm = train_test_split_points(
        data.coords_norm,
        data.values,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    train_pinn(
        model,
        train_x,
        train_vm,
        test_x,
        test_vm,
        data,
        args,
        device,
        physics_params,
    )


if __name__ == "__main__":
    main()
