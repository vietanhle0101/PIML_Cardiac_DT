
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from physics import (
    boundary_no_flux_loss,
    initial_condition_loss,
    mitchell_schaeffer_pde_loss,
)


def sample_domain_points(num_points, device):
    """Sample normalized interior collocation points (x, y, t) in [-1, 1]^3."""
    return 2.0 * torch.rand(num_points, 3, device=device) - 1.0


def sample_boundary_points(num_points, device):
    """Sample normalized boundary points and outward normals for no-flux loss."""
    coords = sample_domain_points(num_points, device)
    normals = torch.zeros_like(coords)
    side = torch.randint(0, 4, (num_points,), device=device)

    left = side == 0
    right = side == 1
    bottom = side == 2
    top = side == 3

    coords[left, 0] = -1.0
    normals[left, 0] = -1.0

    coords[right, 0] = 1.0
    normals[right, 0] = 1.0

    coords[bottom, 1] = -1.0
    normals[bottom, 1] = -1.0

    coords[top, 1] = 1.0
    normals[top, 1] = 1.0

    return coords, normals


def sample_initial_points(data, num_points, device):
    """Sample normalized t=0 data points and their voltage targets."""
    initial_mask = abs(data.coords_norm[:, 2] + 1.0) < 1e-6
    initial_coords = torch.from_numpy(data.coords_norm[initial_mask]).to(device)
    initial_values = torch.from_numpy(data.values[initial_mask]).to(device)

    if initial_coords.shape[0] == 0:
        raise ValueError("No initial-condition points found at normalized t = -1.")

    indices = torch.randint(0, initial_coords.shape[0], (num_points,), device=device)
    return initial_coords[indices], initial_values[indices]


def train_pinn(
    model,
    train_x,
    train_vm,
    test_x,
    test_vm,
    data,
    args,
    device,
    physics_params,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_vm))
    test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_vm))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    print("\nStarting PINN training")
    print(f"train points: {len(train_dataset)}")
    print(f"test points: {len(test_dataset)}")
    print(f"batch size: {args.batch_size}")
    print(f"epochs: {args.epochs}")
    print(f"domain points per step: {args.num_domain}")
    print(f"boundary points per step: {args.num_boundary}")
    print(f"initial points per step: {args.num_initial}")

    best_test_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {
            "total": 0.0,
            "data": 0.0,
            "pde": 0.0,
            "boundary": 0.0,
            "initial": 0.0,
        }
        steps = 0

        for batch_x, batch_vm in train_loader:
            batch_x = batch_x.to(device)
            batch_vm = batch_vm.to(device)

            vm_pred = model(batch_x)[:, 0:1]
            loss_data = F.mse_loss(vm_pred, batch_vm)

            domain_coords = sample_domain_points(args.num_domain, device)
            loss_pde_vm, loss_pde_h = mitchell_schaeffer_pde_loss(
                model,
                domain_coords,
                data.bounds,
                physics_params,
            )
            loss_pde = loss_pde_vm + loss_pde_h

            boundary_coords, boundary_normals = sample_boundary_points(
                args.num_boundary,
                device,
            )
            loss_boundary = boundary_no_flux_loss(
                model,
                boundary_coords,
                boundary_normals,
                data.bounds,
            )

            initial_coords, initial_vm = sample_initial_points(
                data,
                args.num_initial,
                device,
            )
            loss_initial = initial_condition_loss(model, initial_coords, initial_vm)

            loss = (
                args.data_weight * loss_data
                + args.physics_weight * loss_pde
                + args.boundary_weight * loss_boundary
                + args.initial_weight * loss_initial
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sums["total"] += loss.item()
            sums["data"] += loss_data.item()
            sums["pde"] += loss_pde.item()
            sums["boundary"] += loss_boundary.item()
            sums["initial"] += loss_initial.item()
            steps += 1

        train_total = sums["total"] / steps
        train_data = sums["data"] / steps
        train_pde = sums["pde"] / steps
        train_boundary = sums["boundary"] / steps
        train_initial = sums["initial"] / steps

        model.eval()
        test_loss_sum = 0.0
        test_count = 0
        with torch.no_grad():
            for batch_x, batch_vm in test_loader:
                batch_x = batch_x.to(device)
                batch_vm = batch_vm.to(device)
                vm_pred = model(batch_x)[:, 0:1]
                loss = F.mse_loss(vm_pred, batch_vm)
                batch_size = batch_x.shape[0]
                test_loss_sum += loss.item() * batch_size
                test_count += batch_size

        test_loss = test_loss_sum / test_count
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        is_best = test_loss < best_test_loss - args.early_stop_min_delta
        if is_best:
            best_test_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:5d}/{args.epochs} | "
                f"total {train_total:.6e} | "
                f"data {train_data:.6e} | "
                f"pde {train_pde:.6e} | "
                f"bc {train_boundary:.6e} | "
                f"ic {train_initial:.6e} | "
                f"test_mse {test_loss:.6e} | "
                f"best_test_mse {best_test_loss:.6e} | "
                f"lr {current_lr:.3e} | "
            )

        if is_best and args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_widths": args.hidden_widths,
                    "bounds": data.bounds,
                    "physics_params": physics_params,
                    "epoch": epoch,
                    "train_total": train_total,
                    "train_data": train_data,
                    "train_pde": train_pde,
                    "train_boundary": train_boundary,
                    "train_initial": train_initial,
                    "test_mse": test_loss,
                    "lr": current_lr,
                },
                checkpoint_path,
            )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"early stopping at epoch {epoch}: "
                f"no test MSE improvement greater than {args.early_stop_min_delta} "
                f"for {args.early_stop_patience} epochs"
            )
            break

    if args.checkpoint:
        print(f"saved best PINN checkpoint: {args.checkpoint}")

