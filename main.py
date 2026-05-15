import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_load import load_opencarp_2d_voltage, train_test_split_points
from model import MLP_Net
from physics import mitchell_schaeffer_pde_loss


def train_data_only(model, train_x, train_vm, test_x, test_vm, args, device):
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

    print("\nStarting data-only training")
    print(f"train points: {len(train_dataset)}")
    print(f"test points: {len(test_dataset)}")
    print(f"batch size: {args.batch_size}")
    print(f"epochs: {args.epochs}")

    best_test_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_x, batch_vm in train_loader:
            batch_x = batch_x.to(device)
            batch_vm = batch_vm.to(device)

            pred = model(batch_x)
            vm_pred = pred[:, 0:1]
            loss = F.mse_loss(vm_pred, batch_vm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = batch_x.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / train_count

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
                f"train_mse {train_loss:.6e} | "
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
                    "bounds": args.bounds,
                    "epoch": epoch,
                    "train_mse": train_loss,
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
        print(f"saved best checkpoint: {args.checkpoint}")


def test_physics_batch(model, bounds, args, device):
    coords = 2.0 * torch.rand(args.physics_batch_size, 3, device=device) - 1.0
    loss_vm, loss_h = mitchell_schaeffer_pde_loss(model, coords, bounds)
    total = loss_vm + loss_h

    print("\nPhysics residual smoke test")
    print(f"collocation points: {args.physics_batch_size}")
    print(f"loss_pde_vm: {loss_vm.item():.6e}")
    print(f"loss_pde_h: {loss_h.item():.6e}")
    print(f"loss_pde_total: {total.item():.6e}")
    print(f"finite: {torch.isfinite(total).item()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vm", default="openCARP-PINNs/Data/MS/Single_Corner/vm.igb")
    parser.add_argument("--pts", default="openCARP-PINNs/Data/MS/Mesh/block")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--z-slice", default="middle")
    parser.add_argument("--hidden-widths", type=int, nargs="+", default=[64, 64, 64, 64, 64])
    parser.add_argument("--train-data-only", action="store_true")
    parser.add_argument("--test-physics", action="store_true")
    parser.add_argument("--physics-batch-size", type=int, default=1024)
    parser.add_argument("--train-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="outputs/mlp_data_only.pt")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    data = load_opencarp_2d_voltage(args.vm, args.pts, dt=args.dt, z_slice=args.z_slice)

    print("Loaded openCARP voltage data")
    print(f"x shape: {data.x.shape}, range: [{data.x.min():.3f}, {data.x.max():.3f}]")
    print(f"y shape: {data.y.shape}, range: [{data.y.min():.3f}, {data.y.max():.3f}]")
    print(f"t shape: {data.t.shape}, range: [{data.t.min():.3f}, {data.t.max():.3f}]")
    print(f"vm shape: {data.vm.shape}, range: [{data.vm.min():.6f}, {data.vm.max():.6f}]")
    print(f"coords shape: {data.coords.shape}")
    print(f"coords_norm range: [{data.coords_norm.min():.3f}, {data.coords_norm.max():.3f}]")
    print(f"values shape: {data.values.shape}")
    print(f"bounds: {data.bounds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_Net(
        hidden_widths=args.hidden_widths,
    ).to(device)

    sample_coords = torch.from_numpy(data.coords_norm[:8]).to(device)
    with torch.no_grad():
        sample_pred = model(sample_coords)

    n_params = sum(param.numel() for param in model.parameters())
    print("\nInitialized Mitchell-Schaeffer MLP")
    print(f"device: {device}")
    print(f"hidden widths: {args.hidden_widths}")
    print(f"parameters: {n_params}")
    print(f"sample input shape: {sample_coords.shape}")
    print(f"sample output shape: {sample_pred.shape}")
    print(f"sample vm range: [{sample_pred[:, 0].min().item():.4f}, {sample_pred[:, 0].max().item():.4f}]")
    print(f"sample h range: [{sample_pred[:, 1].min().item():.4f}, {sample_pred[:, 1].max().item():.4f}]")

    if args.train_data_only:
        train_x, train_vm, test_x, test_vm = train_test_split_points(
            data.coords_norm,
            data.values,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )
        args.bounds = data.bounds
        train_data_only(model, train_x, train_vm, test_x, test_vm, args, device)

    if args.test_physics:
        test_physics_batch(model, data.bounds, args, device)


if __name__ == "__main__":
    main()
