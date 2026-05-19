import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplconfig").resolve()))
import matplotlib.pyplot as plt


def laplacian_no_flux(u, dx, dy):
    padded = np.pad(u, ((1, 1), (1, 1)), mode="edge")
    return (
        (padded[2:, 1:-1] - 2.0 * u + padded[:-2, 1:-1]) / dx**2
        + (padded[1:-1, 2:] - 2.0 * u + padded[1:-1, :-2]) / dy**2
    )


def smooth_switch(u, gate, sharpness):
    return 1.0 / (1.0 + np.exp(-sharpness * (u - gate)))


def apply_s1s2_stimulus(u, x_grid, y_grid, time, args):
    stim = np.zeros_like(u)

    if args.s1_start <= time < args.s1_start + args.s1_duration:
        stim[x_grid <= args.s1_width] += args.stim_magnitude

    if args.s2_start <= time < args.s2_start + args.s2_duration:
        s2_region = (
            (x_grid <= args.s2_xmax)
            & (y_grid <= args.s2_ymax)
            & (y_grid >= args.s2_ymin)
        )
        stim[s2_region] += args.stim_magnitude

    return stim


def initialize_phase_spiral(x_grid, y_grid, args):
    x0 = 0.5 * args.length_x
    y0 = 0.5 * args.length_y
    dx = x_grid - x0
    dy = y_grid - y0
    radius = np.sqrt(dx**2 + dy**2) + 1e-8
    theta = np.arctan2(dy, dx)
    phase = theta + args.phase_radial_wavenumber * radius

    # One Archimedean spiral front. Keeping phase_radial_wavenumber modest is
    # important: large values create many concentric rings instead of one rotor.
    wrapped = np.arctan2(np.sin(phase), np.cos(phase))
    front = np.exp(-(wrapped / args.phase_front_width) ** 2)
    front *= 1.0 - np.exp(-(radius / args.phase_core_radius) ** 2)
    u = np.clip(front, 0.0, 1.0)

    # Recovery variable is low behind the front. This refractory tail is what
    # makes the initialized pattern look and behave like a rotating wave.
    tail = (
        1.0 / (1.0 + np.exp(-args.phase_sharpness * (wrapped - args.phase_front_width)))
        * 1.0 / (1.0 + np.exp(args.phase_sharpness * (wrapped - args.phase_tail_width)))
    )
    h = np.clip(1.0 - args.phase_tail_refractory * tail, 0.05, 1.0)

    # Keep the rotor core partially refractory to prevent the first few steps
    # from collapsing into a target wave.
    core = radius < args.phase_core_radius
    u[core] *= 0.1
    h[core] = np.minimum(h[core], 0.2)
    return u.astype(np.float32), h.astype(np.float32)


def initialize_resting(args):
    return (
        np.zeros((args.nx, args.ny), dtype=np.float32),
        np.ones((args.nx, args.ny), dtype=np.float32),
    )


def initialize_opencarp_like(x_grid, y_grid, args):
    """Initialize broad broken wavefronts similar to the openCARP RP_E example."""
    x0 = 0.55 * args.length_x
    y0 = 0.50 * args.length_y
    dx = x_grid - x0
    dy = y_grid - y0
    radius = np.sqrt(dx**2 + dy**2) + 1e-8
    theta = np.arctan2(dy, dx)

    phase = theta + args.phase_radial_wavenumber * radius
    wrapped = np.arctan2(np.sin(phase), np.cos(phase))

    # Broad excited side of the wave plus a curved refractory tail. This makes
    # snapshots look closer to the Courtemanche/openCARP reentry movies than a
    # thin mathematical spiral line.
    excited = 1.0 / (1.0 + np.exp(-args.phase_sharpness * (wrapped + 0.15)))
    front_band = np.exp(-(wrapped / args.phase_front_width) ** 2)
    refractory_tail = 1.0 / (1.0 + np.exp(-args.phase_sharpness * (wrapped - 0.25)))

    # Break the wave with a vertical refractory channel, leaving two lobes.
    channel = np.exp(-((x_grid - 0.47 * args.length_x) / 0.35) ** 2)
    channel *= 1.0 / (1.0 + np.exp(-10.0 * (np.abs(y_grid - y0) - 1.2)))
    channel = np.clip(channel, 0.0, 1.0)

    # Add one small excited island like the local wavelets in the RP_E example.
    island = np.exp(
        -(
            ((x_grid - 0.16 * args.length_x) / 0.45) ** 2
            + ((y_grid - 0.50 * args.length_y) / 0.45) ** 2
        )
    )

    core = np.exp(-(radius / args.phase_core_radius) ** 2)
    u = np.clip(0.85 * excited + 0.18 * front_band + 0.95 * island, 0.0, 1.0)
    u *= 1.0 - 0.9 * core
    u *= 1.0 - 0.75 * channel

    h = np.ones_like(u)
    h -= args.phase_tail_refractory * np.clip(refractory_tail + 0.7 * channel + 0.8 * core, 0.0, 1.0)
    h = np.clip(h, 0.05, 1.0)
    return u.astype(np.float32), h.astype(np.float32)


def step_ms(u, h, stim, dx, dy, args):
    lap_u = laplacian_no_flux(u, dx, dy)
    reaction = h * u**2 * (1.0 - u) / args.tau_in - u / args.tau_out
    switch = smooth_switch(u, args.v_gate, args.switch_sharpness)
    h_rhs = (
        (1.0 - switch) * (1.0 - h) / args.tau_open
        + switch * (-h / args.tau_close)
    )

    u_next = u + args.dt * (args.diffusion * lap_u + reaction + stim)
    h_next = h + args.dt * h_rhs

    return (
        np.clip(u_next, 0.0, 1.0).astype(np.float32),
        np.clip(h_next, 0.0, 1.0).astype(np.float32),
    )


def plot_snapshots(x, y, t, u, output_path, title):
    indices = np.linspace(0, t.shape[0] - 1, 9, dtype=int)
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    for ax, index in zip(axes.ravel(), indices):
        image = ax.imshow(
            u[index].T,
            origin="lower",
            extent=extent,
            cmap="turbo",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        ax.set_title(f"t={t[index]:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(title)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="u")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_gif(x, y, t, u, output_path, max_frames, duration_ms):
    frame_dir = output_path.parent / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, t.shape[0] - 1, min(max_frames, t.shape[0]), dtype=int)
    frame_paths = []
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    for index in indices:
        fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
        image = ax.imshow(
            u[index].T,
            origin="lower",
            extent=extent,
            cmap="turbo",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
            aspect="equal",
        )
        ax.set_title(f"Mitchell-Schaeffer u, t={t[index]:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label="u")
        frame_path = frame_dir / f"ms_spiral_{index:04d}.png"
        fig.savefig(frame_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

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


def simulate(args):
    x = np.linspace(0.0, args.length_x, args.nx, dtype=np.float32)
    y = np.linspace(0.0, args.length_y, args.ny, dtype=np.float32)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    if args.init == "phase":
        u, h = initialize_phase_spiral(x_grid, y_grid, args)
    elif args.init == "opencarp_like":
        u, h = initialize_opencarp_like(x_grid, y_grid, args)
    elif args.init == "s1s2":
        u, h = initialize_resting(args)
    else:
        raise ValueError(f"Unsupported init: {args.init}")

    n_steps = int(round(args.duration / args.dt))
    save_every = max(1, args.save_every)
    saved_u = []
    saved_h = []
    saved_t = []

    for step in range(n_steps + 1):
        time = step * args.dt
        if step % save_every == 0:
            saved_u.append(u.copy())
            saved_h.append(h.copy())
            saved_t.append(time)

        if step == n_steps:
            break

        stim = apply_s1s2_stimulus(u, x_grid, y_grid, time, args) if args.init == "s1s2" else 0.0
        u, h = step_ms(u, h, stim, dx, dy, args)

        if step % args.log_every == 0:
            print(
                f"step {step:6d}/{n_steps} | t={time:8.2f} | "
                f"u=[{float(u.min()):.3f}, {float(u.max()):.3f}] | "
                f"h=[{float(h.min()):.3f}, {float(h.max()):.3f}]"
            )

    return (
        x,
        y,
        np.asarray(saved_t, dtype=np.float32),
        np.asarray(saved_u, dtype=np.float32),
        np.asarray(saved_h, dtype=np.float32),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2D Mitchell-Schaeffer spiral-wave data.")
    parser.add_argument("--output-dir", default="outputs/ms_spiral")
    parser.add_argument("--data-output", default="Data/MS/Spiral/ms_spiral_2d.npz")
    parser.add_argument("--init", choices=["phase", "s1s2", "opencarp_like"], default="opencarp_like")
    parser.add_argument("--nx", type=int, default=180)
    parser.add_argument("--ny", type=int, default=180)
    parser.add_argument("--length-x", type=float, default=10.0)
    parser.add_argument("--length-y", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=260.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--diffusion", type=float, default=0.01)
    parser.add_argument("--tau-in", type=float, default=0.3)
    parser.add_argument("--tau-out", type=float, default=5.0)
    parser.add_argument("--tau-open", type=float, default=120.0)
    parser.add_argument("--tau-close", type=float, default=150.0)
    parser.add_argument("--v-gate", type=float, default=0.13)
    parser.add_argument("--switch-sharpness", type=float, default=80.0)
    parser.add_argument("--stim-magnitude", type=float, default=1.0)
    parser.add_argument("--s1-start", type=float, default=0.0)
    parser.add_argument("--s1-duration", type=float, default=2.0)
    parser.add_argument("--s1-width", type=float, default=0.4)
    parser.add_argument("--s2-start", type=float, default=45.0)
    parser.add_argument("--s2-duration", type=float, default=2.0)
    parser.add_argument("--s2-xmax", type=float, default=5.0)
    parser.add_argument("--s2-ymin", type=float, default=0.0)
    parser.add_argument("--s2-ymax", type=float, default=4.8)
    parser.add_argument("--phase-radial-wavenumber", type=float, default=1.15)
    parser.add_argument("--phase-front-width", type=float, default=0.34)
    parser.add_argument("--phase-tail-width", type=float, default=2.35)
    parser.add_argument("--phase-tail-refractory", type=float, default=0.9)
    parser.add_argument("--phase-core-radius", type=float, default=0.35)
    parser.add_argument("--phase-sharpness", type=float, default=12.0)
    parser.add_argument("--gif-frames", type=int, default=100)
    parser.add_argument("--gif-duration-ms", type=int, default=70)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_output = Path(args.data_output)

    x, y, t, u, h = simulate(args)

    data_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_output,
        x=x,
        y=y,
        t=t,
        u=u,
        h=h,
        params=vars(args),
    )

    plot_snapshots(
        x,
        y,
        t,
        u,
        output_dir / "ms_spiral_snapshots.png",
        "Mitchell-Schaeffer 2D spiral snapshots",
    )
    save_gif(
        x,
        y,
        t,
        u,
        output_dir / "ms_spiral.gif",
        args.gif_frames,
        args.gif_duration_ms,
    )

    print(f"saved data: {data_output}")
    print(f"u shape: {u.shape}")
    print(f"h shape: {h.shape}")
    print(f"saved snapshots: {output_dir / 'ms_spiral_snapshots.png'}")
    print(f"saved gif: {output_dir / 'ms_spiral.gif'}")


if __name__ == "__main__":
    main()
