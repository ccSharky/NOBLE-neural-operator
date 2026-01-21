import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ----------------------------
# 1) Your original solver (diffusion–advection)
# ----------------------------
def run_diffusion_advection():
    from neuralop.losses.differentiation import FiniteDiff

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Lx, Ly = 2.0, 2.0
    nx, ny = 64, 64
    T = 1.6
    dt = 0.001
    nu = 0.02
    cx, cy = 1.0, 0.6

    X = torch.linspace(0, Lx, nx, device=device).repeat(ny, 1).T
    Y = torch.linspace(0, Ly, ny, device=device).repeat(nx, 1)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    nt = int(T / dt)

    fd = FiniteDiff(dim=2, h=(dx, dy))

    u = (
        -torch.sin(2 * np.pi * Y) * torch.cos(2 * np.pi * X)
        + 0.3 * torch.exp(-((X - 0.75) ** 2 + (Y - 0.5) ** 2) / 0.02)
        - 0.3 * torch.exp(-((X - 1.25) ** 2 + (Y - 1.5) ** 2) / 0.02)
    ).to(device)

    def source_term(X, Y, t):
        return 0.2 * torch.sin(3 * np.pi * X) * torch.cos(3 * np.pi * Y) * torch.cos(4 * np.pi * t)

    u_evolution = [u.clone()]
    t = torch.tensor(0.0, device=device)

    for _ in range(nt):
        u_x = fd.dx(u)
        u_y = fd.dy(u)
        u_xx = fd.dx(u_x)
        u_yy = fd.dy(u_y)

        u = u + dt * (-cx * u_x - cy * u_y + nu * (u_xx + u_yy) + source_term(X, Y, t))
        t += dt
        u_evolution.append(u.clone())

    u_evolution = torch.stack(u_evolution).cpu().numpy()

    num_frames = 100
    frame_indices = np.linspace(0, len(u_evolution) - 1, num_frames, dtype=int)
    u_frames = u_evolution[frame_indices]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(u_frames[0], extent=[0, Lx, 0, Ly], origin="lower", cmap="plasma")
    plt.colorbar(im, ax=ax, shrink=0.75)

    def update(i):
        im.set_data(u_frames[i])
        ax.set_title(f"Time: {frame_indices[i] * dt:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        return (im,)

    _ = animation.FuncAnimation(fig, update, frames=len(u_frames), interval=50, blit=False)
    plt.show()


# ----------------------------
# 2) Navier–Stokes dataset (nsforcing_train_128.pt / nsforcing_test_128.pt)
# ----------------------------
def load_nsforcing(data_dir="./data", batch_size=4):
    """
    Loads the Navier-Stokes dataset using NeuralOperator's built-in loader.
    It expects files like:
      data/nsforcing_train_128.pt
      data/nsforcing_test_128.pt
    (or it will download nsforcing_128.tgz from Zenodo if missing).
    """
    from neuralop.data.datasets import load_navier_stokes_pt

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        n_train=100,
        n_tests=[20],
        batch_size=batch_size,
        test_batch_sizes=[batch_size],
        data_root=data_root,
        train_resolution=128,
        test_resolutions=[128],
    )
    return train_loader, test_loaders, data_processor


def _unpack_batch(batch):
    """
    Makes this robust even if the dataset returns dict or tuple.
    """
    if isinstance(batch, dict):
        # common names in NeuralOperator datasets are "x" and "y"
        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]
        # fallback: take first 2 values
        vals = list(batch.values())
        return vals[0], vals[1]

    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise ValueError(f"Don't know how to unpack batch type: {type(batch)}")


def ns_demo_plot(data_dir="./data"):
    train_loader, test_loaders, _ = load_nsforcing(data_dir=data_dir, batch_size=2)

    batch = next(iter(train_loader))
    x, y = _unpack_batch(batch)

    # x,y usually are [B, C, H, W]
    x0 = x[0, 0].detach().cpu().numpy()
    y0 = y[0, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x0)
    axes[0].set_title("NS input (x)")
    axes[0].axis("off")

    axes[1].imshow(y0)
    axes[1].set_title("NS output (y)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["diffusion", "ns_demo"], default="diffusion")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    if args.mode == "diffusion":
        run_diffusion_advection()
    elif args.mode == "ns_demo":
        ns_demo_plot(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
