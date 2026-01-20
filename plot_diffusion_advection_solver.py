import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from neuralop.losses.differentiation import FiniteDiff


def run_diffusion_advection():
    """Your original diffusion-advection solver (wrapped into a function)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulation parameters
    Lx, Ly = 2.0, 2.0
    nx, ny = 64, 64
    T = 1.6
    dt = 0.001
    nu = 0.02
    cx, cy = 1.0, 0.6

    # Create grid
    X = torch.linspace(0, Lx, nx, device=device).repeat(ny, 1).T
    Y = torch.linspace(0, Ly, ny, device=device).repeat(nx, 1)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    nt = int(T / dt)

    # FD operator
    fd = FiniteDiff(dim=2, h=(dx, dy))

    # Initial condition
    u = (
        -torch.sin(2 * np.pi * Y) * torch.cos(2 * np.pi * X)
        + 0.3 * torch.exp(-((X - 0.75) ** 2 + (Y - 0.5) ** 2) / 0.02)
        - 0.3 * torch.exp(-((X - 1.25) ** 2 + (Y - 1.5) ** 2) / 0.02)
    ).to(device)

    def source_term(X, Y, t):
        return 0.2 * torch.sin(3 * np.pi * X) * torch.cos(3 * np.pi * Y) * torch.cos(4 * np.pi * t)

    # Simulate
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

    # Animate (note: in a .py script you usually need plt.show())
    num_frames = 100
    frame_indices = np.linspace(0, len(u_evolution) - 1, num_frames, dtype=int)
    u_frames = u_evolution[frame_indices]

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap_u = ax.imshow(u_frames[0], extent=[0, Lx, 0, Ly], origin="lower", cmap="plasma")
    ax.set_title("Advection-Diffusion: u")
    plt.colorbar(cmap_u, ax=ax, shrink=0.75)

    def update(frame):
        cmap_u.set_data(u_frames[frame])
        ax.set_title(f"Time: {frame_indices[frame] * dt:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        return (cmap_u,)

    _ = animation.FuncAnimation(fig, update, frames=len(u_frames), interval=50, blit=False)
    plt.show()


def darcy_demo(data_dir: str = "./data/darcy"):
    """
    Loads the small Darcy dataset and visualizes one sample (input permeability x, output pressure y).
    """
    # Import here so diffusion mode can still run even if you haven’t installed neuralop deps correctly yet.
    from neuralop.data.datasets import load_darcy_flow_small

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    # Small example: train at 16x16, test at 16x16 and 32x32
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=20,
        batch_size=4,
        test_resolutions=[16, 32],
        n_tests=[10, 10],
        test_batch_sizes=[4, 2],
        data_root=data_root,
    )

    # Grab one sample from the 16x16 test set
    sample = test_loaders[16].dataset[0]
    sample = data_processor.preprocess(sample, batched=False)  # returns dict with "x" and "y"

    x = sample["x"]  # shape ~ [channels=1, H, W]
    y = sample["y"]  # shape ~ [1, H, W] (or [H,W] depending on squeeze)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x[0], cmap="gray")
    axes[0].set_title("Darcy input x (permeability)")
    axes[0].axis("off")

    axes[1].imshow(y.squeeze())
    axes[1].set_title("Darcy output y (pressure)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def train_tfno_darcy(data_dir: str = "./data/darcy"):
    """
    Trains a TFNO on the small Darcy dataset (basically the official FNO tutorial, but using TFNO).
    """
    from neuralop.models import TFNO
    from neuralop import Trainer, LpLoss, H1Loss
    from neuralop.training import AdamW
    from neuralop.data.datasets import load_darcy_flow_small

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000,
        batch_size=64,
        n_tests=[100, 50],
        test_resolutions=[16, 32],
        test_batch_sizes=[32, 32],
        data_root=data_root,
    )
    data_processor = data_processor.to(device)

    # TFNO: like FNO, but Tucker-factorized weights by default
    model = TFNO(
        n_modes=(8, 8),
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        rank=0.1,  # ~10% of dense params
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    trainer = Trainer(
        model=model,
        n_epochs=15,
        device=device,
        data_processor=data_processor,
        wandb_log=False,
        eval_interval=5,
        use_distributed=False,
        verbose=True,
    )

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=h1loss,
        eval_losses={"h1": h1loss, "l2": l2loss},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="diffusion",
        choices=["diffusion", "darcy_demo", "darcy_train_tfno"],
        help="Which thing to run.",
    )
    parser.add_argument("--data_dir", type=str, default="./data/darcy", help="Where to store/download Darcy data.")
    args = parser.parse_args()

    if args.mode == "diffusion":
        run_diffusion_advection()
    elif args.mode == "darcy_demo":
        darcy_demo(data_dir=args.data_dir)
    elif args.mode == "darcy_train_tfno":
        train_tfno_darcy(data_dir=args.data_dir)


if __name__ == "__main__":
    main()

