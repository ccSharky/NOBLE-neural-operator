import argparse
import os
import tarfile
from pathlib import Path

import numpy as np
import torch


def extract_tgz(tgz_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tar:
        def is_within_directory(directory: str, target: str) -> bool:
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

        for member in tar.getmembers():
            member_path = os.path.join(str(out_dir), member.name)
            if not is_within_directory(str(out_dir), member_path):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tar.extractall(str(out_dir))


def load_pt(pt_path: Path) -> dict:
    obj = torch.load(str(pt_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a dict in {pt_path}, got {type(obj)}")
    if "x" not in obj or "y" not in obj:
        raise ValueError(f"Expected keys 'x' and 'y' in {pt_path}. Keys: {list(obj.keys())}")
    return obj


def to_numpy_2d(t: torch.Tensor) -> np.ndarray:
    # Supports [H,W] or [C,H,W] (takes first channel) or [H,W,C] (takes first channel)
    t = t.detach().cpu()
    if t.ndim == 2:
        arr = t.numpy()
    elif t.ndim == 3 and t.shape[0] in (1, 2, 3, 4):
        arr = t[0].numpy()
    elif t.ndim == 3 and t.shape[-1] in (1, 2, 3, 4):
        arr = t[..., 0].numpy()
    else:
        raise ValueError(f"Expected 2D field (or 3D with small channel dim), got shape {tuple(t.shape)}")
    return arr.astype(np.float32, copy=False)


def save_static_pair(x2d: np.ndarray, y2d: np.ndarray, out_png: Path, *, cmap: str = "viridis") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin = float(min(x2d.min(), y2d.min()))
    vmax = float(max(x2d.max(), y2d.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(x2d, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("x")
    axes[0].axis("off")
    im1 = axes[1].imshow(y2d, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("y")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes, shrink=0.9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)


def save_animation_over_samples(
    x: torch.Tensor,
    y: torch.Tensor,
    out_path: Path,
    *,
    start: int,
    nframes: int,
    fps: int,
    cmap: str,
) -> None:
    """
    This dataset is [N, H, W] (no time dimension), so we animate across sample index.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    end = min(start + nframes, x.shape[0])
    idxs = list(range(start, end))
    if len(idxs) < 2:
        raise ValueError("Need at least 2 frames to animate. Increase --nframes or lower --start.")

    # Precompute global color scale across shown frames for stable colors
    xs = [to_numpy_2d(x[i]) for i in idxs]
    ys = [to_numpy_2d(y[i]) for i in idxs]
    vmin = float(min(min(a.min() for a in xs), min(a.min() for a in ys)))
    vmax = float(max(max(a.max() for a in xs), max(a.max() for a in ys)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    imx = axes[0].imshow(xs[0], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("x")
    axes[0].axis("off")
    imy = axes[1].imshow(ys[0], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("y")
    axes[1].axis("off")
    fig.colorbar(imy, ax=axes, shrink=0.9)

    title = fig.suptitle(f"sample {idxs[0]} / {idxs[-1]}")

    def update(frame_i: int):
        imx.set_data(xs[frame_i])
        imy.set_data(ys[frame_i])
        title.set_text(f"sample {idxs[frame_i]} / {idxs[-1]}")
        return (imx, imy, title)

    anim = FuncAnimation(fig, update, frames=len(idxs), interval=int(1000 / max(1, fps)), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Default: GIF (works without ffmpeg). If user chooses .mp4, they need ffmpeg installed.
    if out_path.suffix.lower() == ".gif":
        anim.save(str(out_path), writer=PillowWriter(fps=fps))
    elif out_path.suffix.lower() == ".mp4":
        anim.save(str(out_path), fps=fps)  # relies on ffmpeg in matplotlib config
    else:
        raise ValueError("Animation output must end with .gif or .mp4")

    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize nsforcing .pt files (and optionally extract a .tgz first).")
    ap.add_argument("--tgz", type=str, default=None, help="Path to a .tgz/.tar.gz archive to extract before loading.")
    ap.add_argument("--extract-dir", type=str, default="extracted", help="Where to extract archive contents.")
    ap.add_argument("--pt", type=str, default="nsforcing_128/nsforcing_train_128.pt", help="Path to .pt file to visualize.")
    ap.add_argument("--outdir", type=str, default="viz_out", help="Output directory for images/animations.")

    ap.add_argument("--index", type=int, default=0, help="Sample index for static visualization.")
    ap.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap.")

    ap.add_argument("--animate", action="store_true", help="Create an animation over sample indices.")
    ap.add_argument("--start", type=int, default=0, help="Start sample index for animation.")
    ap.add_argument("--nframes", type=int, default=100, help="Number of frames (samples) in animation.")
    ap.add_argument("--fps", type=int, default=15, help="Frames per second for animation.")
    ap.add_argument("--anim-out", type=str, default="samples.gif", help="Animation filename (.gif or .mp4).")

    args = ap.parse_args()

    root = Path(__file__).resolve().parent

    if args.tgz:
        tgz_path = Path(args.tgz).expanduser()
        extract_dir = (root / args.extract_dir).resolve()
        extract_tgz(tgz_path, extract_dir)
        print(f"Extracted {tgz_path} -> {extract_dir}")

    pt_path = (root / args.pt).resolve() if not os.path.isabs(args.pt) else Path(args.pt).expanduser().resolve()
    data = load_pt(pt_path)
    x, y = data["x"], data["y"]

    if x.ndim < 3 or y.ndim < 3:
        raise ValueError(f"Expected x,y to be at least [N,H,W]. Got x={tuple(x.shape)} y={tuple(y.shape)}")

    n = int(x.shape[0])
    if not (0 <= args.index < n):
        raise ValueError(f"--index must be in [0, {n-1}], got {args.index}")

    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Basic stats
    print(f"Loaded: {pt_path}")
    print(f"x: shape={tuple(x.shape)} dtype={x.dtype} min={float(x.min()):.6g} max={float(x.max()):.6g}")
    print(f"y: shape={tuple(y.shape)} dtype={y.dtype} min={float(y.min()):.6g} max={float(y.max()):.6g}")

    x2d = to_numpy_2d(x[args.index])
    y2d = to_numpy_2d(y[args.index])
    out_png = outdir / f"sample_{args.index:05d}.png"
    save_static_pair(x2d, y2d, out_png, cmap=args.cmap)
    print(f"Saved static visualization: {out_png}")

    if args.animate:
        anim_path = outdir / args.anim_out
        save_animation_over_samples(
            x, y, anim_path, start=args.start, nframes=args.nframes, fps=args.fps, cmap=args.cmap
        )
        print(f"Saved animation: {anim_path}")


if __name__ == "__main__":
    main()

