#!/usr/bin/env python3
"""
3D ellipses showing how the top PCA variances evolve with prefix length.
Each ellipse lives in the plane y = constant, so walking along y traces longer prefixes.
PC1 controls the ellipse span along x; PC2 controls the ellipse span along z.
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw 3D PCA ellipses by prefix length."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/qwen/Qwen__Qwen3-8B_probe_details.json"),
        help="Probe details JSON (expects explained_variance_ratio entries).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qwen/Qwen__Qwen3-8B_pca_prefix_ellipses.png"),
        help="Where to save the rendered figure.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Multiplier applied to the ellipse radii for readability.",
    )
    return parser.parse_args()


def load_data(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Probe details JSON must contain a mapping of prefixes.")
    return payload


def _configure_fonts() -> None:
    if shutil.which("latex"):
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}",
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["New PX", "Times New Roman", "DejaVu Serif"],
            }
        )


def render(prefix_map: dict[str, dict], scale: float, output: Path) -> None:
    _configure_fonts()

    def log_pos(value: float, base: float = 2.0) -> float:
        """Return the logarithmic coordinate so doubling prefixes stay evenly spaced."""
        return np.log(value) / np.log(base)

    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    prefix_keys = sorted(prefix_map.keys(), key=lambda k: int(k))

    skip_keys = {"192", "384"}
    entries = []

    for key in prefix_keys:
        if key in skip_keys:
            continue
        details = prefix_map[key]
        variances = details.get("explained_variance_ratio")
        if not variances or len(variances) < 2:
            continue

        v1, v2 = float(variances[0]), float(variances[1])
        raw_prefix = float(int(key))
        prefix_pos = log_pos(raw_prefix)
        area = v1 * v2
        entries.append((key, raw_prefix, prefix_pos, v1, v2, area))

    if not entries:
        return

    entries.sort(key=lambda item: item[1])  # draw earlier prefixes first

    areas = [area for *_, area in entries]
    min_area, max_area = min(areas), max(areas)

    def area_to_alpha(a: float) -> float:
        if np.isclose(min_area, max_area):
            return 0.6
        norm = (a - min_area) / (max_area - min_area)
        return 0.20 + 0.60 * norm

    fill_cmap = plt.colormaps["Purples"]
    edge_rgb = (0.25, 0.0, 0.3)
    edge_alpha = 0.25

    radii = []
    raw_prefixes = []
    log_prefixes = []

    for idx, (key, raw_prefix, prefix_pos, v1, v2, area) in enumerate(entries):
        raw_prefixes.append(raw_prefix)
        log_prefixes.append(prefix_pos)
        radii.append(max(v1, v2))

        x = (v1 * scale) * np.sin(theta)
        z = (v2 * scale) * np.cos(theta)
        y = np.full_like(theta, prefix_pos)

        verts = np.column_stack((x, y, z))
        color = fill_cmap(0.35 + 0.5 * idx / max(1, len(entries) - 1))
        alpha = area_to_alpha(area)
        poly = Poly3DCollection(
            [verts],
            facecolor=(*color[:3], alpha),
            edgecolor=(*edge_rgb, edge_alpha),
            linewidths=0.9,
        )
        if hasattr(poly, "set_sort_zpos"):
            poly.set_sort_zpos(prefix_pos)
        poly.set_zorder(idx)
        ax.add_collection3d(poly)
        line = ax.plot(
            x,
            y,
            z,
            color=(*edge_rgb, edge_alpha),
            linewidth=0.9,
            label=f"t={key}",
        )
        if line:
            line[0].set_zorder(idx + len(entries))

    ax.set_xlabel("1st PCA variance")
    ax.set_ylabel("Prefix length (tokens)")
    ax.set_zlabel("2nd PCA variance")
    ax.set_title("Top-2 PCA variances by prefix length", pad=1)

    ax.view_init(elev=15, azim=20)

    if log_prefixes:
        pad = 0.4
        ax.set_ylim(min(log_prefixes) - pad, max(log_prefixes) + pad)
        ax.set_yticks(log_prefixes)
        ax.set_yticklabels([str(int(p)) for p in raw_prefixes])
    if radii:
        bound = max(radii) * scale * 1.1
        ax.set_xlim(-bound, bound)
        ax.set_zlim(-bound, bound)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prefix_map = load_data(args.input)
    render(prefix_map, args.scale, args.output)
    print(f"Saved PCA ellipse figure to {args.output}")


if __name__ == "__main__":
    main()
