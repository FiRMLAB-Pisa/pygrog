"""fig_package — PyGROG package-structure diagram (pure matplotlib).

A boxes-and-arrows architecture diagram of the public package layout.
Pure matplotlib — no external services or installs.

A Mermaid source is also exported as ``fig_package.mmd`` next to this
script for reference / documentation purposes; it is not used to render
the PNG.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import POSTER_STYLE, save_fig  # noqa: E402

MMD_SOURCE = """\
flowchart LR
  K["non-Cartesian k-space + trajectory"]
  U["utils.coil_compress / nlinv_calib"]
  G["calib.GrogInterpolator"]
  OP["operator.SparseFFT / MaskedFFT"]
  GAD["gadgets.OffResonance / Subspace"]
  CORE["_solve / _toep   (cg / lsmr / pcg, Toeplitz op.normal)"]
  IO["interop.sigpy / deepinv / mrpro"]
  OUT["coefficient images / x"]

  K --> U --> G --> OP
  OP --> GAD
  OP --> CORE
  GAD --> CORE
  CORE --> OUT
  OP -. wrap .-> IO
"""


def _box(ax, x, y, w, h, label, fc, ec):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        fc=fc, ec=ec, lw=2.0,
    ))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=14, color="#1a1a1a")
    return {
        "left":   (x, y + h / 2),
        "right":  (x + w, y + h / 2),
        "top":    (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
    }


def _arrow(ax, src, dst, *, linestyle="-", rad=0.0, color="#444", lw=1.8):
    ax.add_patch(FancyArrowPatch(
        src, dst, arrowstyle="-|>", mutation_scale=20,
        color=color, lw=lw, linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2,
    ))


def main() -> None:
    Path(__file__).with_suffix(".mmd").write_text(MMD_SOURCE)

    with POSTER_STYLE():
        fig, ax = plt.subplots(figsize=(18, 8.0))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        # Layout
        #   ┌────┐  ┌──────┐  ┌──────┐  ┌─────────────┐
        #   │ in │→ │ util │→ │ grog │→ │   operator  │ ┐
        #   └────┘  └──────┘  └──────┘  └─────────────┘ │→ ┌────────┐ → ┌─────┐
        #                                ┌─────────────┐ │  │ solver │   │ out │
        #                                │   gadgets   │ ┘  └────────┘   └─────┘
        #                                └─────────────┘
        #               ┌─────────────────────────┐
        #               │   interop (sigpy ...)   │  ← dashed wrap from operator
        #               └─────────────────────────┘
        in_box   = _box(ax, 0.005, 0.55, 0.165, 0.20,
                        "non-Cartesian\nk-space + trajectory",
                        "#fff5e6", "#cc8800")
        util_box = _box(ax, 0.190, 0.55, 0.20, 0.20,
                        "utils\ncoil_compress · nlinv_calib",
                        "#fdf6e3", "#8c6d3f")
        grog_box = _box(ax, 0.410, 0.55, 0.18, 0.20,
                        "calib\nGrogInterpolator",
                        "#e8f0ff", "#1f77b4")
        op_box   = _box(ax, 0.610, 0.78, 0.22, 0.13,
                        "operator\nSparseFFT  ·  MaskedFFT",
                        "#e8f0ff", "#1f77b4")
        gad_box  = _box(ax, 0.610, 0.43, 0.22, 0.13,
                        "gadgets\nOffResonance  ·  Subspace",
                        "#fde6f0", "#c2185b")
        solv_box = _box(ax, 0.852, 0.55, 0.142, 0.30,
                        "_solve · _toep\n(cg / lsmr / pcg)\nToeplitz",
                        "#f0e6ff", "#6a1b9a")
        out_box  = _box(ax, 0.610, 0.20, 0.384, 0.13,
                        "coefficient images   ·   reconstruction x",
                        "#e6ffe6", "#2ca02c")
        io_box   = _box(ax, 0.190, 0.04, 0.40, 0.13,
                        "interop.sigpy  ·  interop.deepinv  ·  interop.mrpro",
                        "#e6f7ee", "#2ca02c")

        # Forward data-flow arrows (anchored to box edges).
        _arrow(ax, in_box["right"],   util_box["left"])
        _arrow(ax, util_box["right"], grog_box["left"])
        _arrow(ax, grog_box["right"], op_box["left"],  rad=0.20)
        _arrow(ax, grog_box["right"], gad_box["left"], rad=-0.20)
        _arrow(ax, op_box["right"],   solv_box["left"], rad=-0.15)
        _arrow(ax, gad_box["right"],  solv_box["left"], rad=0.15)
        _arrow(ax, solv_box["bottom"], out_box["top"], rad=0.15)

        # Interop wrap (dashed, green).
        _arrow(ax, op_box["bottom"], io_box["top"],
               linestyle="--", rad=-0.45, color="#2ca02c")
        ax.text(0.46, 0.30, "wrap", fontsize=12,
                color="#2ca02c", style="italic")

        ax.set_title("PyGROG package architecture", fontsize=22, pad=12)
        save_fig(fig, "fig_package")


if __name__ == "__main__":
    main()
