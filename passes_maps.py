"""Impact pass maps (StatsBomb pitch layout)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from mplsoccer import Pitch

FIG_W, FIG_H = 7.2, 4.8
FIG_DPI = 220
FIG_W_COMPACT, FIG_H_COMPACT = 4.5, 3.0
FIG_DPI_COMPACT = 320
MAP_REF_WIDTH = 7.2
FIELD_X, FIELD_Y = 120.0, 80.0
PASS_DEST_HEATMAP_COLS = 6
PASS_DEST_HEATMAP_ROWS = 4
ARROW_WIDTH = 0.75
ARROW_HEADWIDTH = 1.15
ARROW_HEADLENGTH = 1.15
ARROW_ALPHA_EMPH = 0.82
PASS_START_MARKER_SIZE = 7

COLOR_PROGRESSIVE = "#7dd3fc"
COLOR_HIGHLY_PROGRESSIVE = "#fcd34d"
CMAP_PASS_DEST = LinearSegmentedColormap.from_list(
    "pass_dest", ["#1a1a2e", "#1e3a8a", "#3b82f6", "#fbbf24", "#ef4444"]
)


def _map_scale(fig_w: float) -> float:
    return fig_w / MAP_REF_WIDTH


def _base_pitch(*, figsize: tuple[float, float], dpi: int, bg: str = "#1a1a2e"):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=bg, line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor(bg)
    fig.set_dpi(dpi)
    return fig, ax, pitch


def _add_map_legend(ax, handles: list, *, fig_w: float) -> None:
    scale = _map_scale(fig_w)
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=6.0 * scale,
        labelspacing=0.35 * scale,
        borderpad=0.45 * scale,
        handlelength=1.9 * scale,
    )
    for text in leg.get_texts():
        text.set_color("white")
    leg.get_frame().set_alpha(0.90)


def _attack_arrow(fig, *, fig_w: float) -> None:
    scale = _map_scale(fig_w)
    fig.patches.append(
        FancyArrowPatch(
            (0.44, 0.045),
            (0.56, 0.045),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=10 * scale,
            linewidth=1.4 * scale,
            color="#aaaaaa",
        )
    )
    fig.text(
        0.50, 0.012, "Attacking Direction",
        ha="center", va="bottom", transform=fig.transFigure,
        fontsize=7.0 * scale, color="#aaaaaa",
    )


def _delicate_arrows(pitch, ax, x1, y1, x2, y2, color, scale: float, *, alpha: float) -> None:
    pitch.arrows(
        x1, y1, x2, y2,
        color=color,
        width=ARROW_WIDTH * scale,
        headwidth=ARROW_HEADWIDTH * scale,
        headlength=ARROW_HEADLENGTH * scale,
        ax=ax,
        zorder=3,
        alpha=alpha,
    )


def draw_impact_pass_map(
    passes,
    player_name: str,
    match_label: str = "todos os jogos",
    *,
    compact: bool = True,
):
    """Impact passes only — same visual language as the legacy pass map."""
    if compact:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT
    else:
        figsize = (FIG_W, FIG_H)
        dpi = FIG_DPI

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    subset = passes[passes["impact_success"] & passes["has_end"]].copy()
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    if subset.empty:
        ax.text(60, 40, "Sem passes de impact", ha="center", va="center", color="white", fontsize=9)
    else:
        for row in subset.itertuples(index=False):
            is_high = bool(row.high_impact_success)
            color, alpha = (
                (COLOR_HIGHLY_PROGRESSIVE, ARROW_ALPHA_EMPH)
                if is_high
                else (COLOR_PROGRESSIVE, ARROW_ALPHA_EMPH)
            )
            _delicate_arrows(
                pitch, ax,
                row.x_start, row.y_start, row.x_end, row.y_end,
                color, scale, alpha=alpha,
            )
            pitch.scatter(
                row.x_start, row.y_start,
                s=PASS_START_MARKER_SIZE, marker="o", color=color,
                edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=alpha,
            )

    legend_handles = [
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=1.4 * scale, label="Impact", alpha=0.80),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=1.4 * scale, label="High Impact", alpha=0.85),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PROGRESSIVE,
               markersize=4, linestyle="None", label="Origem do passe"),
    ]
    _add_map_legend(ax, legend_handles, fig_w=fig_w)
    ax.set_title(
        f"{player_name}\nPasses Impact · {match_label}",
        color="white", fontsize=8.4 * scale, pad=5,
    )
    _attack_arrow(fig, fig_w=fig_w)
    return fig


def draw_pass_destination_heatmap(
    passes,
    player_name: str,
    match_label: str = "todos os jogos",
    *,
    compact: bool = True,
):
    """6×4 heatmap of completed pass end locations."""
    if compact:
        figsize = (FIG_W_COMPACT, FIG_H_COMPACT)
        dpi = FIG_DPI_COMPACT
    else:
        figsize = (FIG_W, FIG_H)
        dpi = FIG_DPI

    fig_w = figsize[0]
    scale = _map_scale(fig_w)
    completed = passes[passes["has_end"] & passes["is_success"]].copy()
    fig, ax, pitch = _base_pitch(figsize=figsize, dpi=dpi)

    x_bins = np.linspace(0.0, FIELD_X, PASS_DEST_HEATMAP_COLS + 1)
    y_bins = np.linspace(0.0, FIELD_Y, PASS_DEST_HEATMAP_ROWS + 1)
    grid = np.zeros((PASS_DEST_HEATMAP_ROWS, PASS_DEST_HEATMAP_COLS), dtype=float)

    if not completed.empty:
        x_idx = np.clip(
            np.digitize(completed["x_end"].to_numpy(), x_bins, right=True) - 1,
            0,
            PASS_DEST_HEATMAP_COLS - 1,
        )
        y_idx = np.clip(
            np.digitize(completed["y_end"].to_numpy(), y_bins, right=True) - 1,
            0,
            PASS_DEST_HEATMAP_ROWS - 1,
        )
        for ix, iy in zip(x_idx, y_idx):
            grid[iy, ix] += 1.0

    vmax = max(float(grid.max()), 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)
    threshold = vmax * 0.45

    for iy in range(PASS_DEST_HEATMAP_ROWS):
        for ix in range(PASS_DEST_HEATMAP_COLS):
            value = float(grid[iy, ix])
            x0, x1 = x_bins[ix], x_bins[ix + 1]
            y0, y1 = y_bins[iy], y_bins[iy + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=CMAP_PASS_DEST(norm(value)),
                    edgecolor=(1, 1, 1, 0.22),
                    linewidth=0.5,
                    alpha=0.94,
                    zorder=2,
                )
            )
            if value > 0:
                ax.text(
                    (x0 + x1) / 2, (y0 + y1) / 2, f"{value:.1f}",
                    ha="center", va="center",
                    color="#000000" if value <= threshold else "#ffffff",
                    fontsize=6.8 * scale, fontweight="600", zorder=4,
                )

    pitch.draw(ax=ax)
    sm = plt.cm.ScalarMappable(cmap=CMAP_PASS_DEST, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02, shrink=0.55)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=6)
    plt.setp(cbar.ax.axes.get_yticklabels(), color="#ffffff")
    cbar.set_label("Passes", color="#c7cdda", fontsize=7 * scale)
    ax.set_title(
        f"{player_name}\nDestino dos passes · 6×4 · {match_label}",
        color="white", fontsize=8.2 * scale, pad=5,
    )
    _attack_arrow(fig, fig_w=fig_w)
    return fig
