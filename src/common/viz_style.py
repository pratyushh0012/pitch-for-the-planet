"""
Shared chart styling so every figure in the report and the web app reads as one
system.

Palette is the validated categorical set (fixed slot order, never cycled):
slot 1 blue, 2 orange, 3 aqua, 4 yellow, 5 magenta, 6 green, 7 violet, 8 red.
Colour is assigned to an ENTITY (AGE, rainfall, a given model) and never to a
rank, so a chart that drops a series never repaints the survivors.

Deliberate choice: NO dual-axis charts anywhere. Rainfall and case counts live
on different scales, so they get stacked panels sharing one x-axis instead of
two y-scales on one frame -- two y-scales let you manufacture any apparent
lead/lag relationship just by rescaling, which is exactly the claim this
project is trying to make honestly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"],
          C["magenta"], C["green"], C["violet"], C["red"]]

# Entity -> colour. Fixed for the whole project.
ROLE = {
    "age": C["blue"],
    "rain": C["aqua"],
    "pred": C["orange"],
    "baseline": C["violet"],
    "actual": C["blue"],
    "dhm": C["aqua"],
    "nasa": C["yellow"],
}

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3de"
SURFACE = "#ffffff"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 4,
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def finish(fig, path=None):
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
        return path
    return fig


def note(ax, text):
    """A small caption under an axis, in muted ink -- never series colour."""
    ax.annotate(text, xy=(0, -0.28), xycoords="axes fraction",
                fontsize=8.5, color=MUTED, va="top")


def break_gaps(x, *series, step=1):
    """Insert NaN wherever consecutive x values jump by more than `step`, so a
    line plot shows a genuine gap instead of bridging missing weeks with a
    straight segment that looks like real data."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    out_x = [x[0]]
    outs = [[np.asarray(s_, dtype=float)[0]] for s_ in series]
    for i in range(1, len(x)):
        if x[i] - x[i - 1] > step:
            out_x.append(np.nan)
            for o in outs:
                o.append(np.nan)
        out_x.append(x[i])
        for o, s_ in zip(outs, series):
            o.append(np.asarray(s_, dtype=float)[i])
    return (np.array(out_x), *[np.array(o) for o in outs])
