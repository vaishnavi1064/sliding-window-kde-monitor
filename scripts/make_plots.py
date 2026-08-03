"""Figures for the Phase 3 evaluation report.

Reads the cached score series, so this is fast and needs no replay. Every figure
also writes its underlying numbers to a CSV beside it -- the chart is never the
only way to read a value.

Usage:  python -m scripts.make_plots
"""

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import style
from evaluation.labels import labelled_windows
from evaluation.metrics import chance_detection, chance_p_value, evaluate
from scripts.run_evaluation import cache_path, rescore
from streaming.config import SETTINGS
from streaming.features import ANALOG_COLUMNS

FIGURES = Path(__file__).resolve().parent.parent / "docs" / "figures"
ALERTING_FRACTIONS = (0.20, 0.10, 0.05, 0.02, 0.01)
# Second identity channel alongside colour.
MARKERS = ("o", "s", "^")
METHOD_LABELS = {
    "swakde": "SW-AKDE (sketch)",
    "exact": "exact windowed KDE",
    "race": "un-windowed RACE",
}


def _load(method: str) -> pd.DataFrame | None:
    path = cache_path(method)
    return pd.read_parquet(path) if path.exists() else None


def figure_operating_points(methods: list[str]) -> None:
    """False-alarm rate against statistical significance.

    The pairing that matters: a low p-value is only meaningful at a low alarm
    rate, since a detector alarming constantly catches a 24h horizon for free.
    Detection counts are annotated rather than given a second axis.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rows = []

    for index, method in enumerate(methods):
        series = _load(method)
        if series is None:
            continue
        scores = rescore(series)
        timestamps = series["timestamp"]
        xs, ys, labels = [], [], []
        for fraction in ALERTING_FRACTIONS:
            threshold = float(np.quantile(scores, 1.0 - fraction))
            result = evaluate(timestamps, scores, threshold)
            total = result.false_alarms + result.detected_count
            p = chance_p_value(chance_detection(timestamps, total), result.detected_count)
            xs.append(result.false_alarms_per_day)
            ys.append(max(p, 1e-3))
            labels.append(result.detected_count)
            rows.append(
                {
                    "method": method,
                    "alerting_fraction": fraction,
                    "threshold": threshold,
                    "events_detected": result.detected_count,
                    "false_alarms_per_day": result.false_alarms_per_day,
                    "p_value": p,
                    "mean_lead_hours": result.mean_lead_hours,
                }
            )

        colour = style.SERIES[index]
        # Markers, not a connected line. p depends on both the detection count
        # and the alarm count, and episode-based counting makes it genuinely
        # non-monotone in the threshold -- joining the points would imply a
        # smooth trade-off curve that does not exist. Distinct marker shapes
        # also give identity a second channel beyond colour, which the aqua slot
        # needs since it sits below 3:1 contrast on this surface.
        marker = MARKERS[index % len(MARKERS)]
        sizes = [26 + 150 * f for f in ALERTING_FRACTIONS]
        # `exact` is drawn as a hollow ring, larger and behind. Where it and the
        # sketch agree -- which is the result -- you see the filled sketch marker
        # sitting inside the exact ring, instead of one silently covering the
        # other.
        if method == "exact":
            ax.scatter([x for x in xs], ys, s=[s * 3.2 for s in sizes], marker=marker,
                       facecolors="none", edgecolors=colour, linewidths=1.8,
                       label=METHOD_LABELS.get(method, method), zorder=2)
        else:
            ax.scatter(xs, ys, s=sizes, marker=marker, color=colour,
                       label=METHOD_LABELS.get(method, method), zorder=4 + index,
                       edgecolors=style.SURFACE, linewidths=1.2)
        # Label only the 5% operating point -- the one quoted in the report.
        five = ALERTING_FRACTIONS.index(0.05)
        ax.annotate(f"{labels[five]}/4", xy=(xs[five], ys[five]),
                    xytext=(12, -12 + 13 * index), textcoords="offset points",
                    color=colour, fontsize=8, fontweight="semibold")

    ax.axhline(0.05, color=style.STATUS_CRITICAL, linewidth=1.2, zorder=2)
    ax.annotate("p = 0.05", xy=(ax.get_xlim()[1], 0.05), xytext=(-2, 4),
                textcoords="offset points", ha="right", fontsize=7,
                color=style.STATUS_CRITICAL)

    ax.set_yscale("log")
    ax.set_xlabel("false alarms per day of normal operation")
    ax.set_ylabel("p vs a random detector with the same alarm count")
    ax.set_title("Detection is only meaningful at a low alarm rate")
    ax.annotate("marker size = alerting fraction (1%-20%)",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7,
                color=style.MUTED)
    ax.legend(loc="lower right")

    fig.savefig(FIGURES / "operating_points.png")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(FIGURES / "operating_points.csv", index=False)
    print("  operating_points.png")


def figure_scores_around_failures(method: str = "swakde") -> None:
    """Score in the +/-48h around each failure -- faceted, one panel per event.

    Four panels rather than four coloured series: the palette validates three
    categorical slots under all-pairs, and faceting is the documented move past
    that. Each panel is a single series, so it needs no legend.
    """
    series = _load(method)
    if series is None:
        return
    scores = rescore(series)
    times = pd.to_datetime(series["timestamp"])
    windows = labelled_windows()

    threshold = float(np.quantile(scores, 0.95))  # the 5% operating point
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4), sharey=True)
    rows = []

    for ax, window in zip(axes, windows):
        start = window.event.start
        mask = (times >= start - timedelta(hours=48)) & (times <= start + timedelta(hours=24))
        hours = (times[mask] - start).dt.total_seconds() / 3600.0
        values = scores[mask.to_numpy()]

        ax.plot(hours, values, color=style.SERIES[0], linewidth=1.4, zorder=3)
        ax.axvline(0.0, color=style.STATUS_CRITICAL, linewidth=1.2, zorder=2)
        ax.axhline(threshold, color=style.STATUS_WARNING, linewidth=1.1, zorder=2)
        ax.set_title(window.name.replace("failure-", "failure "), fontsize=9)
        ax.set_xlabel("hours from onset")
        rows.append({
            "failure": window.name,
            "peak_score_before_onset": float(values[hours < 0].max()) if (hours < 0).any() else np.nan,
            "peak_score_after_onset": float(values[hours >= 0].max()) if (hours >= 0).any() else np.nan,
        })

    axes[0].set_ylabel("anomaly score")
    axes[0].annotate("onset", xy=(0, axes[0].get_ylim()[1]), xytext=(3, -10),
                     textcoords="offset points", fontsize=7, color=style.STATUS_CRITICAL)
    axes[0].annotate("5% threshold", xy=(axes[0].get_xlim()[0], threshold),
                     xytext=(3, 4), textcoords="offset points", fontsize=7,
                     color=style.STATUS_WARNING)
    # Deliberately not "rises only at onset": failure 2 spends roughly -45h to
    # -15h above the threshold. The peak is at onset in all four cases, but
    # earlier excursions happen too and are not specific to failures -- which is
    # exactly why the wide-horizon lead times do not survive the chance test.
    fig.suptitle("Score peaks at onset; earlier excursions occur too and are not "
                 "failure-specific", y=1.04, fontsize=10,
                 fontweight="semibold", color=style.INK)

    fig.savefig(FIGURES / "scores_around_failures.png")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(FIGURES / "scores_around_failures.csv", index=False)
    print("  scores_around_failures.png")


def figure_memory_crossover() -> None:
    """Where sketching starts to beat storing the window.

    Measured sketch footprints (scripts/memory_check.py --rows-sweep) are flat
    lines: the sketch does not depend on dimension. Exact storage is
    window x dim x 8, so it rises. Both series are megabytes, so one axis.
    """
    window = SETTINGS.window_size
    measured = {50: 5.2, 400: 47.5}  # MB, window=3600, from --rows-sweep
    dims = np.logspace(0, 4, 200)
    exact_mb = window * dims * 8 / 1e6

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(dims, exact_mb, color=style.SERIES[0], label="exact: store the window",
            zorder=3)
    ax.annotate("exact: store the window", xy=(dims[-1], exact_mb[-1]),
                xytext=(-4, -12), textcoords="offset points", ha="right",
                fontsize=8, fontweight="semibold", color=style.SERIES[0])

    for offset, (rows, mb) in enumerate(sorted(measured.items())):
        colour = style.SERIES[1 + offset]
        ax.axhline(mb, color=colour, zorder=3, label=f"sketch, {rows} rows")
        ax.annotate(f"sketch, {rows} rows", xy=(dims[0], mb), xytext=(4, 5),
                    textcoords="offset points", fontsize=8, fontweight="semibold",
                    color=colour)
        crossover = mb * 1e6 / (window * 8)
        ax.plot([crossover], [mb], marker="o", color=colour, zorder=4,
                markeredgecolor=style.SURFACE, markeredgewidth=1.5)
        ax.annotate(f"crossover ~{crossover:,.0f} dim", xy=(crossover, mb),
                    xytext=(6, -14), textcoords="offset points", fontsize=7,
                    color=style.INK_SECONDARY)

    ax.axvline(len(ANALOG_COLUMNS), color=style.STATUS_CRITICAL, linewidth=1.2, zorder=2)
    ax.annotate(f"MetroPT: {len(ANALOG_COLUMNS)} channels",
                xy=(len(ANALOG_COLUMNS), exact_mb[0]), xytext=(6, 20),
                textcoords="offset points", fontsize=8,
                color=style.STATUS_CRITICAL, fontweight="semibold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("feature dimension")
    ax.set_ylabel("memory (MB)")
    ax.set_title(f"The sketch only saves memory above ~5x rows dimensions "
                 f"(window = {window:,})")
    ax.legend(loc="lower right")

    fig.savefig(FIGURES / "memory_crossover.png")
    plt.close(fig)
    pd.DataFrame(
        [{"rows": r, "sketch_mb": m, "crossover_dim": m * 1e6 / (window * 8)}
         for r, m in sorted(measured.items())]
    ).to_csv(FIGURES / "memory_crossover.csv", index=False)
    print("  memory_crossover.png")


def figure_effect_sizes() -> None:
    """Standardized channel shift before each failure -- faceted per event.

    Position encodes the signed effect, which is more precise than colour and
    sidesteps needing a diverging scale for 28 values.
    """
    from scripts.diagnose_failures import effect_sizes

    frame = pd.read_parquet(SETTINGS.parquet_path)
    table = effect_sizes(frame, ANALOG_COLUMNS, timedelta(hours=24))

    fig, axes = plt.subplots(1, len(table), figsize=(12.0, 3.6), sharey=True, sharex=True)
    positions = np.arange(len(ANALOG_COLUMNS))

    for ax, (_, row) in zip(axes, table.iterrows()):
        values = [row[c] for c in ANALOG_COLUMNS]
        ax.barh(positions, values, height=0.55, color=style.SERIES[0], zorder=3)
        ax.axvline(0.0, color=style.AXIS, linewidth=0.9, zorder=2)
        # |d| > 0.8 is conventionally a large effect.
        for edge in (-0.8, 0.8):
            ax.axvline(edge, color=style.STATUS_WARNING, linewidth=1.0, zorder=2)
        largest = max(abs(v) for v in values)
        ax.set_title(f"{row['failure'].replace('failure-', 'failure ')}\n"
                     f"max |d| = {largest:.2f}", fontsize=9)
        ax.set_xlabel("standardized shift (sigma)")
        ax.grid(axis="y", visible=False)

    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(ANALOG_COLUMNS, fontsize=8)
    axes[0].annotate("|d| = 0.8", xy=(0.8, len(positions) - 0.6), xytext=(3, 0),
                     textcoords="offset points", fontsize=7, color=style.STATUS_WARNING)
    fig.suptitle("Two of the four failures leave almost no trace in these channels",
                 y=1.05, fontsize=10, fontweight="semibold", color=style.INK)

    fig.savefig(FIGURES / "effect_sizes.png")
    plt.close(fig)
    table.to_csv(FIGURES / "effect_sizes.csv", index=False)
    print("  effect_sizes.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["swakde", "exact", "race"])
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    style.apply()

    print(f"Writing figures to {FIGURES.relative_to(FIGURES.parent.parent)} ...")
    figure_operating_points(args.methods)
    figure_scores_around_failures()
    figure_memory_crossover()
    if SETTINGS.parquet_path.exists():
        figure_effect_sizes()
    else:
        print("  (skipping effect sizes: dataset not present)")
    print("Each figure has a .csv beside it with the same numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
