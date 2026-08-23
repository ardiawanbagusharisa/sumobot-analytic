"""
Compare raw (offline-recomputed) pacing factors between bots that had the dynamic
pacing filter applied ("Filtered") and the same bots' natural behavior with no
filter running ("Unfiltered").

Unlike plotting.pacing_target_analyzer (which reads PacingSegment rows the C#
runtime only logs while the dynamic pacing filter is steering a match), this module
works off compile.generator.batch_process_pacing / load_bot_pacing_factors output -
the 8 Threat/Tempo factors (CollisionRatio, AbilityRatio, Angle, SafeDistance,
ActionIntensity, ActionDensity, BotsDistance, Velocity) recomputed offline from raw
Action/Collision/position event rows using the same formula for both groups. That
makes it the one pacing metric that exists on both sides: an unfiltered run has
Action/Collision rows like any other, but never emits a PacingSegment row.

Expects a tidy DataFrame (pandas or polars) with one row per (Bot, Group, TimeBin,
...) and one column per factor in FACTOR_NAMES - i.e. compile.generator.
load_bot_pacing_factors's output, concatenated across bots/groups.
"""
import json
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .analyzer_config import get_theme_color

FACTOR_NAMES = [
    "CollisionRatio", "AbilityRatio", "Angle", "SafeDistance",
    "ActionIntensity", "ActionDensity", "BotsDistance", "Velocity",
]

THREAT_FACTORS = {"CollisionRatio", "AbilityRatio", "Angle", "SafeDistance"}
TEMPO_FACTORS = {"ActionIntensity", "ActionDensity", "BotsDistance", "Velocity"}

# Maps the merged composite's metric names to the engine's own *Scaled columns from
# compile.generator.load_filtered_target_tracking (Actual = ground truth, Target =
# the predefined curve the filter steers towards).
METRIC_TO_SCALED = {
    "Threat": ("ActualThreatScaled", "TargetThreatScaled"),
    "Tempo": ("ActualTempoScaled", "TargetTempoScaled"),
    "OverallPacing": ("ActualOverallPacingScaled", "TargetOverallPacingScaled"),
}


def load_pacing_target_curves(sim_targets_dir):
    """
    Load the engine's predefined per-tick Threat/Tempo target curves straight from
    Resources/.../Sim_Targets/<subfolder>/<PacingTarget>.json (ThreatTargets/
    TempoTargets arrays, e.g. linear_increase.json) - the deterministic curve the
    dynamic pacing filter steers towards, indexed by LocalSegmentIndex. These values
    were verified to match the engine's own TargetThreatScaled/TargetTempoScaled
    columns exactly at every LocalSegmentIndex any round actually reached, so using
    the JSON directly (instead of reconstructing Target purely from logged
    PacingSegment rows) lets the Target line span its full defined length even when
    every observed round ended early - see _draw_target_tracking_panels.

    Args:
        sim_targets_dir: Folder of pacing target *.json files (Resources/.../
            Sim_Targets/<subfolder> - must be the subfolder actually used for the
            run being plotted, since different subfolders define different curve
            sets/names, e.g. Experiments vs Experiments_real).

    Returns:
        Dict of {pacing_target_name: {"Threat": np.array, "Tempo": np.array,
        "OverallPacing": np.array}}, one entry per *.json file in sim_targets_dir.
    """
    curves = {}
    for fname in sorted(os.listdir(sim_targets_dir)):
        if not fname.endswith(".json"):
            continue
        name = os.path.splitext(fname)[0]
        with open(os.path.join(sim_targets_dir, fname)) as f:
            data = json.load(f)
        threat = np.asarray(data["ThreatTargets"], dtype=float)
        tempo = np.asarray(data["TempoTargets"], dtype=float)
        curves[name] = {
            "Threat": threat,
            "Tempo": tempo,
            "OverallPacing": (threat + tempo) / 2.0,
        }
    return curves


def _resolve_curve_key(pacing_target, available_keys):
    """
    Match a df_target_tracking PacingTarget value against the clean curve names
    load_pacing_target_curves loaded (its *.json filename stems), tolerating a
    PacingConstraint glued onto the end - some simulation batches' config folders
    are named "Pacing_<target>_constraint_<constraint>" (no "|" separator; see
    compile.log_to_parquet.parse_pacing_folder_name), which left PacingTarget itself
    holding the compound string (e.g. "lin_down_06_04_constraint_avg_bot") in any
    already-batched parquet - re-running batch_process_pacing_segments would fix it
    at the source, but that means reprocessing the whole simulation batch just to
    draw a chart correctly, so this resolves it here instead, purely from data
    already in hand.

    Tries, in order:
      1. Exact match - the common case for cleanly-separated ("|") folder names.
      2. Split on the literal "_constraint_" separator and match the part before it -
         the specific glued convention seen in practice.
      3. Longest available_keys entry that's a strict prefix of pacing_target,
         followed by "_" - covers any other stray glued suffix without needing to
         know its exact shape upfront.

    Returns the matching key from available_keys, or None if nothing matches.
    """
    if pacing_target in available_keys:
        return pacing_target
    if "_constraint_" in pacing_target:
        candidate = pacing_target.split("_constraint_", 1)[0]
        if candidate in available_keys:
            return candidate
    prefix_matches = [k for k in available_keys if pacing_target.startswith(f"{k}_")]
    if prefix_matches:
        return max(prefix_matches, key=len)
    return None


def _get_timer(df, default=None):
    """
    Pull the configured Timer (match length in seconds) out of df's Timer column -
    constant across every row once config_filter has pinned it to one value (see
    compile.generator.load_filtered_target_tracking). Used to spread the
    deterministic Target curve (indexed 0..N-1, see load_pacing_target_curves) evenly
    across [0, Timer], independent of how far any observed round's LocalSegmentIndex
    actually reached.
    """
    values = df["Timer"].dropna().unique()
    if len(values) == 0:
        return default
    if len(values) > 1:
        print(f"⚠️ Multiple Timer values in target-tracking data ({sorted(values)}); using the first.")
    return float(values[0])


def _to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _stats(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = int(values.size)
    mean = float(np.mean(values)) if n else float("nan")
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n}


def _linear_trend(x, y):
    """Least-squares (slope, intercept) for y vs x, or None if fewer than 2 valid points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return None
    return tuple(np.polyfit(x[mask], y[mask], 1))


def _mae(actual, target):
    """
    Mean absolute error between two same-length series, ignoring NaN pairs. Used
    (instead of squared error) for the per-segment Actual-vs-Target tracking
    annotation: with a target curve that swings from 0 to 1 and back within the
    observed window while Actual stays comparatively flat, squaring dilutes the
    error - most rows sit near the curve's low tails where the gap is small, so a
    handful of large near-peak misses get averaged away. Mean absolute error scales
    linearly with the gap instead, so it tracks what's visible on the chart.
    """
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = ~np.isnan(actual) & ~np.isnan(target)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - target[mask])))


def plot_pacing_factor_comparison(df, bot, width=16, height=8):
    """
    One figure for a single bot: 2x4 grid of the 8 pacing factors, each subplot a
    Filtered-vs-Unfiltered box plot for that factor.

    Args:
        df: Tidy DataFrame with columns "Bot", "Group" ("Filtered"/"Unfiltered"),
            and one column per FACTOR_NAMES entry (see compile.generator.
            load_bot_pacing_factors)
        bot: Bot name to filter to (matches df["Bot"])

    Returns:
        Matplotlib Figure, or None if no data matched.
    """
    df = _to_pandas(df)
    bot_df = df[df["Bot"] == bot]
    if bot_df.empty:
        print(f"⚠️ No pacing factor data found for bot={bot}")
        return None

    groups = [g for g in ["Filtered", "Unfiltered"] if g in bot_df["Group"].unique()]
    if not groups:
        print(f"⚠️ No 'Filtered'/'Unfiltered' Group values found for bot={bot}")
        return None
    colors = {"Filtered": get_theme_color("primary"), "Unfiltered": get_theme_color("secondary")}

    fig, axes = plt.subplots(2, 4, figsize=(width, height))
    axes = axes.flatten()

    for ax, factor in zip(axes, FACTOR_NAMES):
        data = [bot_df.loc[bot_df["Group"] == g, factor].dropna().values for g in groups]
        bp = ax.boxplot(data, tick_labels=groups, patch_artist=True, showmeans=True, widths=0.6)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(colors.get(g, get_theme_color("bar_default")))
            patch.set_alpha(0.6)

        stats_by_group = {g: _stats(bot_df.loc[bot_df["Group"] == g, factor]) for g in groups}
        lines = [
            f"{g}: μ={s['mean']:.3f}  σ={s['std']:.3f}  (n={s['n']})"
            for g, s in stats_by_group.items() if s["n"] > 0
        ]
        ax.text(
            0.5, -0.16, "\n".join(lines), transform=ax.transAxes, fontsize=7,
            ha="center", va="top", family="monospace",
        )

        kind = "Threat" if factor in THREAT_FACTORS else "Tempo"
        ax.set_title(f"{factor} ({kind})", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(f"{bot}: Filtered vs Unfiltered Pacing Factors", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def _bots_in_both_groups(df):
    bots_per_group = df.groupby("Group")["Bot"].unique().to_dict()
    if "Filtered" not in bots_per_group or "Unfiltered" not in bots_per_group:
        print("⚠️ Need both 'Filtered' and 'Unfiltered' rows in df to compare.")
        return []
    return sorted(set(bots_per_group["Filtered"]) & set(bots_per_group["Unfiltered"]))


def plot_all_pacing_factor_comparisons(df, output_dir=None, width=16, height=8):
    """
    Iterate every bot present under both "Filtered" and "Unfiltered" Group values
    and produce one comparison figure each.

    Args:
        df: Tidy DataFrame (see plot_pacing_factor_comparison)
        output_dir: If provided, saves "<bot>_filtered_vs_unfiltered.png" for each
            bot into this dir

    Returns:
        Dict of {bot_name: Figure}
    """
    df = _to_pandas(df)
    if df is None or df.empty:
        print("⚠️ No pacing factor data to compare.")
        return {}

    bots = _bots_in_both_groups(df)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    figs = {}
    for bot in bots:
        fig = plot_pacing_factor_comparison(df, bot, width=width, height=height)
        if fig is None:
            continue
        figs[bot] = fig
        if output_dir:
            out_path = os.path.join(output_dir, f"{bot}_filtered_vs_unfiltered.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: {out_path}")
    return figs


def plot_pacing_factor_timeseries(df, bot, width=16, height=8):
    """
    One figure for a single bot: 2x4 grid of the 8 pacing factors, each subplot a
    Filtered-vs-Unfiltered line (mean +/- std band, shaded) over TimeBin - i.e. one
    point per 1s segment across the match (e.g. 30 segments for a Timer=30 config),
    so you can see WHEN in the match the two groups' behavior diverges, not just
    whether it diverges on average (see plot_pacing_factor_comparison for that).

    Args:
        df: Tidy DataFrame with columns "Bot", "Group" ("Filtered"/"Unfiltered"),
            "TimeBin", and one column per FACTOR_NAMES entry (see compile.generator.
            load_bot_pacing_factors)
        bot: Bot name to filter to (matches df["Bot"])

    Returns:
        Matplotlib Figure, or None if no data matched.
    """
    df = _to_pandas(df)
    bot_df = df[df["Bot"] == bot]
    if bot_df.empty:
        print(f"⚠️ No pacing factor data found for bot={bot}")
        return None

    groups = [g for g in ["Filtered", "Unfiltered"] if g in bot_df["Group"].unique()]
    if not groups:
        print(f"⚠️ No 'Filtered'/'Unfiltered' Group values found for bot={bot}")
        return None
    colors = {"Filtered": get_theme_color("primary"), "Unfiltered": get_theme_color("secondary")}

    fig, axes = plt.subplots(2, 4, figsize=(width, height))
    axes = axes.flatten()

    for ax, factor in zip(axes, FACTOR_NAMES):
        for g in groups:
            g_df = bot_df[bot_df["Group"] == g]
            agg = (
                g_df.groupby("TimeBin")[factor]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("TimeBin")
            )
            if agg.empty:
                continue
            x = agg["TimeBin"].values
            mean = agg["mean"].values
            std = np.nan_to_num(agg["std"].values)
            color = colors.get(g, get_theme_color("bar_default"))
            ax.plot(x, mean, color=color, linewidth=2, marker="o", markersize=3, label=g, zorder=3)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, zorder=1)

        kind = "Threat" if factor in THREAT_FACTORS else "Tempo"
        ax.set_title(f"{factor} ({kind})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Segment (s)", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"{bot}: Filtered vs Unfiltered Pacing Over Match Segments", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_all_pacing_factor_timeseries(df, output_dir=None, width=16, height=8):
    """
    Iterate every bot present under both "Filtered" and "Unfiltered" Group values
    and produce one over-time comparison figure each (see
    plot_pacing_factor_timeseries).

    Returns:
        Dict of {bot_name: Figure}
    """
    df = _to_pandas(df)
    if df is None or df.empty:
        print("⚠️ No pacing factor data to compare.")
        return {}

    bots = _bots_in_both_groups(df)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    figs = {}
    for bot in bots:
        fig = plot_pacing_factor_timeseries(df, bot, width=width, height=height)
        if fig is None:
            continue
        figs[bot] = fig
        if output_dir:
            out_path = os.path.join(output_dir, f"{bot}_filtered_vs_unfiltered_timeseries.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: {out_path}")
    return figs


def _add_overall_pacing_columns(df):
    """
    Merge the 8 raw factors down into a single Threat/Tempo/OverallPacing composite
    per row: min-max normalize each factor into [0, 1] (fit across the full df - both
    Filtered and Unfiltered rows together, so the two groups land on the same scale),
    then average THREAT_FACTORS into "Threat", TEMPO_FACTORS into "Tempo", and the two
    of those into "OverallPacing" - the same Threat/Tempo grouping the engine itself
    uses (see compile.generator.process_pacing_factors_timebins_single_csv).

    This is NOT the engine's real weighted formula - those weights aren't available
    offline, only the 8 raw factors - so treat Threat/Tempo/OverallPacing here as an
    equal-weight approximation for visual comparison, not the actual in-engine score.
    """
    df = df.copy()
    for factor in FACTOR_NAMES:
        col = df[factor].astype(float)
        lo, hi = col.min(), col.max()
        span = hi - lo
        df[f"_{factor}_norm"] = (col - lo) / span if span > 0 else 0.0
    df["Threat"] = df[[f"_{f}_norm" for f in sorted(THREAT_FACTORS)]].mean(axis=1)
    df["Tempo"] = df[[f"_{f}_norm" for f in sorted(TEMPO_FACTORS)]].mean(axis=1)
    df["OverallPacing"] = df[["Threat", "Tempo"]].mean(axis=1)
    return df


def plot_overall_pacing_timeseries(df, bot, width=15, height=5):
    """
    One figure for a single bot: 3 panels (Threat, Tempo, OverallPacing - see
    _add_overall_pacing_columns), each a Filtered-vs-Unfiltered line (mean +/- std
    band) over TimeBin - the same over-time view as plot_pacing_factor_timeseries,
    but collapsed from 8 raw factors down to the merged composite.

    Args:
        df: Tidy DataFrame (see plot_pacing_factor_timeseries)
        bot: Bot name to filter to (matches df["Bot"])

    Returns:
        Matplotlib Figure, or None if no data matched.
    """
    df = _to_pandas(df)
    bot_df = df[df["Bot"] == bot]
    if bot_df.empty:
        print(f"⚠️ No pacing factor data found for bot={bot}")
        return None
    bot_df = _add_overall_pacing_columns(bot_df)

    groups = [g for g in ["Filtered", "Unfiltered"] if g in bot_df["Group"].unique()]
    if not groups:
        print(f"⚠️ No 'Filtered'/'Unfiltered' Group values found for bot={bot}")
        return None
    colors = {"Filtered": get_theme_color("primary"), "Unfiltered": get_theme_color("secondary")}

    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    for ax, metric in zip(axes, ["Threat", "Tempo", "OverallPacing"]):
        for g in groups:
            g_df = bot_df[bot_df["Group"] == g]
            agg = (
                g_df.groupby("TimeBin")[metric]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("TimeBin")
            )
            if agg.empty:
                continue
            x = agg["TimeBin"].values
            mean = agg["mean"].values
            std = np.nan_to_num(agg["std"].values)
            color = colors.get(g, get_theme_color("bar_default"))
            ax.plot(x, mean, color=color, linewidth=2.5, marker="o", markersize=3, label=g, zorder=3)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, zorder=1)

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Segment (s)", fontsize=9)
        ax.set_ylabel("Normalized (0-1)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"{bot}: Filtered vs Unfiltered — Overall Pacing (equal-weight Threat/Tempo blend)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def plot_all_overall_pacing_timeseries(df, output_dir=None, width=15, height=5):
    """
    Iterate every bot present under both "Filtered" and "Unfiltered" Group values
    and produce one merged Threat/Tempo/OverallPacing over-time figure each (see
    plot_overall_pacing_timeseries).

    Returns:
        Dict of {bot_name: Figure}
    """
    df = _to_pandas(df)
    if df is None or df.empty:
        print("⚠️ No pacing factor data to compare.")
        return {}

    bots = _bots_in_both_groups(df)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    figs = {}
    for bot in bots:
        fig = plot_overall_pacing_timeseries(df, bot, width=width, height=height)
        if fig is None:
            continue
        figs[bot] = fig
        if output_dir:
            out_path = os.path.join(output_dir, f"{bot}_filtered_vs_unfiltered_overall.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: {out_path}")
    return figs


def _draw_target_tracking_panels(axes, track_df, unfiltered_df, pacing_target=None,
                                  target_curves=None, target_pool_df=None):
    """
    Shared per-metric drawing logic for plot_filtered_target_tracking and
    plot_filtered_target_tracking_merged: draws the Target/Actual/trend/Unfiltered
    lines and the MAE annotation into each of the 3 given axes (Threat, Tempo,
    OverallPacing, in that order). track_df/unfiltered_df are grouped by TimeBin as
    given - callers control pooling (single bot vs every applied bot) by how they
    filter these DataFrames before calling this.

    When target_curves has an entry for pacing_target (see load_pacing_target_curves),
    the Target line is drawn straight from that deterministic per-tick curve, its N
    points spread evenly across [0, Timer] (Timer read off target_pool_df/track_df's
    own Timer column - see _get_timer) - so it spans the full configured match length
    regardless of how far any observed round's PacingSegment log actually reached.
    Falls back to reconstructing Target from target_pool_df's logged TimeBin/*Scaled
    columns (defaults to track_df; bounded by whichever rounds were actually
    observed) when no curve is available for this pacing_target, or Timer can't be
    determined.
    """
    actual_color = get_theme_color("primary")
    target_color = get_theme_color("danger")
    unfiltered_color = get_theme_color("secondary")
    target_pool_df = track_df if target_pool_df is None else target_pool_df
    curve = None
    if target_curves:
        curve_key = _resolve_curve_key(pacing_target, target_curves.keys())
        curve = target_curves.get(curve_key) if curve_key else None
    timer = _get_timer(target_pool_df) if curve is not None else None
    if curve is not None and timer is None:
        print(f"⚠️ No Timer value found for PacingTarget={pacing_target}; falling back to logged Target rows.")
        curve = None

    for ax, metric in zip(axes, ["Threat", "Tempo", "OverallPacing"]):
        actual_col, target_col = METRIC_TO_SCALED[metric]

        actual_agg = (
            track_df.groupby("TimeBin")
            .agg(Actual_mean=(actual_col, "mean"), Actual_std=(actual_col, "std"))
            .reset_index()
            .sort_values("TimeBin")
        )
        x = actual_agg["TimeBin"].values
        actual_mean = actual_agg["Actual_mean"].values
        actual_std = np.nan_to_num(actual_agg["Actual_std"].values)

        if curve is not None:
            target_mean = curve[metric]
            target_x = (np.arange(len(target_mean)) + 0.5) * (timer / len(target_mean))
        else:
            target_agg = (
                target_pool_df.groupby("TimeBin")[target_col]
                .mean()
                .reset_index()
                .sort_values("TimeBin")
            )
            target_x = target_agg["TimeBin"].values
            target_mean = target_agg[target_col].values

        ax.plot(target_x, target_mean, color=target_color, linewidth=2, linestyle="--", label="Target", zorder=2)
        ax.plot(x, actual_mean, color=actual_color, linewidth=2.5, marker="o", markersize=3,
                label="Filtered Actual", zorder=3)
        ax.fill_between(x, actual_mean - actual_std, actual_mean + actual_std,
                         color=actual_color, alpha=0.15, zorder=1)

        trend = _linear_trend(x, actual_mean)
        if trend is not None:
            slope, intercept = trend
            trend_y = slope * x + intercept
            ax.plot(x, trend_y, color=actual_color, linewidth=1.5, linestyle=":", alpha=0.8,
                    label=f"Actual Trend ({slope:+.4f}/s)", zorder=2.5)

        if unfiltered_df is not None and not unfiltered_df.empty:
            u_agg = unfiltered_df.groupby("TimeBin")[metric].mean().reset_index().sort_values("TimeBin")
            ax.plot(u_agg["TimeBin"], u_agg[metric], color=unfiltered_color, linewidth=1.5,
                    linestyle="-.", alpha=0.7, label="Unfiltered (approx.)", zorder=1.5)

        mae = _mae(track_df[actual_col], track_df[target_col])
        n_segments = int(track_df[actual_col].notna().sum())
        ax.text(
            0.02, 0.02, f"MAE(Actual, Target) = {mae:.4f}\n(n={n_segments} segments)",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999", alpha=0.85), zorder=5,
        )

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Segment (s)", fontsize=9)
        ax.set_ylabel("Scaled (0-1)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=7, loc="upper right")


def plot_filtered_target_tracking(df_factors, df_target_tracking, bot, pacing_target, width=15, height=5,
                                   target_curves=None):
    """
    One figure for a single (bot, PacingTarget) pair: 3 panels (Threat, Tempo,
    OverallPacing), each showing the engine's own ground-truth Actual vs Target
    curves for that PacingTarget - i.e. does the dynamic pacing filter actually
    track the curve it was steering towards - with:
      - Target: the predefined curve for this PacingTarget (near-deterministic
        given Timer + elapsed time, so no meaningful spread across rounds/matches).
        Drawn from target_curves (see load_pacing_target_curves) when given, so it
        spans its full defined length even if every observed round for this
        bot/PacingTarget ended before the match timer ran out; otherwise
        reconstructed from logged rows and bounded by whatever those rounds reached.
      - Filtered Actual: mean +/- std band across every matching round, plus a
        dotted linear trend line (labeled with slope) showing whether tracking
        systematically drifts over the match rather than just being noisy
      - MAE(Actual, Target): annotated per panel - single-number tracking error,
        lower is better
      - Unfiltered: the same bot's offline-approximated composite (see
        _add_overall_pacing_columns) as a light reference line, IF df_factors is
        given - not on the same calibrated [PacingMinUsed, PacingMaxUsed] scale as
        Target/Actual (no engine formula exists for unfiltered runs), so treat it as
        directional context only, not a strict apples-to-apples comparison

    Args:
        df_factors: Tidy DataFrame from compile.generator.load_bot_pacing_factors
            (for the Unfiltered reference line), or None to omit it
        df_target_tracking: Raw (unaggregated) DataFrame from compile.generator.
            load_filtered_target_tracking
        bot, pacing_target: Which Bot / PacingTarget to plot
        target_curves: Optional dict from load_pacing_target_curves - when given and
            it has an entry for pacing_target, draws the Target line from the
            deterministic curve instead of from logged rows (see
            _draw_target_tracking_panels)

    Returns:
        Matplotlib Figure, or None if no data matched.
    """
    all_track_df = _to_pandas(df_target_tracking)
    track_df = all_track_df[(all_track_df["Bot"] == bot) & (all_track_df["PacingTarget"] == pacing_target)]
    if track_df.empty:
        print(f"⚠️ No target-tracking data for bot={bot}, PacingTarget={pacing_target}")
        return None

    # Every bot sharing this PacingTarget, not just `bot` - so the seg-duration
    # estimate/fallback Target aggregation (see _draw_target_tracking_panels) is
    # based on as broad a sample as possible, even if `bot`'s own rounds all ended
    # early.
    target_pool_df = all_track_df[all_track_df["PacingTarget"] == pacing_target]

    unfiltered_df = None
    if df_factors is not None:
        factors_df = _to_pandas(df_factors)
        factors_bot_df = factors_df[factors_df["Bot"] == bot]
        if not factors_bot_df.empty:
            # Fit normalization across both Groups for this bot (not just Unfiltered
            # rows) so this line matches plot_overall_pacing_timeseries's Unfiltered
            # curve exactly, rather than a differently-scaled recomputation.
            factors_bot_df = _add_overall_pacing_columns(factors_bot_df)
            unfiltered_df = factors_bot_df[factors_bot_df["Group"] == "Unfiltered"]

    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    _draw_target_tracking_panels(axes, track_df, unfiltered_df, pacing_target=pacing_target,
                                  target_curves=target_curves, target_pool_df=target_pool_df)

    fig.suptitle(f"{bot} — {pacing_target}: Filtered Actual vs Target (engine ground truth)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def plot_all_filtered_target_tracking(df_factors, df_target_tracking, output_dir=None, width=15, height=5,
                                       sim_targets_dir=None):
    """
    Iterate every (Bot, PacingTarget) pair present in df_target_tracking and produce
    one Actual-vs-Target tracking figure each (see plot_filtered_target_tracking).

    Args:
        sim_targets_dir: Optional folder of pacing target *.json files (Resources/
            .../Sim_Targets/<subfolder> for the run being plotted) - when given, the
            Target line is drawn from the deterministic curve (see
            load_pacing_target_curves) instead of being bounded by logged rows.

    Returns:
        Dict of {(bot_name, pacing_target): Figure}
    """
    track_df = _to_pandas(df_target_tracking)
    if track_df is None or track_df.empty:
        print("⚠️ No target-tracking data to plot.")
        return {}

    target_curves = load_pacing_target_curves(sim_targets_dir) if sim_targets_dir else None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    figs = {}
    for bot in sorted(track_df["Bot"].dropna().unique()):
        targets = sorted(track_df.loc[track_df["Bot"] == bot, "PacingTarget"].dropna().unique())
        for pacing_target in targets:
            fig = plot_filtered_target_tracking(df_factors, track_df, bot, pacing_target, width=width, height=height,
                                                 target_curves=target_curves)
            if fig is None:
                continue
            figs[(bot, pacing_target)] = fig
            if output_dir:
                safe_target = re.sub(r"[^\w.\-]+", "_", pacing_target)
                out_path = os.path.join(output_dir, f"{bot}_{safe_target}_target_tracking.png")
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"✅ Saved: {out_path}")
    return figs


def plot_filtered_target_tracking_merged(df_factors, df_target_tracking, pacing_target, bots=None, width=15, height=5,
                                          target_curves=None):
    """
    Same as plot_filtered_target_tracking, but pools every applied bot together into
    one Actual/Target line per PacingTarget instead of a separate figure per bot -
    "how well does the filter track this curve, across all applied bots combined"
    rather than bot-by-bot. Mirrors plotting.pacing_target_analyzer.
    plot_pacing_target_tracking's default (applied_bot=None pools every Applied-role
    bot together) rather than introducing a different pooling convention.

    Args:
        df_factors: Tidy DataFrame from compile.generator.load_bot_pacing_factors
            (for the Unfiltered reference line), or None to omit it
        df_target_tracking: Raw (unaggregated) DataFrame from compile.generator.
            load_filtered_target_tracking
        pacing_target: Which PacingTarget curve to plot
        bots: Optional list to restrict which bots get pooled (default: every bot
            present in df_target_tracking for this pacing_target)
        target_curves: Optional dict from load_pacing_target_curves - when given and
            it has an entry for pacing_target, draws the Target line from the
            deterministic curve instead of from logged rows (see
            _draw_target_tracking_panels)

    Returns:
        Matplotlib Figure, or None if no data matched.
    """
    track_df = _to_pandas(df_target_tracking)
    track_df = track_df[track_df["PacingTarget"] == pacing_target]
    # Every bot sharing this PacingTarget, before the optional `bots` restriction -
    # so the seg-duration estimate/fallback Target aggregation (see
    # _draw_target_tracking_panels) is based on as broad a sample as possible, even
    # when `bots` narrows Actual down to bots whose rounds all ended early.
    target_pool_df = track_df
    if bots is not None:
        track_df = track_df[track_df["Bot"].isin(bots)]
    if track_df.empty:
        print(f"⚠️ No target-tracking data for PacingTarget={pacing_target}")
        return None

    bots_in_data = sorted(track_df["Bot"].dropna().unique())

    unfiltered_df = None
    if df_factors is not None:
        factors_df = _to_pandas(df_factors)
        factors_df = factors_df[factors_df["Bot"].isin(bots_in_data)]
        if not factors_df.empty:
            # Fit normalization across every pooled bot's Filtered+Unfiltered rows
            # together, consistent with pooling the same bots on the Actual/Target side.
            factors_df = _add_overall_pacing_columns(factors_df)
            unfiltered_df = factors_df[factors_df["Group"] == "Unfiltered"]

    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    _draw_target_tracking_panels(axes, track_df, unfiltered_df, pacing_target=pacing_target,
                                  target_curves=target_curves, target_pool_df=target_pool_df)

    fig.suptitle(
        f"Applied ({'/'.join(bots_in_data)}) — {pacing_target}: Filtered Actual vs Target (engine ground truth)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def plot_all_filtered_target_tracking_merged(df_factors, df_target_tracking, bots=None, output_dir=None, width=15,
                                              height=5, sim_targets_dir=None):
    """
    Iterate every PacingTarget present in df_target_tracking and produce one
    bot-pooled Actual-vs-Target tracking figure each (see
    plot_filtered_target_tracking_merged).

    Args:
        sim_targets_dir: Optional folder of pacing target *.json files (Resources/
            .../Sim_Targets/<subfolder> for the run being plotted) - when given, the
            Target line is drawn from the deterministic curve (see
            load_pacing_target_curves) instead of being bounded by logged rows.

    Returns:
        Dict of {pacing_target: Figure}
    """
    track_df = _to_pandas(df_target_tracking)
    if track_df is None or track_df.empty:
        print("⚠️ No target-tracking data to plot.")
        return {}
    if bots is not None:
        track_df = track_df[track_df["Bot"].isin(bots)]

    target_curves = load_pacing_target_curves(sim_targets_dir) if sim_targets_dir else None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    figs = {}
    for pacing_target in sorted(track_df["PacingTarget"].dropna().unique()):
        fig = plot_filtered_target_tracking_merged(df_factors, track_df, pacing_target, bots=bots, width=width,
                                                     height=height, target_curves=target_curves)
        if fig is None:
            continue
        figs[pacing_target] = fig
        if output_dir:
            safe_target = re.sub(r"[^\w.\-]+", "_", pacing_target)
            out_path = os.path.join(output_dir, f"applied_{safe_target}_target_tracking.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: {out_path}")
    return figs


def _draw_target_tracking_panels_by_bot(axes, track_df, unfiltered_by_bot, bot_colors, pacing_target=None,
                                         target_curves=None, target_pool_df=None,
                                         include_target=True, include_unfiltered=True, include_trend=True):
    """
    Per-bot counterpart to _draw_target_tracking_panels: instead of pooling every
    applied bot into one Actual/Target line, draws one Filtered-Actual line (+trend,
    +Unfiltered reference) per bot, each in its own color from bot_colors, plus a
    single Target line shared across bots (same curve regardless of which bot). See
    plot_filtered_target_tracking_by_bot for the include_* toggles.

    unfiltered_by_bot is {bot: DataFrame} (already run through
    _add_overall_pacing_columns and filtered to Group=="Unfiltered"), or None to
    omit Unfiltered lines entirely - a bot missing from the dict is just skipped.
    """
    target_color = get_theme_color("danger")
    target_pool_df = track_df if target_pool_df is None else target_pool_df
    curve = None
    if include_target and target_curves:
        curve_key = _resolve_curve_key(pacing_target, target_curves.keys())
        curve = target_curves.get(curve_key) if curve_key else None
    timer = _get_timer(target_pool_df) if curve is not None else None
    if curve is not None and timer is None:
        print(f"⚠️ No Timer value found for PacingTarget={pacing_target}; falling back to logged Target rows.")
        curve = None

    bots_in_data = sorted(track_df["Bot"].dropna().unique())

    for ax, metric in zip(axes, ["Threat", "Tempo", "OverallPacing"]):
        actual_col, target_col = METRIC_TO_SCALED[metric]

        if include_target:
            if curve is not None:
                target_mean = curve[metric]
                target_x = (np.arange(len(target_mean)) + 0.5) * (timer / len(target_mean))
            else:
                target_agg = (
                    target_pool_df.groupby("TimeBin")[target_col]
                    .mean()
                    .reset_index()
                    .sort_values("TimeBin")
                )
                target_x = target_agg["TimeBin"].values
                target_mean = target_agg[target_col].values
            ax.plot(target_x, target_mean, color=target_color, linewidth=2, linestyle="--",
                    label="Target", zorder=2)

        mae_lines = []
        for bot in bots_in_data:
            color = bot_colors[bot]
            bot_df = track_df[track_df["Bot"] == bot]

            actual_agg = (
                bot_df.groupby("TimeBin")
                .agg(Actual_mean=(actual_col, "mean"), Actual_std=(actual_col, "std"))
                .reset_index()
                .sort_values("TimeBin")
            )
            x = actual_agg["TimeBin"].values
            actual_mean = actual_agg["Actual_mean"].values
            actual_std = np.nan_to_num(actual_agg["Actual_std"].values)

            ax.plot(x, actual_mean, color=color, linewidth=2, marker="o", markersize=3,
                    label=f"{bot} Filtered", zorder=3)
            ax.fill_between(x, actual_mean - actual_std, actual_mean + actual_std,
                             color=color, alpha=0.10, zorder=1)

            if include_trend:
                trend = _linear_trend(x, actual_mean)
                if trend is not None:
                    slope, intercept = trend
                    trend_y = slope * x + intercept
                    ax.plot(x, trend_y, color=color, linewidth=1.3, linestyle=":", alpha=0.85,
                            label=f"{bot} Trend ({slope:+.4f}/s)", zorder=2.5)

            if include_unfiltered and unfiltered_by_bot and bot in unfiltered_by_bot:
                u_df = unfiltered_by_bot[bot]
                if u_df is not None and not u_df.empty:
                    u_agg = u_df.groupby("TimeBin")[metric].mean().reset_index().sort_values("TimeBin")
                    ax.plot(u_agg["TimeBin"], u_agg[metric], color=color, linewidth=1.3,
                            linestyle="-.", alpha=0.6, label=f"{bot} Unfiltered", zorder=1.5)

            mae = _mae(bot_df[actual_col], bot_df[target_col])
            n_segments = int(bot_df[actual_col].notna().sum())
            mae_lines.append(f"{bot}: MAE={mae:.4f} (n={n_segments})")

        ax.text(
            0.02, 0.02, "\n".join(mae_lines), transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left",
            family="monospace", bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999", alpha=0.85),
            zorder=5,
        )

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Segment (s)", fontsize=9)
        ax.set_ylabel("Scaled (0-1)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linestyle="--")
        # No per-axis legend - every panel plots the same set of labels (Target +
        # each bot's Filtered/Trend/Unfiltered), so plot_filtered_target_tracking_by_bot
        # draws one shared legend below the whole figure instead of repeating it 3x.


def plot_filtered_target_tracking_by_bot(df_factors, df_target_tracking, pacing_target, bots=None, width=16,
                                          height=5, target_curves=None, include_target=True,
                                          include_unfiltered=True, include_trend=True):
    """
    Same data as plot_filtered_target_tracking_merged, but WITHOUT pooling applied
    bots together: one Filtered-Actual line (+ optional trend, + optional Unfiltered
    reference) per bot, each in its own color, overlaid on a single shared Target
    line per panel (Threat, Tempo, OverallPacing) - "does each applied bot track the
    curve differently" rather than one averaged answer.

    Args:
        df_factors: Tidy DataFrame from compile.generator.load_bot_pacing_factors
            (for the per-bot Unfiltered reference lines), or None to omit them
        df_target_tracking: Raw (unaggregated) DataFrame from compile.generator.
            load_filtered_target_tracking
        pacing_target: Which PacingTarget curve to plot
        bots: Optional list to restrict which bots are drawn (default: every bot
            present in df_target_tracking for this pacing_target)
        target_curves: Optional dict from load_pacing_target_curves - when given and
            it has an entry for pacing_target, draws the Target line from the
            deterministic curve spread across [0, Timer] instead of from logged rows
            (see _draw_target_tracking_panels_by_bot)
        include_target: Draw the shared Target line (default True)
        include_unfiltered: Draw each bot's Unfiltered reference line, requires
            df_factors (default True)
        include_trend: Draw each bot's dotted Filtered-Actual trend line (default True)

    Returns:
        Matplotlib Figure, or None if no data matched.
    """
    track_df = _to_pandas(df_target_tracking)
    track_df = track_df[track_df["PacingTarget"] == pacing_target]
    # Every bot sharing this PacingTarget, before the optional `bots` restriction -
    # used for the Timer lookup/fallback Target aggregation (see
    # _draw_target_tracking_panels_by_bot), consistent with
    # plot_filtered_target_tracking_merged.
    target_pool_df = track_df
    if bots is not None:
        track_df = track_df[track_df["Bot"].isin(bots)]
    if track_df.empty:
        print(f"⚠️ No target-tracking data for PacingTarget={pacing_target}")
        return None

    bots_in_data = sorted(track_df["Bot"].dropna().unique())
    # Offset past index 0 - THEME_COLORS["categorical"][0] is "danger", already used
    # for the Target line, so bot colors shouldn't repeat it.
    palette = get_theme_color("categorical")
    bot_colors = {bot: palette[(i + 1) % len(palette)] for i, bot in enumerate(bots_in_data)}

    unfiltered_by_bot = None
    if include_unfiltered and df_factors is not None:
        factors_df = _to_pandas(df_factors)
        unfiltered_by_bot = {}
        for bot in bots_in_data:
            factors_bot_df = factors_df[factors_df["Bot"] == bot]
            if factors_bot_df.empty:
                continue
            # Fit normalization across both Groups for this bot (not just Unfiltered
            # rows), consistent with plot_filtered_target_tracking's per-bot Unfiltered line.
            factors_bot_df = _add_overall_pacing_columns(factors_bot_df)
            unfiltered_by_bot[bot] = factors_bot_df[factors_bot_df["Group"] == "Unfiltered"]

    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    _draw_target_tracking_panels_by_bot(
        axes, track_df, unfiltered_by_bot, bot_colors, pacing_target=pacing_target,
        target_curves=target_curves, target_pool_df=target_pool_df,
        include_target=include_target, include_unfiltered=include_unfiltered, include_trend=include_trend,
    )

    fig.suptitle(
        f"{pacing_target}: Per-Bot Filtered Actual vs Target (engine ground truth)",
        fontsize=13, fontweight="bold",
    )
    # One shared legend below the figure instead of one per panel (every panel plots
    # the same set of labels, just different data) - avoids 3x repetition and keeps
    # it out of the way of the lines, which get busy with a bot's worth of lines each.
    handles, labels = axes[0].get_legend_handles_labels()
    n_cols = min(len(labels), 4)
    fig.tight_layout(rect=[0, 0.12, 1, 0.9])
    # bbox_to_anchor's y stays inside [0, 1] (figure fraction) - the rect above
    # already reserved the bottom 12% of the figure for axes NOT to use, so placing
    # the legend at y=0.02 keeps it in that reserved strip without going negative,
    # which would clip it in Jupyter's inline display (bbox_inches="tight" only
    # rescues out-of-bounds legends when saving to a file, not when rendered inline).
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=n_cols, fontsize=8, framealpha=0.9)
    return fig


def plot_all_filtered_target_tracking_by_bot(df_factors, df_target_tracking, bots=None, output_dir=None, width=16,
                                              height=5, sim_targets_dir=None, include_target=True,
                                              include_unfiltered=True, include_trend=True):
    """
    Iterate every PacingTarget present in df_target_tracking and produce one
    per-bot (not pooled) Actual-vs-Target tracking figure each (see
    plot_filtered_target_tracking_by_bot).

    Args:
        sim_targets_dir: Optional folder of pacing target *.json files (Resources/
            .../Sim_Targets/<subfolder> for the run being plotted) - when given, the
            Target line is drawn from the deterministic curve instead of being
            bounded by logged rows.
        include_target, include_unfiltered, include_trend: Toggle individual line
            types on/off across every generated figure (see
            plot_filtered_target_tracking_by_bot).

    Returns:
        Dict of {pacing_target: Figure}
    """
    track_df = _to_pandas(df_target_tracking)
    if track_df is None or track_df.empty:
        print("⚠️ No target-tracking data to plot.")
        return {}
    if bots is not None:
        track_df = track_df[track_df["Bot"].isin(bots)]

    target_curves = load_pacing_target_curves(sim_targets_dir) if sim_targets_dir else None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    figs = {}
    for pacing_target in sorted(track_df["PacingTarget"].dropna().unique()):
        fig = plot_filtered_target_tracking_by_bot(
            df_factors, track_df, pacing_target, bots=bots, width=width, height=height,
            target_curves=target_curves, include_target=include_target,
            include_unfiltered=include_unfiltered, include_trend=include_trend,
        )
        if fig is None:
            continue
        figs[pacing_target] = fig
        if output_dir:
            safe_target = re.sub(r"[^\w.\-]+", "_", pacing_target)
            out_path = os.path.join(output_dir, f"by_bot_{safe_target}_target_tracking.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: {out_path}")
    return figs


def compute_target_tracking_error_table(df_target_tracking):
    """
    Collapse df_target_tracking (see compile.generator.load_filtered_target_tracking
    - one row per raw logged segment, already carrying that segment's own matched
    Actual*Scaled/Target*Scaled pair) down to one row per (Bot, PacingTarget): mean
    absolute tracking error (see _mae) for Threat, Tempo, and OverallPacing, plus how
    many raw segment rows it's built from. Every other chart in this module reads a
    time series; this is the "collapse the time axis away" summary that
    plot_target_tracking_error_vs_volatility and
    plot_target_tracking_error_by_archetype_heatmap both build on.

    Returns:
        DataFrame with columns Bot, PacingTarget, ThreatMAE, TempoMAE,
        OverallPacingMAE, n_segments - one row per (Bot, PacingTarget) combination
        present in df_target_tracking.
    """
    df = _to_pandas(df_target_tracking)
    rows = []
    for (bot, pacing_target), group in df.groupby(["Bot", "PacingTarget"]):
        row = {"Bot": bot, "PacingTarget": pacing_target, "n_segments": int(len(group))}
        for metric, (actual_col, target_col) in METRIC_TO_SCALED.items():
            row[f"{metric}MAE"] = _mae(group[actual_col], group[target_col])
        rows.append(row)
    return pd.DataFrame(rows)


def _label_curve_archetypes(table, flat_amplitude_threshold=0.05, monotonic_direction_changes=1):
    """
    Bucket each PacingTarget (row of compute_target_curve_features's table) into one
    of a handful of shape archetypes from its OverallPacing curve stats, instead of
    leaving each curve as its own uniquely-named category - this is what lets
    plot_target_tracking_error_by_archetype_heatmap stay readable no matter how many
    *.json target files exist.

    - Amplitude below flat_amplitude_threshold: "Flat" (barely moves either way,
      regardless of direction-change count).
    - Otherwise, at most monotonic_direction_changes direction flips: "Rising"/
      "Falling" by the sign of net Trend (a steady ramp, not back-and-forth - a
      single up-then-down hump still counts as 1 flip and nets out by which way it
      ended).
    - Otherwise it's oscillating - split into "Oscillating-Slow"/"Oscillating-Fast"
      by whether DirectionChanges is at or above the *median* among the oscillating
      curves in this table, so the fast/slow split adapts to whatever set of curves
      is actually loaded rather than a fixed change-count cutoff.
    """
    amplitude = table["OverallPacingAmplitude"]
    trend = table["OverallPacingTrend"]
    direction_changes = table["OverallPacingDirectionChanges"]

    oscillating_mask = (amplitude >= flat_amplitude_threshold) & (direction_changes > monotonic_direction_changes)
    osc_median = direction_changes[oscillating_mask].median() if oscillating_mask.any() else 0

    labels = []
    for amp, tr, dc in zip(amplitude, trend, direction_changes):
        if amp < flat_amplitude_threshold:
            labels.append("Flat")
        elif dc <= monotonic_direction_changes:
            labels.append("Rising" if tr >= 0 else "Falling")
        else:
            labels.append("Oscillating-Fast" if dc >= osc_median else "Oscillating-Slow")
    return labels


def compute_target_curve_features(target_curves):
    """
    Turn each PacingTarget's raw deterministic curve (see load_pacing_target_curves)
    into a handful of shape numbers - how erratic it is, how big a swing it demands,
    which direction it nets out - instead of leaving "which PacingTarget is this" as
    an identity that has to be individually named/rendered. Lets
    plot_target_tracking_error_vs_volatility and
    plot_target_tracking_error_by_archetype_heatmap scale to any number of target
    configs: points/cells are positioned or bucketed by curve shape, never by config
    name, so nothing needs per-item text.

    Per metric (Threat, Tempo, OverallPacing), computes:
      - Volatility: mean absolute step-to-step change (how fast the curve moves)
      - Amplitude: max - min (how big a swing it covers)
      - Trend: net change from first point to last (steady net direction, not
        instantaneous slope - a curve that ends where it started nets to 0 even if it
        swung the whole way up and back down in between)
      - DirectionChanges: count of sign flips in the step-to-step delta (0 = strictly
        monotonic ramp, higher = more back-and-forth)
    Also assigns one Archetype label per PacingTarget from the OverallPacing curve -
    see _label_curve_archetypes.

    Returns:
        DataFrame indexed by PacingTarget name, with {Metric}Volatility/
        {Metric}Amplitude/{Metric}Trend/{Metric}DirectionChanges columns for each of
        Threat/Tempo/OverallPacing, plus a single Archetype column.
    """
    rows = []
    for pacing_target, curves in target_curves.items():
        row = {"PacingTarget": pacing_target}
        for metric in ["Threat", "Tempo", "OverallPacing"]:
            arr = np.asarray(curves[metric], dtype=float)
            deltas = np.diff(arr)
            row[f"{metric}Volatility"] = float(np.mean(np.abs(deltas))) if len(deltas) else 0.0
            row[f"{metric}Amplitude"] = float(arr.max() - arr.min()) if len(arr) else 0.0
            row[f"{metric}Trend"] = float(arr[-1] - arr[0]) if len(arr) else 0.0
            signs = np.sign(deltas)
            signs = signs[signs != 0]
            row[f"{metric}DirectionChanges"] = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
        rows.append(row)
    table = pd.DataFrame(rows).set_index("PacingTarget")
    if not table.empty:
        table["Archetype"] = _label_curve_archetypes(table)
    return table


def _merge_on_resolved_curve_key(error_table, feature_table, feature_cols=None):
    """
    Join error_table's PacingTarget column against feature_table's PacingTarget
    index (see compute_target_curve_features), resolving each PacingTarget value
    through _resolve_curve_key first instead of a plain equality merge - so a
    compound value like "lin_down_06_04_constraint_avg_bot" still matches the clean
    curve name "lin_down_06_04" (see _resolve_curve_key's docstring for why that
    mismatch happens and why it's resolved here rather than by reprocessing).

    Args:
        feature_cols: Optional subset of feature_table columns to bring in (passed
            straight to feature_table[feature_cols] before the merge); None keeps
            every column.

    Returns:
        error_table with feature_table's columns joined on, restricted to rows whose
        PacingTarget resolved to a known curve (inner join).
    """
    if feature_cols is not None:
        feature_table = feature_table[feature_cols]
    curve_keys = list(feature_table.index)
    resolved = error_table["PacingTarget"].map(lambda pt: _resolve_curve_key(pt, curve_keys))
    return error_table.assign(_CurveKey=resolved).merge(
        feature_table, left_on="_CurveKey", right_index=True, how="inner"
    ).drop(columns="_CurveKey")


def plot_target_tracking_error_vs_volatility(df_target_tracking, sim_targets_dir, width=15, height=5):
    """
    Does tracking error scale with how erratic the target curve is, rather than
    which specific PacingTarget it happens to be? 3 panels (Threat, Tempo,
    OverallPacing): x = that metric's target curve Volatility (see
    compute_target_curve_features), y = that metric's Actual-vs-Target MAE (see
    compute_target_tracking_error_table), one point per (Bot, PacingTarget), colored
    by Bot, with a linear trend line per panel. Deliberately never plots or labels
    PacingTarget identity anywhere - swaps it out for a continuous shape feature
    instead, so this scales to any number of target configs without becoming a wall
    of overlapping per-point text.

    Args:
        sim_targets_dir: Folder of pacing target *.json files (see
            load_pacing_target_curves) - required here (not optional like elsewhere
            in this module) since curve shape features can only come from the
            deterministic curve, not from logged rows.

    Returns:
        Matplotlib Figure.
    """
    error_table = compute_target_tracking_error_table(df_target_tracking)
    feature_table = compute_target_curve_features(load_pacing_target_curves(sim_targets_dir))
    merged = _merge_on_resolved_curve_key(error_table, feature_table)

    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    if merged.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No PacingTarget curves matched the tracking data.",
                    ha="center", va="center", transform=ax.transAxes)
        return fig

    bots = sorted(merged["Bot"].dropna().unique())
    palette = get_theme_color("categorical")
    bot_colors = {bot: palette[i % len(palette)] for i, bot in enumerate(bots)}

    for ax, metric in zip(axes, ["Threat", "Tempo", "OverallPacing"]):
        x_col, y_col = f"{metric}Volatility", f"{metric}MAE"
        for bot in bots:
            sub = merged[merged["Bot"] == bot]
            ax.scatter(sub[x_col], sub[y_col], color=bot_colors[bot], alpha=0.75,
                       edgecolor="black", linewidth=0.5, s=70, label=bot, zorder=3)

        trend = _linear_trend(merged[x_col], merged[y_col])
        if trend is not None:
            slope, intercept = trend
            xs = np.linspace(merged[x_col].min(), merged[x_col].max(), 50)
            ax.plot(xs, slope * xs + intercept, color="gray", linestyle=":", linewidth=1.5,
                    alpha=0.8, zorder=2, label=f"Trend ({slope:+.3f})")

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{metric} Target Volatility (avg. change per segment)", fontsize=9)
        ax.set_ylabel(f"{metric} MAE (Actual vs Target)", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Tracking Error vs Target Curve Volatility (by Bot)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def plot_target_tracking_error_by_archetype_heatmap(df_target_tracking, sim_targets_dir, width=13, height=4.5):
    """
    Bot x curve-Archetype heatmap (see compute_target_curve_features/
    _label_curve_archetypes) of mean tracking error - one panel per metric (Threat,
    Tempo, OverallPacing). Replaces a literal Bot x PacingTarget grid, which stops
    being readable once there are more than a handful of *.json target configs:
    Archetype buckets every config into Flat/Rising/Falling/Oscillating-Slow/
    Oscillating-Fast, so the column count stays small and meaningful regardless of
    how many raw curve files exist.

    Args:
        sim_targets_dir: Folder of pacing target *.json files (see
            load_pacing_target_curves) - required, same reasoning as
            plot_target_tracking_error_vs_volatility.

    Returns:
        Matplotlib Figure.
    """
    error_table = compute_target_tracking_error_table(df_target_tracking)
    feature_table = compute_target_curve_features(load_pacing_target_curves(sim_targets_dir))
    merged = _merge_on_resolved_curve_key(error_table, feature_table, feature_cols=["Archetype"])

    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    if merged.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No PacingTarget curves matched the tracking data.",
                    ha="center", va="center", transform=ax.transAxes)
        return fig

    cmap = get_theme_color("heatmap_cmap")
    for ax, metric in zip(axes, ["Threat", "Tempo", "OverallPacing"]):
        pivot = merged.pivot_table(index="Bot", columns="Archetype", values=f"{metric}MAE", aggfunc="mean")
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap=cmap, linewidths=0.5,
            cbar_kws={"label": "Mean MAE (lower = better)"}, ax=ax,
        )
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Target Curve Archetype", fontsize=9)
        ax.set_ylabel("Bot", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Tracking Error by Target Curve Archetype (Bot x Archetype)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def compute_game_level_tracking_error(df_target_tracking):
    """
    One row per real game - (Bot, PacingTarget, ConfigFolder, GameIndex) together,
    since GameIndex alone only numbers games *within* one ConfigFolder (see
    compile.generator.load_filtered_target_tracking) - with that single game's
    Threat/Tempo/OverallPacing MAE (Actual vs Target, see _mae) and whether the
    applied bot Won it. This is the finer-grained counterpart to
    compute_target_tracking_error_table (which collapses every game into one Bot x
    PacingTarget average): plot_tracking_error_vs_winrate needs one independent
    data point per game, not per (Bot, PacingTarget) group, to say anything
    meaningful about whether tracking quality predicts winning.

    Returns:
        DataFrame with columns Bot, PacingTarget, ConfigFolder, GameIndex, Won,
        ThreatMAE, TempoMAE, OverallPacingMAE, n_segments.
    """
    df = _to_pandas(df_target_tracking)
    rows = []
    group_cols = ["Bot", "PacingTarget", "ConfigFolder", "GameIndex"]
    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["Won"] = bool(group["Won"].iloc[0])
        row["n_segments"] = int(len(group))
        for metric, (actual_col, target_col) in METRIC_TO_SCALED.items():
            row[f"{metric}MAE"] = _mae(group[actual_col], group[target_col])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_tracking_error_vs_winrate(df_target_tracking, n_bins=5, width=15, height=5):
    """
    Does better Actual-vs-Target tracking (lower MAE) actually predict winning? 3
    panels (Threat, Tempo, OverallPacing): every applied bot's individual game gets
    one MAE number (compute_game_level_tracking_error, pooled across bots for sample
    size), binned into up to n_bins MAE quantiles (low error -> high error, left to
    right), and each bin's win rate is drawn as a bar with a binomial standard-error
    whisker and its game count labeled above. Each panel's title also reports the
    point-biserial correlation r between the continuous per-game MAE and the binary
    Won outcome (equivalent to Pearson r) - the bars show the shape of the
    relationship, r summarizes its direction/strength in one number. A dotted
    reference line at 0.5 marks a coin-flip win rate.

    Args:
        n_bins: Requested quantile bin count; silently reduced (via pandas.qcut's
            duplicates="drop") when there aren't enough distinct MAE values to fill
            that many bins.

    Returns:
        Matplotlib Figure.
    """
    table = compute_game_level_tracking_error(df_target_tracking)
    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    if table.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No per-game tracking data to plot.", ha="center", va="center", transform=ax.transAxes)
        return fig

    bar_color = get_theme_color("primary")
    for ax, metric in zip(axes, ["Threat", "Tempo", "OverallPacing"]):
        col = f"{metric}MAE"
        sub = table.dropna(subset=[col])
        if len(sub) < 2 or sub[col].std() == 0:
            ax.text(0.5, 0.5, "Not enough varying data to bin/correlate.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(metric, fontsize=12, fontweight="bold")
            continue

        r = float(np.corrcoef(sub[col], sub["Won"].astype(float))[0, 1])

        try:
            mae_bin = pd.qcut(sub[col], min(n_bins, sub[col].nunique()), duplicates="drop")
        except ValueError:
            mae_bin = pd.cut(sub[col], 1)
        sub = sub.assign(MAEBin=mae_bin)

        binned = sub.groupby("MAEBin", observed=True)["Won"].agg(["mean", "count"]).reset_index()
        binned["se"] = np.sqrt(binned["mean"] * (1 - binned["mean"]) / binned["count"])
        labels = [f"{iv.left:.3f}-{iv.right:.3f}" for iv in binned["MAEBin"]]

        x = np.arange(len(binned))
        ax.bar(x, binned["mean"], yerr=binned["se"], color=bar_color, alpha=0.8,
               edgecolor="black", linewidth=0.5, capsize=4, zorder=3)
        for xi, mean, count in zip(x, binned["mean"], binned["count"]):
            ax.text(xi, min(mean + binned["se"].max() + 0.03, 1.0), f"n={count}",
                    ha="center", fontsize=7, zorder=4)

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{metric} (r = {r:+.3f})", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{metric} MAE bin (low → high error)", fontsize=9)
        ax.set_ylabel("Win Rate", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Tracking Error vs Win Rate (per applied-bot game)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig
