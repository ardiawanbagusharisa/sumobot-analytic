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

    fig.suptitle(f"{bot} — {pacing_target}: Filtered Actual vs Target",
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
        f"Applied ({'/'.join(bots_in_data)}) — {pacing_target}: Filtered Actual vs Target",
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
        f"{pacing_target}: Per-Bot Filtered Actual vs Target",
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


def _label_curve_archetypes(table, flat_amplitude_threshold=0.05):
    """
    Bucket each PacingTarget (row of compute_target_curve_features's table) into one
    of a handful of shape archetypes from its OverallPacing curve stats, instead of
    leaving each curve as its own uniquely-named category - this is what lets
    plot_target_tracking_error_by_archetype_heatmap stay readable no matter how many
    *.json target files exist.

    - Amplitude below flat_amplitude_threshold: "Flat" (barely moves either way,
      regardless of how much it wobbles).
    - Otherwise: "Rising"/"Falling" by the sign of net Trend (first point vs last
      point), regardless of DirectionChanges - a curve that oscillates its way from
      low to high still nets out as "Rising", same as a curve that ramps there in a
      straight line, so there's no separate oscillating bucket.
    """
    amplitude = table["OverallPacingAmplitude"]
    trend = table["OverallPacingTrend"]

    labels = []
    for amp, tr in zip(amplitude, trend):
        if amp < flat_amplitude_threshold:
            labels.append("Flat")
        else:
            labels.append("Rising" if tr >= 0 else "Falling")
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


ARCHETYPE_ORDER = ["Flat", "Rising", "Falling"]


def plot_target_curve_archetypes(sim_targets_dir, flat_amplitude_threshold=0.05, width=14, height=5.5):
    """
    Diagnostic for _label_curve_archetypes: shows the real OverallPacing curves
    loaded from sim_targets_dir (see load_pacing_target_curves), so "why did
    curve X get labeled Y" has a picture to point at instead of just the
    Amplitude/Trend numbers.

    Left panel: every curve plotted over its LocalSegmentIndex, colored by the
    Archetype it actually received.
    Right panel: the decision surface those labels come from - each curve placed
    at (Amplitude, Trend), with the flat_amplitude_threshold and the Trend=0
    Rising/Falling split drawn as reference lines.

    Args:
        sim_targets_dir: Folder of pacing target *.json files (see
            load_pacing_target_curves).
        flat_amplitude_threshold: Same knob as _label_curve_archetypes - passed
            through so the threshold drawn matches the labels shown.

    Returns:
        Matplotlib Figure.
    """
    target_curves = load_pacing_target_curves(sim_targets_dir)
    table = compute_target_curve_features(target_curves)
    fig, (ax_curves, ax_scatter) = plt.subplots(1, 2, figsize=(width, height))
    if table.empty:
        for ax in (ax_curves, ax_scatter):
            ax.text(0.5, 0.5, "No target curves found.", ha="center", va="center", transform=ax.transAxes)
        return fig

    table["Archetype"] = _label_curve_archetypes(table, flat_amplitude_threshold=flat_amplitude_threshold)
    palette = get_theme_color("categorical")
    colors = {a: palette[i % len(palette)] for i, a in enumerate(ARCHETYPE_ORDER)}

    for pacing_target, curves in target_curves.items():
        arch = table.loc[pacing_target, "Archetype"]
        ax_curves.plot(curves["OverallPacing"], color=colors[arch], alpha=0.85, linewidth=1.5)
    ax_curves.axhspan(0, flat_amplitude_threshold, color="gray", alpha=0.08, zorder=0)
    handles = [plt.Line2D([0], [0], color=colors[a], lw=2, label=f"{a} (n={(table['Archetype'] == a).sum()})")
               for a in ARCHETYPE_ORDER if (table["Archetype"] == a).any()]
    ax_curves.legend(handles=handles, loc="best", fontsize=8)
    ax_curves.set_xlabel("LocalSegmentIndex")
    ax_curves.set_ylabel("OverallPacing (target)")
    ax_curves.set_title(f"Real target curves by Archetype (n={len(table)})")

    for arch in ARCHETYPE_ORDER:
        sub = table[table["Archetype"] == arch]
        if sub.empty:
            continue
        ax_scatter.scatter(sub["OverallPacingAmplitude"], sub["OverallPacingTrend"],
                            color=colors[arch], label=arch, s=70, edgecolor="black", linewidth=0.5, zorder=3)
    for pacing_target, row in table.iterrows():
        ax_scatter.annotate(pacing_target, (row["OverallPacingAmplitude"], row["OverallPacingTrend"]),
                             fontsize=7, alpha=0.75, xytext=(4, 4), textcoords="offset points")
    ax_scatter.axvline(flat_amplitude_threshold, color="gray", linestyle="--", linewidth=1,
                        label=f"flat_amplitude_threshold={flat_amplitude_threshold}")
    ax_scatter.axhline(0, color="black", linestyle=":", linewidth=1, label="Rising/Falling split (Trend=0)")
    ax_scatter.set_xlabel("OverallPacingAmplitude")
    ax_scatter.set_ylabel("OverallPacingTrend")
    ax_scatter.set_title("Decision surface: amplitude x trend")
    ax_scatter.legend(fontsize=7, loc="best")

    fig.tight_layout()
    return fig


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


def plot_target_tracking_error_vs_volatility(df_target_tracking, sim_targets_dir, bin_width=10, y_bin_width=10,
                                              max_x_axis=None, width=15, height=5):
    """
    Does tracking error scale with how big a swing the target curve demands,
    rather than which specific PacingTarget it happens to be? 3 panels (Threat,
    Tempo, OverallPacing): x = that metric's target curve Amplitude (max - min
    over the whole curve, see compute_target_curve_features), y = that metric's
    Actual-vs-Target MAE (see compute_target_tracking_error_table), one point per
    (Bot, PacingTarget), colored by Bot, with a linear trend line per panel.
    Deliberately never plots or labels PacingTarget identity anywhere - swaps it
    out for a continuous shape feature instead, so this scales to any number of
    target configs without becoming a wall of overlapping per-point text.

    Amplitude (not Volatility, mean step-to-step change) is used here because a
    curve with one big single-step leap - a step function that jumps once and
    stays flat the rest of the way - gets that leap averaged away by Volatility
    (diluted across every other near-zero step), while Amplitude (max - min)
    still reflects it directly; Volatility would put every curve near x=0
    regardless of whether some of them actually contain a large leap.

    Both axes tick at fixed, evenly-spaced steps instead of auto-scaling to their
    own data, so a given x/y position means the same thing on every panel and any
    two runs of this chart line up. y always uses fixed-width 0.1 steps (matching
    plot_tracking_error_vs_winrate's MAE bins): y_bin_width gives the *count* of
    those steps, so the y-axis spans [0, y_bin_width * 0.1]. x does the same by
    default (bin_width 0.1-wide steps spanning [0, bin_width * 0.1]) unless
    max_x_axis is given, in which case the x-axis spans [0, max_x_axis] instead,
    split into bin_width *equal* steps of max_x_axis / bin_width each (not fixed
    at 0.1) - e.g. max_x_axis=0.2, bin_width=5 ticks at 0, 0.04, 0.08, 0.12, 0.16,
    0.2, for zooming into a narrow Amplitude range at a finer resolution than
    0.1-wide steps would allow.

    Args:
        sim_targets_dir: Folder of pacing target *.json files (see
            load_pacing_target_curves) - required here (not optional like elsewhere
            in this module) since curve shape features can only come from the
            deterministic curve, not from logged rows.
        bin_width: Number of x-axis (Amplitude) steps. Without max_x_axis, each
            step is fixed at 0.1 wide, so the x-axis spans [0, bin_width * 0.1]
            (default 10, i.e. [0, 1]). With max_x_axis, this instead is the
            number of equal divisions of [0, max_x_axis].
        y_bin_width: Number of 0.1-wide steps on the y-axis (MAE), i.e. the
            y-axis spans [0, y_bin_width * 0.1] (default 10, i.e. [0, 1]).
        max_x_axis: Optional explicit x-axis (Amplitude) upper bound. When
            given, overrides bin_width's default fixed-0.1-step behavior - the
            x-axis spans [0, max_x_axis] split into bin_width equal steps
            instead.

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

    metrics = ["Threat", "Tempo", "OverallPacing"]
    if max_x_axis is not None:
        x_ticks = np.linspace(0, max_x_axis, int(bin_width) + 1)
    else:
        x_ticks = np.arange(int(bin_width) + 1) * 0.1
    y_ticks = np.arange(int(y_bin_width) + 1) * 0.1
    x_max, y_max = x_ticks[-1], y_ticks[-1]

    for ax, metric in zip(axes, metrics):
        x_col, y_col = f"{metric}Amplitude", f"{metric}MAE"
        for bot in bots:
            sub = merged[merged["Bot"] == bot]
            ax.scatter(sub[x_col], sub[y_col], color=bot_colors[bot], alpha=0.75,
                       edgecolor="black", linewidth=0.5, s=70, label=bot, zorder=3)

        trend = _linear_trend(merged[x_col], merged[y_col])
        if trend is not None:
            slope, intercept = trend
            xs = np.linspace(0, x_max, 50)
            ax.plot(xs, slope * xs + intercept, color="gray", linestyle=":", linewidth=1.5,
                    alpha=0.8, zorder=2, label=f"Trend ({slope:+.3f})")

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{metric} Target Amplitude (max - min)", fontsize=9)
        ax.set_ylabel(f"{metric} MAE (Actual vs Target)", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Tracking Error vs Target Curve Amplitude (by Bot)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def plot_target_tracking_error_by_archetype_heatmap(df_target_tracking, sim_targets_dir, bin_width=10, width=13,
                                                      height=4.5):
    """
    Bot x curve-Archetype heatmap (see compute_target_curve_features/
    _label_curve_archetypes) of mean tracking error - one panel per metric (Threat,
    Tempo, OverallPacing). Replaces a literal Bot x PacingTarget grid, which stops
    being readable once there are more than a handful of *.json target configs:
    Archetype buckets every config into Flat/Rising/Falling, so the column count
    stays small and meaningful regardless of how many raw curve files exist.

    The color scale (MAE, encoded here as cell color/annotation rather than a
    plotted axis) uses fixed-width 0.1 steps (matching plot_tracking_error_vs_
    winrate's MAE bins) instead of auto-scaling its color range to each panel's
    own min/max: bin_width gives the *count* of those 0.1-wide steps, so the
    scale spans [0, bin_width * 0.1] - e.g. bin_width=10 covers the full [0, 1]
    normalized range, bin_width=5 zooms into just [0, 0.5] (values above that
    clip to the top color, annotation still shows the real value). So a given
    color means the same MAE value on every panel and any two runs of this
    chart line up (same standardization as plot_target_tracking_error_vs_
    volatility's axes).

    Args:
        sim_targets_dir: Folder of pacing target *.json files (see
            load_pacing_target_curves) - required, same reasoning as
            plot_target_tracking_error_vs_volatility.
        bin_width: Number of 0.1-wide steps on the MAE color scale, i.e. the
            scale spans [0, bin_width * 0.1] (default 10, i.e. [0, 1]).

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
    cbar_ticks = np.arange(int(bin_width) + 1) * 0.1
    cbar_max = cbar_ticks[-1]
    for ax, metric in zip(axes, ["Threat", "Tempo", "OverallPacing"]):
        pivot = merged.pivot_table(index="Bot", columns="Archetype", values=f"{metric}MAE", aggfunc="mean")
        # Always show every archetype column, even ones absent from this
        # metric/run's data (left blank) - keeps the grid the same shape/order
        # across panels and runs instead of shrinking to whatever showed up.
        pivot = pivot.reindex(columns=ARCHETYPE_ORDER)
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap=cmap, linewidths=0.5, vmin=0, vmax=cbar_max,
            cbar_kws={"label": "Mean MAE (lower = better)", "ticks": cbar_ticks}, ax=ax,
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
    meaningful about whether tracking quality predicts winning. PacingConstraint
    is carried along (constant within one ConfigFolder, so free to include in the
    grouping) for callers that need to split by it, e.g.
    plot_winrate_by_archetype_heatmap.

    Returns:
        DataFrame with columns Bot, PacingTarget, PacingConstraint, ConfigFolder,
        GameIndex, Won, ThreatMAE, TempoMAE, OverallPacingMAE, n_segments.
    """
    df = _to_pandas(df_target_tracking)
    rows = []
    group_cols = ["Bot", "PacingTarget", "PacingConstraint", "ConfigFolder", "GameIndex"]
    # dropna=False - a null PacingConstraint (older data predating that field, or a
    # config folder name with no constraint suffix) must still keep its game, not
    # silently disappear because one of the group_cols is null (pandas' default).
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["Won"] = bool(group["Won"].iloc[0])
        row["n_segments"] = int(len(group))
        for metric, (actual_col, target_col) in METRIC_TO_SCALED.items():
            row[f"{metric}MAE"] = _mae(group[actual_col], group[target_col])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_tracking_error_vs_winrate(df_target_tracking, bin_width=10, width=15, height=5):
    """
    Does better Actual-vs-Target tracking (lower MAE) actually predict winning? 3
    panels (Threat, Tempo, OverallPacing): every applied bot's individual game gets
    one MAE number (compute_game_level_tracking_error, pooled across bots for sample
    size), binned into fixed-width 0.1 MAE bins (low error -> high error, left to
    right), and each bin's win rate is drawn as a bar with a binomial
    standard-error whisker and its game count labeled above. Fixed-width bins
    (rather than quantile bins) keep the x-axis directly comparable across
    panels/metrics/runs, since a given bin - e.g. "0.100-0.200" - always means
    the same absolute error range regardless of how the underlying MAE values
    happen to be distributed. Each panel's title also reports the point-biserial
    correlation r between the continuous per-game MAE and the binary Won outcome
    (equivalent to Pearson r) - the bars show the shape of the relationship, r
    summarizes its direction/strength in one number. A dotted reference line at
    0.5 marks a coin-flip win rate.

    Args:
        bin_width: Number of 0.1-wide MAE bins, i.e. the x-axis spans
            [0, bin_width * 0.1] (default 10, i.e. the full normalized [0, 1]
            MAE range in 10 bins of 0.1 each; bin_width=5 would only cover
            [0, 0.5]). All bins in that range are always drawn, even ones with
            zero games (bar height 0, "n=0"), so the x-axis is identical across
            all 3 panels and any two runs of this chart.

    Returns:
        Matplotlib Figure.
    """
    bin_edges = np.arange(int(bin_width) + 1) * 0.1
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

        # observed=False keeps every bin_edges bin in the result (mean=NaN,
        # count=0 for ones with no games in this sub/metric) so all 3 panels
        # share the same fixed 0.0-0.1 ... 0.9-1.0 bin set/x-axis, instead of
        # each only showing whichever bins happened to contain data.
        mae_bin = pd.cut(sub[col], bin_edges, include_lowest=True)
        sub = sub.assign(MAEBin=mae_bin)

        binned = sub.groupby("MAEBin", observed=False)["Won"].agg(["mean", "count"]).reset_index()
        binned["se"] = np.sqrt(binned["mean"] * (1 - binned["mean"]) / binned["count"])
        labels = [f"{iv.left:.3f}-{iv.right:.3f}" for iv in binned["MAEBin"]]

        x = np.arange(len(binned))
        ax.bar(x, binned["mean"].fillna(0), yerr=binned["se"].fillna(0), color=bar_color, alpha=0.8,
               edgecolor="black", linewidth=0.5, capsize=4, zorder=3)
        for xi, mean, se, count in zip(x, binned["mean"], binned["se"], binned["count"]):
            label_y = 0.03 if pd.isna(mean) else min(mean + (0 if pd.isna(se) else se) + 0.03, 1.0)
            ax.text(xi, label_y, f"n={count}", ha="center", fontsize=7, zorder=4)

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


def compute_winrate_by_archetype_table(df_target_tracking, target_curves):
    """
    One row per real game (see compute_game_level_tracking_error) with its
    PacingTarget's curve Archetype attached (Flat/Rising/Falling, see
    compute_target_curve_features/_label_curve_archetypes) instead of the raw
    PacingTarget name - lets win rate be grouped by curve *shape* rather than
    needing one column/row per PacingTarget (61+ configs in prod, the same
    scaling problem plot_target_tracking_error_by_archetype_heatmap solves for
    tracking error).

    Args:
        target_curves: Dict from load_pacing_target_curves.

    Returns:
        compute_game_level_tracking_error's table with an Archetype column
        joined on (via _merge_on_resolved_curve_key, so it tolerates the same
        glued "_constraint_" PacingTarget naming that function already handles),
        restricted to rows whose PacingTarget resolved to a known curve.
    """
    game_table = compute_game_level_tracking_error(df_target_tracking)
    feature_table = compute_target_curve_features(target_curves)
    return _merge_on_resolved_curve_key(game_table, feature_table, feature_cols=["Archetype"])


def plot_winrate_by_archetype_heatmap(df_target_tracking, sim_targets_dir, width=13, height=4.5):
    """
    Bot x curve-Archetype heatmap of win rate (see compute_winrate_by_archetype_
    table) - one panel per PacingConstraint (prod carries avg_bot/top_5/nn, so 3
    panels there). Mirrors plot_target_tracking_error_by_archetype_heatmap's
    Bot x Archetype grid, applied to win rate instead of MAE, and additionally
    split by PacingConstraint since a curve's shape means something different
    depending on which constraint normalized it - pooling constraints together
    would blend that apart, the same reasoning the rest of this notebook already
    splits every PacingConstraint-aware chart by.

    Args:
        sim_targets_dir: Folder of pacing target *.json files (see
            load_pacing_target_curves) - required, same reasoning as
            plot_target_tracking_error_vs_volatility.

    Returns:
        Matplotlib Figure.
    """
    target_curves = load_pacing_target_curves(sim_targets_dir)
    merged = compute_winrate_by_archetype_table(df_target_tracking, target_curves)

    # None (single "All" panel) only if PacingConstraint is entirely missing -
    # otherwise one panel per distinct constraint actually present in the data.
    constraints = sorted(c for c in merged["PacingConstraint"].dropna().unique()) if not merged.empty else []
    if not constraints:
        constraints = [None]

    fig, axes = plt.subplots(1, len(constraints), figsize=(width, height), squeeze=False)
    axes = axes[0]
    if merged.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No PacingTarget curves matched the tracking data.",
                    ha="center", va="center", transform=ax.transAxes)
        return fig

    cmap = get_theme_color("heatmap_cmap")
    for ax, constraint in zip(axes, constraints):
        sub = merged if constraint is None else merged[merged["PacingConstraint"] == constraint]
        pivot = sub.pivot_table(index="Bot", columns="Archetype", values="Won", aggfunc="mean")
        # Always show every archetype column, even ones absent for this
        # constraint's data (left blank) - keeps the grid the same shape/order
        # across panels and runs instead of shrinking to whatever showed up.
        pivot = pivot.reindex(columns=ARCHETYPE_ORDER)
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap=cmap, linewidths=0.5, vmin=0, vmax=1.0,
            cbar_kws={"label": "Win Rate"}, ax=ax,
        )
        ax.set_title(str(constraint) if constraint is not None else "All", fontsize=12, fontweight="bold")
        ax.set_xlabel("Target Curve Archetype", fontsize=9)
        ax.set_ylabel("Bot", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Win Rate by Target Curve Archetype (Bot x Archetype, per Constraint)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# Patterns checked in order (first match wins) against a PacingTarget name stem.
# Covers both naming conventions seen in the wild:
#   - The small hand-authored test set (this repo's cached CSVs): "default_target_
#     {high,med,low}" (Constant), "linear_{increase,decrease}" (Linear).
#   - The real Sim_Targets/Experiments_real config folder (see
#     Assets/Resources/Pacing/Sim_Targets/Experiments_real): "stat_NN" (Constant),
#     "lin_{up,down}_NN_MM" (Linear), "sigF_/sigH_/sigQ_{up,down}_NN_MM" (three
#     sigmoid steepness variants, all folded into one "Sigmoid" family here rather
#     than 3 separate types - see _classify_target_type), "step_{up,down}_NN_MM"
#     (Step - a single discontinuous jump, distinct from a Linear ramp or a smooth
#     Sigmoid, so it needs its own bucket rather than falling into "Other").
# "sig[a-z]*_" (not a literal "sigmoid" check) so it matches sigF_/sigH_/sigQ_
# without hardcoding each steepness-variant letter.
TARGET_TYPE_PATTERNS = [
    ("Step", re.compile(r"step_(up|down)|\bstep\b", re.IGNORECASE)),
    ("Sigmoid", re.compile(r"sigmoid|sig[a-z]*_(up|down)", re.IGNORECASE)),
    ("Linear", re.compile(r"linear|lin_(up|down)", re.IGNORECASE)),
    ("Constant", re.compile(r"constant|default_target|^stat_\d", re.IGNORECASE)),
]

# Preferred display/legend order for target Type - used wherever a chart needs a
# stable column/bar ordering (see plot_winrate_by_target_type_and_value) instead
# of whatever order table["Type"].unique() happens to yield. "Other" stays last as
# the catch-all bucket for anything TARGET_TYPE_PATTERNS doesn't recognize.
TARGET_TYPE_ORDER = ["Constant", "Linear", "Sigmoid", "Step", "Other"]


def _classify_target_type(pacing_target):
    """
    Bucket a PacingTarget name into a coarse curve-family label (see
    TARGET_TYPE_ORDER) purely from its naming convention - e.g.
    "default_target_high"/"stat_00" -> "Constant", "linear_increase"/"lin_up_00_10"
    -> "Linear", "sigF_up_00_10" -> "Sigmoid", "step_up_00_10" -> "Step" - no curve
    JSON needed. This is what lets compute_final_target_winrate_table group games
    by target *family* first (per the analysis request: split by target type, then
    by where it ends up), independent of the Amplitude/Trend-based Archetype
    labeling in _label_curve_archetypes (which buckets by curve shape math and
    needs sim_targets_dir, not naming).

    Falls back to "Other" for any PacingTarget matching none of the known
    patterns, so an unrecognized/new target name just lands in its own bucket
    instead of breaking the grouping.
    """
    name = "" if pacing_target is None else str(pacing_target)
    for label, pattern in TARGET_TYPE_PATTERNS:
        if pattern.search(name):
            return label
    return "Other"


def compute_final_target_winrate_table(df_target_tracking):
    """
    One row per real game (same (Bot, PacingTarget, PacingConstraint, ConfigFolder,
    GameIndex) grouping as compute_game_level_tracking_error), with that game's
    target Type (see _classify_target_type) and its *last logged* TargetThreatScaled/
    TargetTempoScaled/TargetOverallPacingScaled value (the row with the highest
    TimeBin in that game) attached, plus whether the applied bot Won.

    This is the "what was the target actually set to by the end of this game"
    counterpart to compute_game_level_tracking_error, which instead measures
    tracking *error* against the target. It answers a different question: not
    "does tracking quality predict winning" (already covered by
    plot_tracking_error_vs_winrate, which pools every PacingTarget together), but
    "does win rate depend on the target level itself, once curves are first split
    by family" - e.g. is a high Constant target associated with a different win
    rate than a low one, without averaging that signal away against Linear/Sigmoid
    games in between.

    Uses the last logged Target*Scaled row directly (not the deterministic curve
    JSON from load_pacing_target_curves), so this works from df_target_tracking
    alone with no sim_targets_dir needed - for a Constant target this final value
    equals the target throughout the game; for Linear/Sigmoid it's the endpoint
    the ramp/curve reached in that particular game.

    Returns:
        DataFrame with columns Bot, PacingTarget, PacingConstraint, ConfigFolder,
        GameIndex, Type, Won, FinalThreatTarget, FinalTempoTarget,
        FinalOverallPacingTarget.
    """
    df = _to_pandas(df_target_tracking)
    rows = []
    group_cols = ["Bot", "PacingTarget", "PacingConstraint", "ConfigFolder", "GameIndex"]
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["Type"] = _classify_target_type(row["PacingTarget"])
        row["Won"] = bool(group["Won"].iloc[0])
        last = group.loc[group["TimeBin"].idxmax()]
        row["FinalThreatTarget"] = float(last["TargetThreatScaled"])
        row["FinalTempoTarget"] = float(last["TargetTempoScaled"])
        row["FinalOverallPacingTarget"] = float(last["TargetOverallPacingScaled"])
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_winrate_by_type_and_value_panel(ax, table, types, type_colors, bin_edges, metric, title):
    """
    Draws one metric's win-rate-by-Type-and-value bar group into a single given
    axis. Factored out of plot_winrate_by_target_type_and_value so each metric
    (Threat/Tempo/OverallPacing) gets its own standalone figure instead of 3
    panels sharing one figure - each is a genuinely separate chart the user can
    read/save/scroll independently rather than a squeezed-together row.
    """
    metric_cols = {
        "Threat": "FinalThreatTarget", "Tempo": "FinalTempoTarget", "OverallPacing": "FinalOverallPacingTarget",
    }
    col = metric_cols[metric]
    n_bins = len(bin_edges) - 1
    bar_width = 0.8 / max(len(types), 1)

    sub_all = table.dropna(subset=[col])
    if sub_all.empty:
        ax.text(0.5, 0.5, "No data.", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=12, fontweight="bold")
        return

    # observed=False keeps every bin_edges bin in each Type's grouped result
    # (mean=NaN, count=0 for empty ones) so every Type's bars share the same
    # fixed bin set/x-axis, instead of each Type only showing bins it has data in.
    value_bin = pd.cut(sub_all[col], bin_edges, include_lowest=True)
    sub_all = sub_all.assign(ValueBin=value_bin)
    categories = sub_all["ValueBin"].cat.categories

    x = np.arange(n_bins)
    for i, t in enumerate(types):
        sub = sub_all[sub_all["Type"] == t]
        binned = sub.groupby("ValueBin", observed=False)["Won"].agg(["mean", "count"]).reindex(categories)
        binned["se"] = np.sqrt(binned["mean"] * (1 - binned["mean"]) / binned["count"])
        offset = (i - (len(types) - 1) / 2) * bar_width
        ax.bar(x + offset, binned["mean"].fillna(0), width=bar_width, yerr=binned["se"].fillna(0),
               color=type_colors[t], alpha=0.85, edgecolor="black", linewidth=0.5, capsize=3,
               label=f"{t} (n={int(len(sub))})", zorder=3)
        for xi, mean, count in zip(x, binned["mean"], binned["count"]):
            if count > 0:
                ax.text(xi + offset, min((0 if pd.isna(mean) else mean) + 0.03, 1.0), f"{int(count)}",
                        ha="center", fontsize=6, zorder=4)

    labels = [f"{iv.left:.1f}-{iv.right:.1f}" for iv in categories]
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Final {metric} Target value bin", fontsize=9)
    ax.set_ylabel("Win Rate", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, loc="best")


def plot_winrate_by_target_type_and_value(df_target_tracking, metric, bin_width=10, width=10, height=5.5,
                                           constraint=None):
    """
    Does win rate depend on where the target curve is actually set, once games are
    first split by target Type (Constant/Linear/Sigmoid/Step/Other, see
    _classify_target_type) rather than averaged across every target shape first?
    One standalone figure for a single metric (Threat, Tempo, or OverallPacing):
    x = that metric's final logged target value (see
    compute_final_target_winrate_table), binned into fixed-width 0.1 bins
    (matching plot_tracking_error_vs_winrate's MAE bins), one colored bar group
    per Type within each bin, y = win rate with a binomial standard-error whisker
    and game count labeled above each bar.

    Deliberately one metric per figure rather than all 3 sharing one row of
    panels - each is its own chart to read/save independently. Call this 3x (once
    per metric) to get the full Threat/Tempo/OverallPacing picture.

    This is the direct answer to "compare win rate at different target levels,
    within a target type" that plot_tracking_error_vs_winrate can't give - that
    chart pools every PacingTarget together and bins by tracking *error*, so a
    Constant-high vs Constant-low difference gets diluted against Linear/Sigmoid
    games in the same error bin. Here Type is kept as a first-class grouping
    dimension instead, so e.g. Constant bars at a high value bin can be compared
    directly against Constant bars at a low one. A dotted reference line at 0.5
    marks a coin-flip win rate.

    Args:
        metric: Which of "Threat"/"Tempo"/"OverallPacing" to plot.
        bin_width: Number of 0.1-wide final-target-value bins, i.e. the x-axis
            spans [0, bin_width * 0.1] (default 10, i.e. the full normalized
            [0, 1] range). All bins are always drawn (even ones with zero games
            for a given Type), so the x-axis is identical across any two runs of
            this chart.
        constraint: Restrict to games under this single PacingConstraint (e.g.
            "avg_bot"/"top_5"/"nn" on prod) - pooling every constraint together
            (default None) can make win rate look erratic/small across target
            levels for reasons that have nothing to do with the target level
            itself (a mix of easier/harder constraint games landing in the same
            bin), so callers that care about the per-constraint breakdown should
            call this once per constraint value instead.

    Returns:
        Matplotlib Figure.
    """
    bin_edges = np.arange(int(bin_width) + 1) * 0.1
    table = compute_final_target_winrate_table(df_target_tracking)
    if constraint is not None:
        table = table[table["PacingConstraint"] == constraint]

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    if table.empty:
        ax.text(0.5, 0.5, "No per-game target data to plot.", ha="center", va="center", transform=ax.transAxes)
        return fig

    types = [t for t in TARGET_TYPE_ORDER if t in table["Type"].unique()]
    palette = get_theme_color("categorical")
    type_colors = {t: palette[i % len(palette)] for i, t in enumerate(types)}

    label = "All Constraints (Average)" if constraint is None else constraint
    _draw_winrate_by_type_and_value_panel(ax, table, types, type_colors, bin_edges, metric,
                                           title=f"{label} — {metric}")
    fig.tight_layout()
    return fig


def summarize_winrate_by_target_type(df_target_tracking):
    """
    Collapse compute_final_target_winrate_table's one-row-per-game table down to
    one row per (Bot, Type, PacingTarget, PacingConstraint) - the compact
    counterpart for reading as a table instead of the full per-game rows (which
    run into the hundreds/thousands of rows once every game is listed
    individually). Since PacingTarget already pins down the actual target level
    within a Type (e.g. every "default_target_high" game reaches the same
    Constant level), grouping by PacingTarget directly gives one clean row per
    level instead of needing another value-bin dimension the way the plot does
    for its continuous x-axis.

    PacingConstraint is kept as its own grouping column (not averaged over) for
    the same reason plot_winrate_by_archetype_heatmap/compute_winrate_by_
    archetype_table never pool it away: prod carries multiple constraints
    (avg_bot/top_5/nn) per PacingTarget, each normalizing difficulty differently,
    so blending them into one WinRate would average away exactly the effect this
    table exists to show - e.g. a weak bot's win rate can look erratic/small
    across target levels purely because harder-constraint games are mixed in with
    easier ones at the same nominal target, not because the target level itself
    did anything.

    Returns:
        DataFrame with columns Bot, Type, PacingTarget, PacingConstraint,
        n_games, WinRate, WinRateSE (binomial standard error), FinalThreatTarget,
        FinalTempoTarget, FinalOverallPacingTarget (each the mean final target
        value across that group's games - constant within the group for a
        Constant target, an average endpoint for Linear/Sigmoid), sorted by Type
        then FinalOverallPacingTarget so levels read low-to-high within each Type.
    """
    table = compute_final_target_winrate_table(df_target_tracking)
    if table.empty:
        return pd.DataFrame(columns=[
            "Bot", "Type", "PacingTarget", "PacingConstraint", "n_games", "WinRate", "WinRateSE",
            "FinalThreatTarget", "FinalTempoTarget", "FinalOverallPacingTarget",
        ])

    group_cols = ["Bot", "Type", "PacingTarget", "PacingConstraint"]
    grouped = table.groupby(group_cols, dropna=False).agg(
        n_games=("Won", "size"),
        WinRate=("Won", "mean"),
        FinalThreatTarget=("FinalThreatTarget", "mean"),
        FinalTempoTarget=("FinalTempoTarget", "mean"),
        FinalOverallPacingTarget=("FinalOverallPacingTarget", "mean"),
    ).reset_index()
    grouped["WinRateSE"] = np.sqrt(grouped["WinRate"] * (1 - grouped["WinRate"]) / grouped["n_games"])
    grouped = grouped[group_cols + [
        "n_games", "WinRate", "WinRateSE", "FinalThreatTarget", "FinalTempoTarget", "FinalOverallPacingTarget",
    ]]
    return grouped.sort_values(["Type", "FinalOverallPacingTarget", "PacingConstraint", "Bot"]).reset_index(drop=True)
