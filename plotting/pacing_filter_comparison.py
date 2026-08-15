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
import os
import re
import numpy as np
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


def _mse(actual, target):
    """Mean squared error between two same-length series, ignoring NaN pairs."""
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = ~np.isnan(actual) & ~np.isnan(target)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((actual[mask] - target[mask]) ** 2))


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


def _draw_target_tracking_panels(axes, track_df, unfiltered_df):
    """
    Shared per-metric drawing logic for plot_filtered_target_tracking and
    plot_filtered_target_tracking_merged: draws the Target/Actual/trend/Unfiltered
    lines and the MSE annotation into each of the 3 given axes (Threat, Tempo,
    OverallPacing, in that order). track_df/unfiltered_df are grouped by TimeBin as
    given - callers control pooling (single bot vs every applied bot) by how they
    filter these DataFrames before calling this.
    """
    actual_color = get_theme_color("primary")
    target_color = get_theme_color("danger")
    unfiltered_color = get_theme_color("secondary")

    for ax, metric in zip(axes, ["Threat", "Tempo", "OverallPacing"]):
        actual_col, target_col = METRIC_TO_SCALED[metric]

        agg = (
            track_df.groupby("TimeBin")
            .agg(
                Actual_mean=(actual_col, "mean"),
                Actual_std=(actual_col, "std"),
                Target_mean=(target_col, "mean"),
            )
            .reset_index()
            .sort_values("TimeBin")
        )
        x = agg["TimeBin"].values
        actual_mean = agg["Actual_mean"].values
        actual_std = np.nan_to_num(agg["Actual_std"].values)
        target_mean = agg["Target_mean"].values

        ax.plot(x, target_mean, color=target_color, linewidth=2, linestyle="--", label="Target", zorder=2)
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

        mse = _mse(track_df[actual_col], track_df[target_col])
        n_segments = int(track_df[actual_col].notna().sum())
        ax.text(
            0.02, 0.02, f"MSE(Actual, Target) = {mse:.4f}\n(n={n_segments} segments)",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999", alpha=0.85), zorder=5,
        )

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Segment (s)", fontsize=9)
        ax.set_ylabel("Scaled (0-1)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=7, loc="upper right")


def plot_filtered_target_tracking(df_factors, df_target_tracking, bot, pacing_target, width=15, height=5):
    """
    One figure for a single (bot, PacingTarget) pair: 3 panels (Threat, Tempo,
    OverallPacing), each showing the engine's own ground-truth Actual vs Target
    curves for that PacingTarget - i.e. does the dynamic pacing filter actually
    track the curve it was steering towards - with:
      - Target: the predefined curve for this PacingTarget (near-deterministic
        given Timer + elapsed time, so no meaningful spread across rounds/matches)
      - Filtered Actual: mean +/- std band across every matching round, plus a
        dotted linear trend line (labeled with slope) showing whether tracking
        systematically drifts over the match rather than just being noisy
      - MSE(Actual, Target): annotated per panel - single-number tracking error,
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

    Returns:
        Matplotlib Figure, or None if no data matched.
    """
    track_df = _to_pandas(df_target_tracking)
    track_df = track_df[(track_df["Bot"] == bot) & (track_df["PacingTarget"] == pacing_target)]
    if track_df.empty:
        print(f"⚠️ No target-tracking data for bot={bot}, PacingTarget={pacing_target}")
        return None

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
    _draw_target_tracking_panels(axes, track_df, unfiltered_df)

    fig.suptitle(f"{bot} — {pacing_target}: Filtered Actual vs Target (engine ground truth)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def plot_all_filtered_target_tracking(df_factors, df_target_tracking, output_dir=None, width=15, height=5):
    """
    Iterate every (Bot, PacingTarget) pair present in df_target_tracking and produce
    one Actual-vs-Target tracking figure each (see plot_filtered_target_tracking).

    Returns:
        Dict of {(bot_name, pacing_target): Figure}
    """
    track_df = _to_pandas(df_target_tracking)
    if track_df is None or track_df.empty:
        print("⚠️ No target-tracking data to plot.")
        return {}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    figs = {}
    for bot in sorted(track_df["Bot"].dropna().unique()):
        targets = sorted(track_df.loc[track_df["Bot"] == bot, "PacingTarget"].dropna().unique())
        for pacing_target in targets:
            fig = plot_filtered_target_tracking(df_factors, track_df, bot, pacing_target, width=width, height=height)
            if fig is None:
                continue
            figs[(bot, pacing_target)] = fig
            if output_dir:
                safe_target = re.sub(r"[^\w.\-]+", "_", pacing_target)
                out_path = os.path.join(output_dir, f"{bot}_{safe_target}_target_tracking.png")
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"✅ Saved: {out_path}")
    return figs


def plot_filtered_target_tracking_merged(df_factors, df_target_tracking, pacing_target, bots=None, width=15, height=5):
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

    Returns:
        Matplotlib Figure, or None if no data matched.
    """
    track_df = _to_pandas(df_target_tracking)
    track_df = track_df[track_df["PacingTarget"] == pacing_target]
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
    _draw_target_tracking_panels(axes, track_df, unfiltered_df)

    fig.suptitle(
        f"Applied ({'/'.join(bots_in_data)}) — {pacing_target}: Filtered Actual vs Target (engine ground truth)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def plot_all_filtered_target_tracking_merged(df_factors, df_target_tracking, bots=None, output_dir=None, width=15, height=5):
    """
    Iterate every PacingTarget present in df_target_tracking and produce one
    bot-pooled Actual-vs-Target tracking figure each (see
    plot_filtered_target_tracking_merged).

    Returns:
        Dict of {pacing_target: Figure}
    """
    track_df = _to_pandas(df_target_tracking)
    if track_df is None or track_df.empty:
        print("⚠️ No target-tracking data to plot.")
        return {}
    if bots is not None:
        track_df = track_df[track_df["Bot"].isin(bots)]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    figs = {}
    for pacing_target in sorted(track_df["PacingTarget"].dropna().unique()):
        fig = plot_filtered_target_tracking_merged(df_factors, track_df, pacing_target, bots=bots, width=width, height=height)
        if fig is None:
            continue
        figs[pacing_target] = fig
        if output_dir:
            safe_target = re.sub(r"[^\w.\-]+", "_", pacing_target)
            out_path = os.path.join(output_dir, f"applied_{safe_target}_target_tracking.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: {out_path}")
    return figs
