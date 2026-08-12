"""
Visualize dynamic pacing target tracking.

Compares the actual pacing produced by the action filter (ActualOverallPacingScaled)
against the predefined target curve (TargetOverallPacingScaled) that the filter was
supposed to steer the match towards - one chart per PacingTarget (e.g.
"linear_decrease_0.6_to_0.4_60s", "step_increase_0.0_to_1.0", ...).

Target and Actual Tempo/Threat/OverallPacing are BOTH raw composite scores bounded
by the same [MinPacing, MaxPacing] window (e.g. a run with MaxPacing=0.6 means a
value of 0.6 is the ceiling for both the target curve and what the filter can
actually achieve) - neither is already on a [0, 1] scale. compile.generator.
batch_process_pacing_segments min-max normalizes *both* sides the same way into
[0, 1] and clips (Actual/Target {Tempo,Threat,OverallPacing}Scaled) so they're
directly comparable on the same 0-1 scale. Using the raw (unscaled) Target column
here would be wrong: a target curve sitting at/near MinPacing would look like it's
"capped" near the floor instead of correctly rescaling down toward 0.

Expects the combined DataFrame produced by:
    compile.generator.batch_process_pacing_segments(..., applied_bots=[...])
    compile.generator.generate_pacing_segments_from_batches(...)
which carries (among others) these columns:
    PacingTarget, PacingConstraint, PacingRole, Bot, OpponentBot, RoundIndex,
    LocalSegmentIndex, ActualOverallPacingScaled, TargetOverallPacingScaled,
    PacingMinUsed, PacingMaxUsed

The filter is applied to specific bots (identity), not to a fixed Left/Right slot -
matchups are logged in both directions (BotA_vs_BotB and BotB_vs_BotA), so the
applied bot switches sides across folders. Only the applied bot's row actually had
the pacing filter steering it toward the target - the opponent's Target* fields are
logged by the engine for both sides regardless, but aren't meaningful for the
opponent (no filter steered them).

The "vs all rest" / "vs each rest" chart family only ever plots the APPLIED bot's
own Actual/Target lines (never an opponent's own pacing line) - "rest" refers to
which opponent(s) the applied bot's segments are grouped/averaged over, not a line
drawn for the opponent. "vs all rest" pools every opponent into one average line;
"vs each rest" is the same data broken out into one line per individual opponent
faced, so it's a finer-grained view of the same underlying segments as "vs all
rest", not a different metric.

A "focus match" is a round where the opponent was ALSO an applied bot (e.g. MCTS vs
NN when applied_bots=["MCTS", "NN"] - see compile.generator.
batch_process_pacing_segments) - i.e. two filter-steered bots facing each other
rather than an applied bot facing a normal/static "rest" bot. These are excluded by
default from the "vs rest" charts (see include_focus_match on the functions below)
since averaging them in would blend "how does the filter perform against the
roster" with "how do two filters perform against each other", a different
question. plot_pacing_applied_vs_each_rest with include_focus_match=True already
surfaces a focus-match opponent as its own per-opponent line like any other
opponent, so there's no separate head-to-head chart for it.

compute_pacing_achievement_matrix / plot_pacing_achievement_heatmap answer a
different question - across every applied bot x PacingTarget combination, how well
did that bot's filter track its own predefined curve overall - as a single
scannable matrix instead of paging through per-target/per-bot line charts one at a
time.
"""
import math
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .analyzer_config import get_theme_color


def _to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _stats(values):
    """Mean/std/standard-error summary for a 1D series of pacing values."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = int(values.size)
    mean = float(np.mean(values)) if n else float("nan")
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n else float("nan")
    return {"mean": mean, "std": std, "sem": sem, "n": n}


def _add_stats_box(ax, stats_by_label):
    """Annotate an axis with a mean/std/err textbox, one line per series."""
    stats_by_label = {k: v for k, v in stats_by_label.items() if v is not None and v["n"] > 0}
    if not stats_by_label:
        return
    lines = [
        f"{label}: μ={s['mean']:.3f}  σ={s['std']:.3f}  err={s['sem']:.3f}  (n={s['n']})"
        for label, s in stats_by_label.items()
    ]
    ax.text(
        0.02, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
        va="bottom", ha="left", family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999", alpha=0.85),
        zorder=5,
    )


def _applied_baseline(full_df, bots):
    """
    Reference baseline for one or more applied bots: their typical
    ActualOverallPacingScaled *regardless of which PacingTarget curve was active*,
    so a chart for a single target can be compared against "business as usual".

    Returns the mean plus the empirical min/max actually observed (a shaded band).
    Note this deliberately does NOT use the raw PacingMinUsed/PacingMaxUsed config
    bounds - those are the bounds ActualOverallPacingScaled was itself normalized
    against, so on this already-scaled 0-1 axis they would trivially just be 0/1.
    """
    base = full_df[(full_df["PacingRole"] == "Applied") & (full_df["Bot"].isin(bots))]
    values = base["ActualOverallPacingScaled"].dropna().values
    if values.size == 0:
        return None
    return {"mean": float(np.mean(values)), "low": float(np.min(values)), "high": float(np.max(values))}


def _draw_baseline(ax, baseline, color, label):
    if baseline is None:
        return
    ax.axhspan(baseline["low"], baseline["high"], color=color, alpha=0.08, zorder=0)
    ax.axhline(
        baseline["mean"], color=color, linewidth=1.2, linestyle=":", alpha=0.85,
        label=f"{label} (μ={baseline['mean']:.2f})", zorder=1.5,
    )


def _add_trend_line(ax, x, y, color, label_prefix):
    """
    Overlay a least-squares linear trend line for one series (applied bot's actual
    pacing only - see module docstring on why opponent/target lines are excluded).
    Drawn as a thin dotted line distinct from the solid actual-data line, labeled
    with its slope so trend direction/strength is readable straight off the legend.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    trend_y = slope * x + intercept
    ax.plot(
        x, trend_y, color=color, linewidth=1.5, linestyle=":", alpha=0.75,
        label=f"{label_prefix} Trend ({slope:+.3f}/seg)", zorder=2.5,
    )


def plot_pacing_target_tracking(df, target, applied_bot=None, width=14, height=5, facet_by_round=False):
    """
    Plot the applied bot's actual pacing (ActualOverallPacingScaled) vs the
    predefined target curve (TargetOverallPacingScaled) over segment index, for a
    single dynamic pacing target, to visually check whether the action filter tracks
    the intended pacing curve. Both sides are min-max rescaled into [0, 1] using the
    same [MinPacing, MaxPacing] window (see module docstring), so they're on the
    same comparable scale.

    Args:
        df: Combined pacing-segments DataFrame/LazyFrame (pandas or polars). Must have
            a "PacingRole" column (i.e. batch_process_pacing_segments was run with
            applied_bots=[...]) so the applied bot's rows can be identified regardless
            of which Left/Right slot it occupied in a given matchup.
        target: Value of PacingTarget to plot (see df['PacingTarget'].unique())
        applied_bot: If multiple bots share the "Applied" role (e.g. MCTS and NN),
            optionally restrict the line to a single bot's name so they aren't pooled
            together.
        width, height: Figure size (height is per-row when facet_by_round=False)
        facet_by_round: If True, one subplot row per RoundIndex; otherwise all
            rounds are pooled into a single subplot

    Returns:
        Matplotlib Figure, or None if no data matched
    """
    df = _to_pandas(df)

    if "PacingRole" not in df.columns:
        print(
            "⚠️ 'PacingRole' column not found. Run batch_process_pacing_segments(..., "
            "applied_bots=[...]) so applied-bot rows can be identified, then rebuild "
            "the summary with generate_pacing_segments_from_batches()."
        )
        return None

    target_df = df[(df["PacingTarget"] == target) & (df["PacingRole"] == "Applied")].copy()

    if applied_bot is not None and "Bot" in target_df.columns:
        target_df = target_df[target_df["Bot"] == applied_bot]

    if target_df.empty:
        print(f"⚠️ No 'Applied' data found for PacingTarget='{target}', applied_bot={applied_bot}")
        return None

    rounds = sorted(target_df["RoundIndex"].dropna().unique()) if facet_by_round else [None]
    n_rows = max(len(rounds), 1)

    fig, axes = plt.subplots(n_rows, 1, figsize=(width, height * n_rows), squeeze=False)
    axes = axes.flatten()

    color = get_theme_color("primary")
    bots_in_data = sorted(b for b in target_df["Bot"].dropna().unique()) if "Bot" in target_df.columns else []
    label = f"Applied ({'/'.join(bots_in_data)})" if bots_in_data else "Applied"
    ylabel = "Pacing (0-1)"

    baseline = _applied_baseline(df, bots_in_data)

    for row_idx, round_val in enumerate(rounds):
        ax = axes[row_idx]
        round_df = target_df if round_val is None else target_df[target_df["RoundIndex"] == round_val]

        agg = (
            round_df.groupby("LocalSegmentIndex")
            .agg(
                OverallPacing_mean=("ActualOverallPacingScaled", "mean"),
                OverallPacing_std=("ActualOverallPacingScaled", "std"),
                TargetOverallPacing_mean=("TargetOverallPacingScaled", "mean"),
                n=("ActualOverallPacingScaled", "count"),
            )
            .reset_index()
            .sort_values("LocalSegmentIndex")
        )

        x = agg["LocalSegmentIndex"].values
        # Both columns already live on [0, 1] (see batch_process_pacing_segments) -
        # plot them directly, no further normalization needed.
        actual_mean = agg["OverallPacing_mean"].values
        actual_std = np.nan_to_num(agg["OverallPacing_std"].values)
        target_mean = agg["TargetOverallPacing_mean"].values

        ax.plot(x, actual_mean, color=color, linewidth=2.5, marker="o", markersize=4,
                label=f"{label} Actual", zorder=3)
        ax.fill_between(x, actual_mean - actual_std, actual_mean + actual_std,
                         color=color, alpha=0.15, zorder=1)
        ax.plot(x, target_mean, color=color, linewidth=2, linestyle="--",
                label=f"{label} Target", zorder=2)
        _add_trend_line(ax, x, actual_mean, color, label)

        _draw_baseline(ax, baseline, color, label=f"{label} baseline")
        _add_stats_box(ax, {f"{label} Actual": _stats(round_df["ActualOverallPacingScaled"])})

        title = f"Pacing Target: {target}"
        if round_val is not None:
            title += f" — Round {int(round_val)}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Segment Index", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    constraints = target_df["PacingConstraint"].dropna().unique()
    suptitle = f"Actual vs Target Pacing — {target}"
    if len(constraints) == 1:
        suptitle += f" (constraint: {constraints[0]})"
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_all_pacing_targets(df, output_dir=None, applied_bot=None, width=14, height=5, facet_by_round=False,
                             source_dir=None):
    """
    Iterate every PacingTarget dynamically detected in df and produce one tracking
    chart per target (applied bot's actual OverallPacing vs the predefined
    TargetOverallPacingScaled curve), so pacing filter behavior can be checked
    target-by-target.

    Args:
        df: Combined pacing-segments DataFrame (see plot_pacing_target_tracking)
        output_dir: If provided, saves "<target>.png" for each target into this dir
        applied_bot, width, height, facet_by_round: forwarded to
            plot_pacing_target_tracking
        source_dir: Optional path to the raw simulation logs these charts were
            generated from - stamped as a bottom-left footnote directly on each
            Figure here (rather than later in build_pacing_charts_pdf), so the
            provenance travels with the figure into both the saved PNG and any PDF
            it's later combined into without having to pass source_dir again.

    Returns:
        Dict of {target_name: Figure}
    """
    df = _to_pandas(df)

    targets = sorted(df["PacingTarget"].dropna().unique())
    figs = {}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for target in targets:
        fig = plot_pacing_target_tracking(
            df, target, applied_bot=applied_bot, width=width, height=height,
            facet_by_round=facet_by_round,
        )
        if fig is None:
            continue
        if source_dir:
            _add_source_footer(fig, source_dir)
        figs[target] = fig
        if output_dir:
            safe_name = re.sub(r"[^\w.\-]+", "_", target)
            out_path = os.path.join(output_dir, f"{safe_name}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: {out_path}")

    return figs


def _detect_applied_bots(df):
    if "PacingRole" not in df.columns:
        return []
    return sorted(df.loc[df["PacingRole"] == "Applied", "Bot"].dropna().unique())


def _applied_rows(df, target, applied_bot, include_focus_match=False):
    """
    Shared lookup for the "vs rest" chart family: the applied bot's own rows for
    this target (PacingRole=="Applied"), grouped/averaged downstream by
    "OpponentBot" to produce the "vs all rest" (pooled) and "vs each rest"
    (per-opponent) charts - see module docstring for why only the applied bot's own
    Actual/Target lines are ever plotted here.

    By default excludes "focus match" rows - where OpponentBot is itself an applied
    bot (see module docstring) - so the rest-bot baseline isn't blended with
    filter-vs-filter rounds. Pass include_focus_match=True to fold those back in.
    """
    applied_df = df[
        (df["PacingTarget"] == target) & (df["PacingRole"] == "Applied") & (df["Bot"] == applied_bot)
    ].copy()
    if not include_focus_match and "OpponentBot" in applied_df.columns:
        applied_bots_set = set(_detect_applied_bots(df))
        applied_df = applied_df[~applied_df["OpponentBot"].isin(applied_bots_set)]
    return applied_df


def plot_pacing_applied_vs_all_rest(df, target, applied_bot, width=14, height=5, facet_by_round=False,
                                     include_focus_match=False):
    """
    One applied bot's actual pacing vs its own predefined target curve, averaged
    across ALL of the (non-applied, "rest") opponents it faced under this
    PacingTarget. No opponent line is drawn (see module docstring) - "rest" here
    describes which segments are pooled into the average, not a second line.

    Args:
        include_focus_match: If True, also pool in rounds where the opponent was
            itself an applied bot (see module docstring on "focus match"). Default
            False keeps this chart a clean "applied bot vs the normal roster" view.

    Returns:
        Matplotlib Figure, or None if no data matched
    """
    df = _to_pandas(df)
    applied_df = _applied_rows(df, target, applied_bot, include_focus_match=include_focus_match)

    if applied_df.empty:
        print(f"⚠️ No 'Applied' data found for PacingTarget='{target}', applied_bot={applied_bot}")
        return None

    rounds = sorted(applied_df["RoundIndex"].dropna().unique()) if facet_by_round else [None]
    n_rows = max(len(rounds), 1)

    fig, axes = plt.subplots(n_rows, 1, figsize=(width, height * n_rows), squeeze=False)
    axes = axes.flatten()

    applied_color = get_theme_color("primary")
    opponent_names = sorted(applied_df["OpponentBot"].dropna().unique()) if "OpponentBot" in applied_df.columns else []
    rest_desc = f" (vs {'/'.join(opponent_names)})" if opponent_names else ""

    baseline = _applied_baseline(df, [applied_bot])

    for row_idx, round_val in enumerate(rounds):
        ax = axes[row_idx]
        a_df = applied_df if round_val is None else applied_df[applied_df["RoundIndex"] == round_val]

        agg_a = (
            a_df.groupby("LocalSegmentIndex")
            .agg(mean=("ActualOverallPacingScaled", "mean"), std=("ActualOverallPacingScaled", "std"),
                 target_mean=("TargetOverallPacingScaled", "mean"))
            .reset_index().sort_values("LocalSegmentIndex")
        )
        x = agg_a["LocalSegmentIndex"].values
        a_mean = agg_a["mean"].values
        a_std = np.nan_to_num(agg_a["std"].values)

        ax.plot(x, a_mean, color=applied_color, linewidth=2.5, marker="o", markersize=4,
                label=f"{applied_bot} (Applied) Actual{rest_desc}", zorder=4)
        ax.fill_between(x, a_mean - a_std, a_mean + a_std, color=applied_color, alpha=0.15, zorder=1)
        ax.plot(x, agg_a["target_mean"].values, color=applied_color, linewidth=2, linestyle="--",
                label=f"{applied_bot} Target", zorder=3)
        _add_trend_line(ax, x, a_mean, applied_color, f"{applied_bot} (Applied)")

        stats_map = {f"{applied_bot} Actual": _stats(a_df["ActualOverallPacingScaled"])}

        _draw_baseline(ax, baseline, applied_color, label=f"{applied_bot} baseline")
        _add_stats_box(ax, stats_map)

        title = f"{applied_bot} (Applied) vs All Rest — {target}"
        if round_val is not None:
            title += f" — Round {int(round_val)}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Segment Index", fontsize=9)
        ax.set_ylabel("Pacing (0-1)", fontsize=9)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_pacing_applied_vs_each_rest(df, target, applied_bot, width=14, height=5, facet_by_round=False,
                                      include_focus_match=False):
    """
    Extended version of plot_pacing_applied_vs_all_rest: the SAME applied-bot
    segments (its own Actual/Target pacing, never an opponent's own pacing - see
    module docstring), broken out into one line per individual "rest" opponent
    faced instead of pooled into a single average - i.e. "vs all rest" is the mean
    of this chart's per-opponent lines.

    Args:
        include_focus_match: If True, also include a line for opponents that were
            themselves applied bots (see module docstring on "focus match").
            Default False matches plot_pacing_applied_vs_all_rest.

    Returns:
        Matplotlib Figure, or None if no data matched
    """
    df = _to_pandas(df)
    applied_df = _applied_rows(df, target, applied_bot, include_focus_match=include_focus_match)

    if applied_df.empty:
        print(f"⚠️ No 'Applied' data found for PacingTarget='{target}', applied_bot={applied_bot}")
        return None

    rounds = sorted(applied_df["RoundIndex"].dropna().unique()) if facet_by_round else [None]
    n_rows = max(len(rounds), 1)

    fig, axes = plt.subplots(n_rows, 1, figsize=(width, height * n_rows), squeeze=False)
    axes = axes.flatten()

    applied_color = get_theme_color("primary")
    palette = get_theme_color("categorical")
    opponent_names = sorted(applied_df["OpponentBot"].dropna().unique()) if "OpponentBot" in applied_df.columns else []
    opponent_colors = {bot: palette[i % len(palette)] for i, bot in enumerate(opponent_names)}

    baseline = _applied_baseline(df, [applied_bot])

    for row_idx, round_val in enumerate(rounds):
        ax = axes[row_idx]
        a_df = applied_df if round_val is None else applied_df[applied_df["RoundIndex"] == round_val]

        agg_a = (
            a_df.groupby("LocalSegmentIndex")
            .agg(target_mean=("TargetOverallPacingScaled", "mean"))
            .reset_index().sort_values("LocalSegmentIndex")
        )
        ax.plot(agg_a["LocalSegmentIndex"].values, agg_a["target_mean"].values, color=applied_color,
                linewidth=2, linestyle="--", label=f"{applied_bot} Target", zorder=3)

        stats_map = {}

        for opponent in opponent_names:
            opp_df = a_df[a_df["OpponentBot"] == opponent]
            if opp_df.empty:
                continue
            agg_o = (
                opp_df.groupby("LocalSegmentIndex")
                .agg(mean=("ActualOverallPacingScaled", "mean"), std=("ActualOverallPacingScaled", "std"))
                .reset_index().sort_values("LocalSegmentIndex")
            )
            ox = agg_o["LocalSegmentIndex"].values
            o_mean = agg_o["mean"].values
            o_std = np.nan_to_num(agg_o["std"].values)
            color = opponent_colors[opponent]
            ax.plot(ox, o_mean, color=color, linewidth=2, marker="^", markersize=3.5,
                     label=f"{applied_bot} vs {opponent} Actual", zorder=4)
            ax.fill_between(ox, o_mean - o_std, o_mean + o_std, color=color, alpha=0.1, zorder=1)
            stats_map[f"vs {opponent}"] = _stats(opp_df["ActualOverallPacingScaled"])

        _draw_baseline(ax, baseline, applied_color, label=f"{applied_bot} baseline")
        _add_stats_box(ax, stats_map)

        title = f"{applied_bot} (Applied) vs Each Rest Bot — {target}"
        if round_val is not None:
            title += f" — Round {int(round_val)}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Segment Index", fontsize=9)
        ax.set_ylabel("Pacing (0-1)", fontsize=9)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_all_pacing_targets_vs_all_rest(df, output_dir=None, applied_bots=None, width=14, height=5,
                                         facet_by_round=False, source_dir=None, include_focus_match=False):
    """
    For every applied bot x every PacingTarget, plot that bot's actual pacing vs
    all of its opponents pooled into one "rest" line (plot_pacing_applied_vs_all_rest).

    Args:
        applied_bots: Which applied bots to generate charts for. Defaults to every
            bot that ever appears with PacingRole=="Applied" in df.
        source_dir: Optional path to the raw simulation logs these charts were
            generated from - stamped as a bottom-left footnote directly on each
            Figure here, so provenance travels with it without re-passing
            source_dir to build_pacing_charts_pdf later.
        include_focus_match: Forwarded to plot_pacing_applied_vs_all_rest - see its
            docstring / the module docstring on "focus match".

    Returns:
        Dict of {"<bot>__<target>": Figure}
    """
    df = _to_pandas(df)
    if applied_bots is None:
        applied_bots = _detect_applied_bots(df)
    targets = sorted(df["PacingTarget"].dropna().unique())
    figs = {}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for bot in applied_bots:
        for target in targets:
            fig = plot_pacing_applied_vs_all_rest(
                df, target, bot, width=width, height=height, facet_by_round=facet_by_round,
                include_focus_match=include_focus_match,
            )
            if fig is None:
                continue
            if source_dir:
                _add_source_footer(fig, source_dir)
            key = f"{bot}__{target}"
            figs[key] = fig
            if output_dir:
                safe_name = re.sub(r"[^\w.\-]+", "_", key)
                out_path = os.path.join(output_dir, f"{safe_name}.png")
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"✅ Saved: {out_path}")

    return figs


def plot_all_pacing_targets_vs_each_rest(df, output_dir=None, applied_bots=None, width=14, height=5,
                                          facet_by_round=False, source_dir=None, include_focus_match=False):
    """
    For every applied bot x every PacingTarget, plot that bot's actual pacing vs
    each opponent it faced individually (plot_pacing_applied_vs_each_rest).

    Args:
        applied_bots: Which applied bots to generate charts for. Defaults to every
            bot that ever appears with PacingRole=="Applied" in df.
        source_dir: Optional path to the raw simulation logs these charts were
            generated from - stamped as a bottom-left footnote directly on each
            Figure here, so provenance travels with it without re-passing
            source_dir to build_pacing_charts_pdf later.
        include_focus_match: Forwarded to plot_pacing_applied_vs_each_rest - see its
            docstring / the module docstring on "focus match".

    Returns:
        Dict of {"<bot>__<target>": Figure}
    """
    df = _to_pandas(df)
    if applied_bots is None:
        applied_bots = _detect_applied_bots(df)
    targets = sorted(df["PacingTarget"].dropna().unique())
    figs = {}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for bot in applied_bots:
        for target in targets:
            fig = plot_pacing_applied_vs_each_rest(
                df, target, bot, width=width, height=height, facet_by_round=facet_by_round,
                include_focus_match=include_focus_match,
            )
            if fig is None:
                continue
            if source_dir:
                _add_source_footer(fig, source_dir)
            key = f"{bot}__{target}"
            figs[key] = fig
            if output_dir:
                safe_name = re.sub(r"[^\w.\-]+", "_", key)
                out_path = os.path.join(output_dir, f"{safe_name}.png")
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                print(f"✅ Saved: {out_path}")

    return figs


def compute_pacing_achievement_matrix(df, applied_bots=None, include_focus_match=False):
    """
    For every (applied bot, PacingTarget) combination, score how well the filter's
    actual pacing tracked its own predefined target curve across every segment of
    every match:
        achievement = 1 - mean(|ActualOverallPacingScaled - TargetOverallPacingScaled|)
    (1.0 = perfect tracking every segment, 0.0 = maximally divergent). Only
    PacingRole=="Applied" rows are used - Target* isn't meaningful for opponent rows
    (see module docstring). include_focus_match mirrors _applied_rows: by default
    excludes rounds where the opponent was itself an applied bot.

    Returns:
        DataFrame pivoted Bot (rows) x PacingTarget (columns) -> achievement score,
        or an empty DataFrame if no Applied data matched.
    """
    df = _to_pandas(df)
    if applied_bots is None:
        applied_bots = _detect_applied_bots(df)
    applied_bots = list(applied_bots)

    applied_df = df[(df["PacingRole"] == "Applied") & (df["Bot"].isin(applied_bots))].copy()
    if not include_focus_match and "OpponentBot" in applied_df.columns:
        applied_bots_set = set(applied_bots)
        applied_df = applied_df[~applied_df["OpponentBot"].isin(applied_bots_set)]

    if applied_df.empty:
        return pd.DataFrame()

    applied_df["AbsError"] = (
        applied_df["ActualOverallPacingScaled"] - applied_df["TargetOverallPacingScaled"]
    ).abs()
    grouped = applied_df.groupby(["Bot", "PacingTarget"])["AbsError"].mean().reset_index()
    grouped["Achievement"] = 1.0 - grouped["AbsError"]
    return grouped.pivot(index="Bot", columns="PacingTarget", values="Achievement")


def plot_pacing_achievement_heatmap(df, applied_bots=None, width=10, height=6, include_focus_match=False):
    """
    Heatmap of pacing-target achievement (see compute_pacing_achievement_matrix)
    across every applied bot x PacingTarget combination - answers "which applied
    bot actually achieves which target curve" at a glance, instead of paging
    through the per-target/per-bot line charts one at a time.

    Returns:
        Matplotlib Figure, or None if no Applied data matched
    """
    df = _to_pandas(df)
    matrix = compute_pacing_achievement_matrix(df, applied_bots=applied_bots, include_focus_match=include_focus_match)
    if matrix.empty:
        print("⚠️ No 'Applied' pacing data found to build achievement heatmap")
        return None

    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        matrix, annot=True, cmap=get_theme_color("heatmap_cmap"), vmin=0, vmax=1,
        fmt=".2f", linewidths=0.5, cbar_kws={"label": "Achievement (1 - mean |Actual - Target|)"}, ax=ax,
    )
    ax.set_title("Pacing Target Achievement — Applied Bot x PacingTarget", fontsize=13, fontweight="bold")
    ax.set_xlabel("PacingTarget", fontsize=9)
    ax.set_ylabel("Applied Bot", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_pacing_bias_curve(df, target, applied_bots=None, width=14, height=5, include_focus_match=False):
    """
    Signed tracking error (Actual - Target, NOT absolute) over segment index, one
    line per applied bot, for a single PacingTarget - the "when did it drift, and
    which direction" complement to compute_pacing_achievement_matrix's single
    per-bot/target MAE number, which collapses over-vs-under-shoot into one
    magnitude and says nothing about timing.

    A positive value means the bot's actual pacing ran ABOVE the target curve at
    that segment (overshoot); negative means it ran below (lag/undershoot). This is
    particularly informative for step_increase/step_decrease PacingTargets, where a
    filter with reaction lag shows up as a visible spike/dip right at the step
    rather than as elevated MAE spread evenly across the round.

    Returns:
        Matplotlib Figure, or None if no Applied data matched for any applied bot
    """
    df = _to_pandas(df)
    if applied_bots is None:
        applied_bots = _detect_applied_bots(df)

    fig, ax = plt.subplots(figsize=(width, height))
    palette = get_theme_color("categorical")
    plotted_any = False

    for i, bot in enumerate(applied_bots):
        bot_df = _applied_rows(df, target, bot, include_focus_match=include_focus_match)
        if bot_df.empty:
            continue
        bot_df = bot_df.copy()
        bot_df["SignedError"] = bot_df["ActualOverallPacingScaled"] - bot_df["TargetOverallPacingScaled"]
        agg = (
            bot_df.groupby("LocalSegmentIndex")
            .agg(mean=("SignedError", "mean"), std=("SignedError", "std"))
            .reset_index().sort_values("LocalSegmentIndex")
        )
        x = agg["LocalSegmentIndex"].values
        mean = agg["mean"].values
        std = np.nan_to_num(agg["std"].values)
        color = palette[i % len(palette)]
        ax.plot(x, mean, color=color, linewidth=2.5, marker="o", markersize=4, label=bot, zorder=3)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, zorder=1)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"⚠️ No 'Applied' data found for PacingTarget='{target}'")
        return None

    ax.axhline(0.0, color="#666666", linewidth=1.2, linestyle="--", label="Perfect tracking", zorder=2)
    # Center the y-axis on zero so overshoot/undershoot read as symmetric deviations
    # rather than being visually skewed by whichever direction happened to dominate.
    y_extent = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-y_extent, y_extent)

    ax.set_title(f"Pacing Bias (Actual − Target) — {target}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Segment Index", fontsize=9)
    ax.set_ylabel("Signed Error (Actual − Target)", fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_all_pacing_bias_curves(df, output_dir=None, applied_bots=None, width=14, height=5,
                                 source_dir=None, include_focus_match=False):
    """
    plot_pacing_bias_curve for every PacingTarget detected in df, all applied bots
    overlaid on the same axes per target so their bias/lag shapes can be compared
    directly.

    Returns:
        Dict of {target_name: Figure}
    """
    df = _to_pandas(df)
    if applied_bots is None:
        applied_bots = _detect_applied_bots(df)
    targets = sorted(df["PacingTarget"].dropna().unique())
    figs = {}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for target in targets:
        fig = plot_pacing_bias_curve(
            df, target, applied_bots=applied_bots, width=width, height=height,
            include_focus_match=include_focus_match,
        )
        if fig is None:
            continue
        if source_dir:
            _add_source_footer(fig, source_dir)
        figs[target] = fig
        if output_dir:
            safe_name = re.sub(r"[^\w.\-]+", "_", target)
            out_path = os.path.join(output_dir, f"{safe_name}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: {out_path}")

    return figs


def compute_achievement_win_data(df, applied_bots=None, include_focus_match=False):
    """
    Per (Bot, PacingTarget, GameIndex, RoundIndex) round actually played, pairs
    that round's pacing achievement (1 - mean |Actual - Target| across just that
    round's segments) with whether the applied bot won that round.

    RoundWinner is logged match-wide as 0/1/2 (Left/Right/Draw - see
    compile.log_to_parquet), not per-side, so it's resolved against this row's own
    "Side" to get a per-bot Won flag. Draws are dropped (win rate is undefined for
    them). This is round-level granularity specifically so
    plot_pacing_achievement_vs_winrate has enough independent samples to bin -
    compute_pacing_achievement_matrix's single MAE number per bot x target
    couldn't answer "does better tracking actually correlate with winning".

    Returns:
        DataFrame with columns [Bot, PacingTarget, GameIndex, RoundIndex,
        Achievement, Won], or empty if no Applied data matched.
    """
    df = _to_pandas(df)
    if applied_bots is None:
        applied_bots = _detect_applied_bots(df)
    applied_bots = list(applied_bots)

    applied_df = df[(df["PacingRole"] == "Applied") & (df["Bot"].isin(applied_bots))].copy()
    if not include_focus_match and "OpponentBot" in applied_df.columns:
        applied_bots_set = set(applied_bots)
        applied_df = applied_df[~applied_df["OpponentBot"].isin(applied_bots_set)]
    if applied_df.empty:
        return pd.DataFrame()

    applied_df["AbsError"] = (
        applied_df["ActualOverallPacingScaled"] - applied_df["TargetOverallPacingScaled"]
    ).abs()
    applied_df["Won"] = np.select(
        [
            applied_df["RoundWinner"] == 2,
            (applied_df["Side"] == "Left") & (applied_df["RoundWinner"] == 0),
            (applied_df["Side"] == "Right") & (applied_df["RoundWinner"] == 1),
        ],
        [np.nan, 1.0, 1.0],
        default=0.0,
    )

    grouped = (
        applied_df.groupby(["Bot", "PacingTarget", "GameIndex", "RoundIndex"])
        .agg(Achievement=("AbsError", lambda s: 1.0 - s.mean()), Won=("Won", "first"))
        .reset_index()
    )
    return grouped.dropna(subset=["Won"])


def plot_pacing_achievement_vs_winrate(df, applied_bots=None, n_bins=4, width=10, height=6,
                                        include_focus_match=False):
    """
    Does tracking the target pacing curve well actually correlate with winning?
    Bins rounds into per-bot quantiles of round-level pacing achievement (see
    compute_achievement_win_data) and plots the win rate within each bin, with the
    Pearson correlation between achievement and Won annotated per bot - the "does
    this even matter for match outcomes" complement to the achievement heatmap and
    bias curves, which only ever measure tracking quality in isolation from
    results.

    Args:
        n_bins: Number of achievement quantile bins per bot (fewer bins with more
            rounds each = a smoother but coarser curve)

    Returns:
        Matplotlib Figure, or None if there wasn't enough round-level data to bin
    """
    df = _to_pandas(df)
    data = compute_achievement_win_data(df, applied_bots=applied_bots, include_focus_match=include_focus_match)
    if data.empty:
        print("⚠️ No round-level Applied/Win data found to compare achievement vs win rate")
        return None

    palette = get_theme_color("categorical")
    bots = sorted(data["Bot"].unique())

    fig, ax = plt.subplots(figsize=(width, height))
    corr_lines = []

    for i, bot in enumerate(bots):
        bot_data = data[data["Bot"] == bot].copy()
        if len(bot_data) < n_bins:
            continue
        try:
            bot_data["Bin"] = pd.qcut(bot_data["Achievement"], q=n_bins, duplicates="drop")
        except ValueError:
            continue
        binned = (
            bot_data.groupby("Bin", observed=True)
            .agg(WinRate=("Won", "mean"), AchievementMid=("Achievement", "mean"), n=("Won", "size"))
            .reset_index().sort_values("AchievementMid")
        )
        color = palette[i % len(palette)]
        ax.plot(binned["AchievementMid"], binned["WinRate"], color=color, linewidth=2.5,
                marker="o", markersize=6, label=f"{bot} (n={len(bot_data)})", zorder=3)
        corr = bot_data["Achievement"].corr(bot_data["Won"])
        corr_lines.append(f"{bot}: corr(achievement, win) = {corr:+.3f}")

    if not corr_lines:
        plt.close(fig)
        print(f"⚠️ Not enough rounds per bot (need >= {n_bins}) to bin achievement vs win rate")
        return None

    ax.text(
        0.02, 0.02, "\n".join(corr_lines), transform=ax.transAxes, fontsize=8,
        va="bottom", ha="left", family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999", alpha=0.85), zorder=5,
    )
    ax.set_title("Round Win Rate vs Pacing Achievement", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Pacing Achievement (binned into {n_bins} quantiles per bot)", fontsize=9)
    ax.set_ylabel("Round Win Rate", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def _add_source_footer(fig, source_dir):
    """Stamp a small provenance footnote in the bottom-left of a chart page, so
    each page carries where its data came from even viewed/extracted on its own."""
    fig.text(0.01, 0.005, f"Source: {source_dir}", fontsize=6, color="#888888", ha="left", va="bottom")


def build_pacing_charts_pdf(figs_by_name, output_path, source_dir=None):
    """
    Write every pacing chart Figure produced by plot_all_pacing_targets /
    plot_all_pacing_targets_vs_all_rest / plot_all_pacing_targets_vs_each_rest into
    a single multi-page PDF, one chart per page, so they can all be reviewed/shared
    from one file instead of scrolling through many notebook cells.

    Figures are written directly (not re-rasterized through a saved PNG), so every
    page stays vector/lossless - text and lines stay crisp at any zoom level.

    Args:
        figs_by_name: Ordered mapping (or iterable) of {name: Figure} - e.g. the
            combined dicts returned by the three plot_all_pacing_targets* calls
        output_path: Path to write the .pdf to
        source_dir: Optional fallback path to the raw simulation logs, stamped on
            every page same as _add_source_footer. Normally unnecessary - pass
            source_dir to the plot_all_pacing_targets* calls instead so each Figure
            already carries its footer by the time it gets here, and this function
            doesn't need it repeated. Only use this if those figures weren't
            already stamped (e.g. figures from an older/skipped call).

    Returns:
        The output_path, or None if there were no figures to write
    """
    from matplotlib.backends.backend_pdf import PdfPages

    figs = list(figs_by_name.values()) if hasattr(figs_by_name, "values") else list(figs_by_name)
    figs = [fig for fig in figs if fig is not None]
    if not figs:
        print("⚠️ No figures found to write to PDF")
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with PdfPages(output_path) as pdf:
        for fig in figs:
            if source_dir:
                _add_source_footer(fig, source_dir)
            pdf.savefig(fig)

    print(f"✅ Saved {len(figs)}-page PDF: {output_path}")
    if source_dir:
        print(f"   Source: {source_dir}")
    return output_path
