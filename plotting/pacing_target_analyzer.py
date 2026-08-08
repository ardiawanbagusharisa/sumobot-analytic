"""
Visualize dynamic pacing target tracking.

Compares the actual pacing produced by the action filter (ActualOverallPacingScaled)
against the predefined target curve (TargetOverallPacing, raw) that the filter was
supposed to steer the match towards - one chart per PacingTarget (e.g.
"linear_decrease_0.6_to_0.4_60s", "step_increase_0.0_to_1.0", ...).

TargetTempo/TargetThreat/TargetOverallPacing are already raw values in [0, 1] - the
engine's runtime lerp put them there directly, so they need no transform. Actual
Tempo/Threat/OverallPacing are raw composite scores bounded by [MinPacing,
MaxPacing] (e.g. a run with MaxPacing=0.6 means an OverallPacing of 0.6 is the
ceiling), so compile.generator.batch_process_pacing_segments min-max normalizes
*them* into [0, 1] and caps (ActualTempoScaled/ActualThreatScaled/
ActualOverallPacingScaled) to make the two directly comparable on the same 0-1
scale.

Expects the combined DataFrame produced by:
    compile.generator.batch_process_pacing_segments(..., applied_bots=[...])
    compile.generator.generate_pacing_segments_from_batches(...)
which carries (among others) these columns:
    PacingTarget, PacingConstraint, PacingRole, Bot, RoundIndex, LocalSegmentIndex,
    ActualOverallPacingScaled, TargetOverallPacing

The filter is applied to specific bots (identity), not to a fixed Left/Right slot -
matchups are logged in both directions (BotA_vs_BotB and BotB_vs_BotA), so the
applied bot switches sides across folders. Only the applied bot's row actually had
the pacing filter steering it toward the target - the opponent's Target* fields are
logged by the engine for both sides regardless, but aren't meaningful for the
opponent, so only the "Applied" role is plotted here.
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

from .analyzer_config import get_theme_color


def _to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def plot_pacing_target_tracking(df, target, applied_bot=None, width=14, height=5, facet_by_round=False):
    """
    Plot the applied bot's actual pacing (ActualOverallPacingScaled - the raw
    composite score min-max normalized into [0, 1] and capped) vs the predefined
    target curve (TargetOverallPacing, already raw 0-1) over segment index, for a
    single dynamic pacing target, to visually check whether the action filter tracks
    the intended pacing curve.

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

    for row_idx, round_val in enumerate(rounds):
        ax = axes[row_idx]
        round_df = target_df if round_val is None else target_df[target_df["RoundIndex"] == round_val]

        agg = (
            round_df.groupby("LocalSegmentIndex")
            .agg(
                OverallPacing_mean=("ActualOverallPacingScaled", "mean"),
                OverallPacing_std=("ActualOverallPacingScaled", "std"),
                TargetOverallPacing_mean=("TargetOverallPacing", "mean"),
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


def plot_all_pacing_targets(df, output_dir=None, applied_bot=None, width=14, height=5, facet_by_round=False):
    """
    Iterate every PacingTarget dynamically detected in df and produce one tracking
    chart per target (applied bot's actual OverallPacing vs the predefined
    TargetOverallPacing curve), so pacing filter behavior can be checked
    target-by-target.

    Args:
        df: Combined pacing-segments DataFrame (see plot_pacing_target_tracking)
        output_dir: If provided, saves "<target>.png" for each target into this dir
        applied_bot, width, height, facet_by_round: forwarded to
            plot_pacing_target_tracking

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
        figs[target] = fig
        if output_dir:
            safe_name = re.sub(r"[^\w.\-]+", "_", target)
            out_path = os.path.join(output_dir, f"{safe_name}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: {out_path}")

    return figs
