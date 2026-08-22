"""
Polars GPU-accelerated generator with batch processing
Supports both CSV and Parquet input formats
Uses scan_csv/scan_parquet for lazy evaluation and GPU acceleration
Aggregation logic from generator_duckdb_polars.py
Batch processing pattern from generator.py
"""

import time
import os
import re
import glob
from functools import lru_cache
import polars as pl
import numpy as np
import pandas as pd  # For pd.cut in time bins
import json
from pathlib import Path

from compile.log_to_parquet import parse_pacing_folder_name

# Check if GPU support is available
GPU_AVAILABLE = False
try:
    # Try a simple GPU operation to check availability
    pl.LazyFrame({"test": [1]}).collect(engine="gpu")
    GPU_AVAILABLE = True
    print("GPU support available - will use GPU acceleration")
except Exception:
    print("Using CPU (GPU not available)")

# Load arena configuration from unified config.json
_config_path = Path(__file__).parent.parent / "config.json"
with open(_config_path, 'r') as f:
    _config = json.load(f)

arena_center = np.array(_config['arena']['center'])
arena_radius = _config['arena']['radius']

# Factors that need inversion (1 - normalized_value)
# In C#: Angle, SafeDistance, BotsDistance are inverted
INVERTED_FACTORS = {'Angle', 'SafeDistance', 'BotsDistance'}

def normalize_pacing_factor(value, factor_name, constraints=None):
    """
    Normalize a pacing factor value to 0-1 range using constraints.
    Matches C# ConstraintMinMax.Normalize() behavior.

    Args:
        value: Raw factor value (can be NaN)
        factor_name: Name of the factor (without _L or _R suffix)
        constraints: Dictionary of constraints {factor_name: (min, max)} or None for raw data

    Returns:
        Normalized value in [0, 1] range, NaN if input is NaN, or raw value if constraints is None
    """
    if np.isnan(value):
        return np.nan

    # If constraints is None, return raw value (no normalization)
    if constraints is None:
        return float(value)

    # Get constraints for this factor
    min_val, max_val = constraints.get(factor_name, (0.0, 1.0))

    # Normalize: (value - min) / (max - min)
    if max_val - min_val == 0:
        normalized = 0.0
    else:
        normalized = (value - min_val) / (max_val - min_val)

    # Clamp to [0, 1]
    normalized = np.clip(normalized, 0.0, 1.0)

    # Apply inversion if needed (matching C# behavior)
    # if factor_name in INVERTED_FACTORS:
    #    normalized = 1.0 - normalized

    return float(normalized)

def collect_with_gpu(lf):
    """Helper to collect LazyFrame with GPU if available"""
    if GPU_AVAILABLE:
        return lf.collect(engine="gpu")
    else:
        return lf.collect()


@lru_cache(maxsize=None)
def parse_config_name_cached(name):
    return parse_config_name(name)


def parse_config_name(config_name: str):
    """Extract structured info from config folder name"""
    segments = config_name.split("__")
    config = {}

    for seg in segments:
        if "_" in seg:
            key, value = seg.split("_", 1)
            config[key] = value
        else:
            config[seg] = True

    for k, v in config.items():
        if isinstance(v, str) and re.match(r"^-?\d+(\.\d+)?$", v):
            config[k] = float(v)

    return config


def scan_file(file_path):
    """
    Scan a file (CSV or Parquet) and return a LazyFrame

    Args:
        file_path: Path to CSV or Parquet file

    Returns:
        Polars LazyFrame
    """
    if file_path.endswith('.parquet'):
        return pl.scan_parquet(file_path)
    elif file_path.endswith('.csv'):
        return pl.scan_csv(file_path, infer_schema_length=10000)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Only .csv and .parquet are supported.")


def process_batch_csvs(csv_paths, batch_checkpoint_dir="batched", time_bin_size=None, compute_timebins=False, compute_pacing=False):
    """
    Process a batch of CSV files and create checkpoint

    Args:
        csv_paths: List of CSV file paths to process
        batch_checkpoint_dir: Directory to save checkpoints
        time_bin_size: Size of time bins (only used if compute_timebins=True or compute_pacing=True)
        compute_timebins: Whether to compute time-binned data
        compute_pacing: Whether to compute pacing factors

    Returns:
        tuple: (batch_df, action_timebin_df, collision_timebin_df, pacing_factors_df)
    """
    os.makedirs(batch_checkpoint_dir, exist_ok=True)

    all_games_list = []
    time_fragment_list = [] if compute_timebins else None
    collision_fragment_list = [] if compute_timebins else None
    pacing_fragment_list = [] if compute_pacing else None

    for csv_path in csv_paths:
        # Extract bot names and config from path
        # Expected path: base_dir/BotA_vs_BotB/ConfigName/log.csv
        parts = csv_path.split(os.sep)
        matchup_folder = parts[-3]
        config_folder = parts[-2]

        match = re.match(r"(.+)_vs_(.+)", matchup_folder)
        if not match:
            continue
        bot_a, bot_b = match.groups()

        # Parse config
        config = parse_config_name_cached(config_folder)

        # Scan file (CSV or Parquet) with Polars lazy API
        lf = scan_file(csv_path)

        # Process game metrics
        game_metrics_lf = process_single_csv_lazy(
            lf,
            bot_a,
            bot_b,
            config.get('Timer'),
            config.get('ActInterval'),
            config.get('Round'),
            config.get('SkillLeft'),
            config.get('SkillRight')
        )

        # Collect the results
        game_metrics_df = collect_with_gpu(game_metrics_lf)
        all_games_list.append(game_metrics_df)

        # Process time bins if requested
        if compute_timebins and time_bin_size:
            # Process action time bins
            action_tb = process_action_timebins_single_csv(
                lf, bot_a, bot_b, config, time_bin_size
            )
            if action_tb:
                time_fragment_list.extend(action_tb)

            # Process collision time bins
            collision_tb = process_collision_timebins_single_csv(
                lf, bot_a, bot_b, config, time_bin_size
            )
            if collision_tb:
                collision_fragment_list.extend(collision_tb)

        # Process pacing factors if requested
        if compute_pacing and time_bin_size:
            pacing_tb = process_pacing_factors_timebins_single_csv(
                lf, bot_a, bot_b, config, time_bin_size
            )
            if pacing_tb:
                pacing_fragment_list.extend(pacing_tb)

    # Concatenate all games from this batch
    batch_df = None
    action_timebin_df = None
    collision_timebin_df = None
    pacing_factors_df = None

    if all_games_list:
        batch_df = pl.concat(all_games_list)

    if compute_timebins:
        if time_fragment_list:
            action_timebin_df = pl.DataFrame(time_fragment_list)
        if collision_fragment_list:
            collision_timebin_df = pl.DataFrame(collision_fragment_list)

    if compute_pacing:
        if pacing_fragment_list:
            pacing_factors_df = pl.DataFrame(pacing_fragment_list)

    return batch_df, action_timebin_df, collision_timebin_df, pacing_factors_df


def process_action_timebins_single_csv(lf, bot_a, bot_b, config, time_bin_size):
    """
    Process action time bins for a single CSV file
    Returns list of time-binned action-session records: one count per held action
    overlapping a bin, not per raw resubmission tick (see build_action_sessions),
    and split per (GameIndex, RoundIndex) since Best-of-N rounds each restart their
    own StartedAt/UpdatedAt clock near 0.
    """
    match_dur_lf = lf.group_by(["GameIndex", "RoundIndex"]).agg([
        pl.col("UpdatedAt").max().alias("match_duration")
    ])
    match_durations = collect_with_gpu(match_dur_lf)

    # State==2 (release) rows are kept, not filtered out: build_action_sessions needs
    # their UpdatedAt as the authoritative end-of-hold timestamp.
    action_data_lf = lf.with_row_index("_orig_idx").filter(
        pl.col("Category") == "Action"
    ).select([
        "_orig_idx", "GameIndex", "RoundIndex", "Actor", "StartedAt", "UpdatedAt", "Duration", "Name", "State"
    ])
    action_data_all = _dedupe_action_ticks(collect_with_gpu(action_data_lf).to_pandas())
    action_sessions_all = build_action_sessions(action_data_all)

    time_fragment_list = []

    # Process time bins per (GameIndex, RoundIndex)
    for game_idx, round_idx in zip(match_durations['GameIndex'], match_durations['RoundIndex']):
        match_dur = match_durations.filter(
            (pl.col('GameIndex') == game_idx) & (pl.col('RoundIndex') == round_idx)
        )['match_duration'][0]

        # Use timer config to determine max time bin (cap at timer setting)
        timer_value = config.get('Timer')
        max_time = min(match_dur, timer_value) if timer_value else match_dur

        bins = np.arange(0, max_time + time_bin_size, time_bin_size)
        if len(bins) < 2:
            continue

        sessions = action_sessions_all[
            (action_sessions_all['GameIndex'] == game_idx) & (action_sessions_all['RoundIndex'] == round_idx)
        ]
        if len(sessions) == 0:
            continue

        for side in [0, 1]:
            actor_sessions = sessions[sessions['Actor'] == side]
            if len(actor_sessions) == 0:
                continue

            for i in range(len(bins) - 1):
                time_start, time_end = bins[i], bins[i + 1]
                # Sessions active/overlapping during this bin (not sessions that merely *start* in it)
                bin_sessions = actor_sessions[
                    (actor_sessions['SessionStart'] < time_end) & (actor_sessions['SessionEnd'] > time_start)
                ]
                if len(bin_sessions) == 0:
                    continue

                grouped = bin_sessions.groupby('Name', observed=False).size().reset_index(name='Count')

                for _, row in grouped.iterrows():
                    time_fragment_list.append({
                        'GameIndex': game_idx,
                        'RoundIndex': round_idx,
                        'Bot': bot_a if side == 0 else bot_b,
                        'Timer': config.get('Timer'),
                        'ActInterval': config.get('ActInterval'),
                        'Round': config.get('Round'),
                        'SkillLeft': config.get('SkillLeft'),
                        'SkillRight': config.get('SkillRight'),
                        'TimeBin': float(time_start),
                        'Action': row['Name'],
                        'Count': row['Count']
                    })

    return time_fragment_list


def process_collision_timebins_single_csv(lf, bot_a, bot_b, config, time_bin_size):
    """
    Process collision time bins for a single CSV file
    Returns list of time-binned collision records, split per (GameIndex, RoundIndex)
    since Best-of-N rounds each restart their own StartedAt/UpdatedAt clock near 0.
    """
    match_dur_lf = lf.group_by(["GameIndex", "RoundIndex"]).agg([
        pl.col("UpdatedAt").max().alias("match_duration")
    ])
    match_durations = collect_with_gpu(match_dur_lf)

    # Scan and filter for collisions
    # Cast State to string for consistent comparison (handles both CSV strings and Parquet types)
    # Collision rows don't carry a meaningful StartedAt (it's always 0 — verified across
    # multiple matchup files); UpdatedAt is the actual collision timestamp.
    raw_data = lf.filter(
        (pl.col("Category") == "Collision") & (pl.col("State").cast(pl.Utf8) == "0")
    ).select([
        "GameIndex", "RoundIndex", "Actor", "ColTieBreaker", "ColActor", "UpdatedAt"
    ]).unique(subset=["GameIndex", "RoundIndex", "Actor", "UpdatedAt"], keep="first")  # Guard against duplicate collision logs (e.g. re-fired OnCollisionEnter2D) for the same actor/timestamp
    raw_data_df = collect_with_gpu(raw_data)

    collision_fragment_list = []

    # Process collision time bins per (GameIndex, RoundIndex)
    for game_idx, round_idx in zip(match_durations['GameIndex'], match_durations['RoundIndex']):
        match_dur = match_durations.filter(
            (pl.col('GameIndex') == game_idx) & (pl.col('RoundIndex') == round_idx)
        )['match_duration'][0]

        # Use timer config to determine max time bin (cap at timer setting)
        timer_value = config.get('Timer')
        max_time = min(match_dur, timer_value) if timer_value else match_dur

        bins = np.arange(0, max_time + time_bin_size, time_bin_size)
        if len(bins) < 2:
            continue

        game_df = raw_data_df.filter(
            (pl.col('GameIndex') == game_idx) & (pl.col('RoundIndex') == round_idx)
        )
        if len(game_df) == 0:
            continue

        game_pd = game_df.to_pandas()
        game_pd['TimeBin'] = pd.cut(game_pd['UpdatedAt'], bins=bins,
                                   labels=bins[:-1], include_lowest=True)

        for time_bin, bin_data in game_pd.groupby('TimeBin', observed=False):
            actor_L_count = len(bin_data[(bin_data['Actor'] == True) &
                                        (bin_data['ColTieBreaker'] == False) &
                                        (bin_data["ColActor"] == True)])
            actor_R_count = len(bin_data[(bin_data['Actor'] == False) &
                                        (bin_data['ColTieBreaker'] == False) &
                                        (bin_data["ColActor"] == True)])

            tie = bin_data['ColTieBreaker'].sum() if 'ColTieBreaker' in bin_data.columns else 0

            collision_fragment_list.append({
                'GameIndex': game_idx,
                'RoundIndex': round_idx,
                'Bot_L': bot_a,
                'Bot_R': bot_b,
                'Timer': config.get('Timer'),
                'ActInterval': config.get('ActInterval'),
                'Round': config.get('Round'),
                'SkillLeft': config.get('SkillLeft'),
                'SkillRight': config.get('SkillRight'),
                'TimeBin': float(time_bin),
                'Actor_L': actor_L_count,
                'Actor_R': actor_R_count,
                'Tie': int(tie),
            })

    return collision_fragment_list


def build_action_sessions(action_df):
    """
    Collapse action rows into contiguous sessions per (GameIndex, RoundIndex, Actor, Name).

    Some matchups (e.g. ones involving Bot_LLM_ActionGPT) resubmit the same held action
    every simulation frame instead of once per ActInterval. SumoController.Log() kills and
    re-logs a fresh row every time an already-active action is resubmitted, so a single
    continuous "hold" gets fragmented into dozens of rows per second (verified: median
    inter-row gap for those matchups is ~0.02s regardless of the configured ActInterval),
    wildly inflating raw row counts. A row is a renewal of the current session (not a new
    action) if it starts before every prior row in the group has already expired
    (StartedAt + Duration); otherwise it starts a new session.

    Duration on a live (State 0/1) row is only a forward-looking "valid until" estimate —
    it's ~ActInterval-sized and resets on every renewal. The State==2 row that closes a
    session is the one authoritative signal for when the hold actually ended, and its own
    Duration is stale (still the last renewal's estimate), understating true elapsed hold
    time by up to hundreds of ms (verified against raw logs: a hold renewed at t=0.22 with
    Duration=0.1 didn't actually release until UpdatedAt=0.79). So the terminal row's
    UpdatedAt is used as its end instead of StartedAt + Duration; requires the input to
    include State==2 rows (do not filter them out before calling this) and a State column.
    """
    cols = ["GameIndex", "RoundIndex", "Actor", "Name", "SessionStart", "SessionEnd"]
    if action_df.empty:
        return pd.DataFrame(columns=cols)

    group_keys = ["GameIndex", "RoundIndex", "Actor", "Name"]
    df = action_df.sort_values(group_keys + ["StartedAt"]).reset_index(drop=True)
    is_terminal = df["State"].astype(int) == 2
    df["_end"] = np.where(is_terminal, df["UpdatedAt"], df["StartedAt"] + df["Duration"])
    df["_running_end"] = df.groupby(group_keys)["_end"].cummax()
    prev_running_end = df.groupby(group_keys)["_running_end"].shift(1)
    df["_new_session"] = df["StartedAt"] > prev_running_end.fillna(-np.inf)
    df["_session_id"] = df.groupby(group_keys)["_new_session"].cumsum()

    sessions = df.groupby(group_keys + ["_session_id"], as_index=False).agg(
        SessionStart=("StartedAt", "min"),
        SessionEnd=("_end", "max"),
    )
    return sessions[cols]


def _dedupe_action_ticks(df):
    """
    Keep exactly one row per (GameIndex, RoundIndex, Actor, Name, StartedAt) tick.
    Older log versions logged every queued action instead of only the last one submitted
    per tick. A State==2 (release) row must win any same-tick tie against a State 0/1 row,
    since it carries the authoritative end-of-hold UpdatedAt that build_action_sessions
    needs — losing it to an arbitrary row-order tiebreak would silently reintroduce the
    stale-Duration session-end bug.
    """
    return (
        df.assign(_is_terminal=(df["State"].astype(int) == 2).astype(int))
        .sort_values(["_is_terminal", "_orig_idx"])
        .groupby(["GameIndex", "RoundIndex", "Actor", "Name", "StartedAt"], as_index=False, sort=False)
        .tail(1)
        .drop(columns=["_orig_idx", "_is_terminal"])
    )


def process_pacing_factors_timebins_single_csv(lf, bot_a, bot_b, config, time_bin_size, skip_initial=0.0, constraints=None, collision_window_n=1):
    """
    Process pacing factors per timebin for a single CSV/Parquet file
    Calculates and normalizes 8 pacing factors for both bots in each timebin:
    - Threat: CollisionRatio, AbilityRatio, Angle, SafeDistance
    - Tempo: ActionIntensity, ActionDensity, BotsDistance, Velocity

    All factors are normalized to [0, 1] range using constraints matching C# implementation.
    Inverted factors (Angle, SafeDistance, BotsDistance) are calculated as (1 - normalized_value).

    Args:
        lf: Lazy frame of the data
        bot_a: Name of bot A
        bot_b: Name of bot B
        config: Configuration dictionary
        time_bin_size: Size of time bins in seconds
        skip_initial: Number of seconds to skip at the start (default: 0.0)
        constraints: Dictionary of constraints {factor_name: (min, max)} or None for raw data (default: None)
        collision_window_n: Number of seconds to look back for collision calculation (default: 1 second)
                           e.g., at time 5s with N=3, collects collisions from [2s, 5s]

    Returns list of time-binned pacing factor records with normalized values (per matchup)
    """
    # Get match duration per (GameIndex, RoundIndex). Rounds in a Best-of-N match each
    # restart their StartedAt/UpdatedAt clock near 0, so scoping only by GameIndex would
    # blend up to N rounds' independent timelines into the same time bins, inflating any
    # count-based factor (verified: up to ~4-5x inflation on Best-of-5 files).
    match_dur_lf = lf.group_by(["GameIndex", "RoundIndex"]).agg([
        pl.col("UpdatedAt").max().alias("match_duration")
    ])
    match_durations = collect_with_gpu(match_dur_lf)

    pacing_fragment_list = []

    # ActInterval-sized "decision tick" unit for ActionIntensity/ActionDensity: a session's
    # overlap with a bin is expressed in units of this size rather than as a raw session
    # count, so a continuous hold spanning the whole bin scores ~bin_size/ActInterval
    # (matching how many decisions it represents) instead of a flat 1 regardless of length.
    act_interval = config.get('ActInterval') or 1.0

    # ===== 1. ACTION DATA (for ActionIntensity, ActionDensity, AbilityRatio, Angle) =====
    # Older log versions logged every queued action instead of only the last one
    # submitted per tick (StartedAt/UpdatedAt are set once per Update() call, so same
    # Actor+Name+StartedAt rows belong to the same tick). Keep only the last-submitted
    # row per tick, matching current InputProvider.Flush()'s "unique per tick" behavior.
    # State==2 (release) rows are kept, not filtered out: build_action_sessions needs
    # their UpdatedAt as the authoritative end-of-hold timestamp.
    action_data_lf = lf.with_row_index("_orig_idx").filter(
        (pl.col("Category") == "Action") &
        (pl.col("StartedAt") >= skip_initial)  # Skip initial period
    ).select([
        "_orig_idx", "GameIndex", "RoundIndex", "Actor", "StartedAt", "UpdatedAt", "Duration", "Name", "State",
        "BotPosX", "BotPosY", "BotRot",
        "EnemyBotPosX", "EnemyBotPosY"
    ])
    action_data_all = _dedupe_action_ticks(collect_with_gpu(action_data_lf).to_pandas())

    # Some matchups (e.g. involving Bot_LLM_ActionGPT) resubmit the same held action every
    # simulation frame instead of once per ActInterval, so a single continuous hold gets
    # fragmented into dozens of distinct-tick rows per second. Collapse those into sessions
    # and count sessions overlapping a bin instead of raw rows for the count-based factors
    # (ActionIntensity, ActionDensity, AbilityRatio), so a continuous hold counts once
    # (per bin it overlaps) rather than once per frame.
    action_sessions_all = build_action_sessions(action_data_all)

    # ===== 2. COLLISION DATA (for CollisionRatio) =====
    # Collision rows don't carry a meaningful StartedAt (it's always 0 — verified across
    # multiple matchup files); UpdatedAt is the actual collision timestamp, unlike Action
    # rows where StartedAt is the real per-row time. Binning/filtering on StartedAt would
    # make every collision only ever visible in a window that reaches back to t=0.
    collision_data_lf = lf.filter(
        (pl.col("Category") == "Collision") &
        (pl.col("State").cast(pl.Utf8) == "0") &
        (pl.col("UpdatedAt") >= skip_initial)  # Skip initial period
    ).select([
        "GameIndex", "RoundIndex", "Actor", "UpdatedAt", "ColTieBreaker", "ColActor",
        "BotPosX", "BotPosY", "BotRot", "BotLinv",
        "EnemyBotPosX", "EnemyBotPosY", "EnemyBotRot", "EnemyBotLinv"
    ]).unique(subset=["GameIndex", "RoundIndex", "Actor", "UpdatedAt"], keep="first")  # Guard against duplicate collision logs (e.g. re-fired OnCollisionEnter2D) for the same actor/timestamp
    collision_data_all = collect_with_gpu(collision_data_lf).to_pandas()

    # ===== 3. GENERAL POSITION DATA (for BotsDistance, Velocity) =====
    # Sample from all categories to get average distance/velocity
    position_data_lf = lf.filter(
        (pl.col("StartedAt") >= skip_initial)  # Skip initial period
    ).select([
        "GameIndex", "RoundIndex", "Actor", "StartedAt",
        "BotPosX", "BotPosY", "BotLinv",
        "EnemyBotPosX", "EnemyBotPosY", "EnemyBotLinv"
    ]).unique(subset=["GameIndex", "RoundIndex", "Actor", "StartedAt"], keep="first")  # Same-tick duplicates carry identical cached position/velocity, so any one is representative
    position_data_all = collect_with_gpu(position_data_lf).to_pandas()

    # Process each (GameIndex, RoundIndex) pair independently — each round's own clock
    # restarts near StartedAt=0, so bins must be computed per round, not per game.
    for game_idx, round_idx in zip(match_durations['GameIndex'], match_durations['RoundIndex']):
        match_dur = match_durations.filter(
            (pl.col('GameIndex') == game_idx) & (pl.col('RoundIndex') == round_idx)
        )['match_duration'][0]

        # Use timer config to determine max time bin (cap at timer setting)
        timer_value = config.get('Timer')
        max_time = min(match_dur, timer_value) if timer_value else match_dur

        # Apply skip_initial offset
        start_time = skip_initial
        bins = np.arange(start_time, max_time + time_bin_size, time_bin_size)
        if len(bins) < 2:
            continue

        action_data = action_data_all[
            (action_data_all['GameIndex'] == game_idx) & (action_data_all['RoundIndex'] == round_idx)
        ]
        action_sessions = action_sessions_all[
            (action_sessions_all['GameIndex'] == game_idx) & (action_sessions_all['RoundIndex'] == round_idx)
        ]
        collision_data = collision_data_all[
            (collision_data_all['GameIndex'] == game_idx) & (collision_data_all['RoundIndex'] == round_idx)
        ]
        position_data = position_data_all[
            (position_data_all['GameIndex'] == game_idx) & (position_data_all['RoundIndex'] == round_idx)
        ]

        # Process each timebin
        for i in range(len(bins) - 1):
            time_start = bins[i]
            time_end = bins[i + 1]

            # Calculate collision window start (look back N seconds from current time_end)
            # e.g., if collision_window_n=3 and time_end=5, window=[5-3, 5]=[2, 5]
            # This collects all collisions in the past N seconds up to current time
            collision_window_start = max(skip_initial, time_end - collision_window_n)

            # Filter data for this timebin
            action_bin = action_data[(action_data['StartedAt'] >= time_start) &
                                     (action_data['StartedAt'] < time_end)]
            # Collision uses rolling N-second window ending at current time
            collision_bin = collision_data[(collision_data['UpdatedAt'] >= collision_window_start) &
                                           (collision_data['UpdatedAt'] < time_end)]
            position_bin = position_data[(position_data['StartedAt'] >= time_start) &
                                         (position_data['StartedAt'] < time_end)]
            # Sessions active/overlapping during this bin (not sessions that merely *start* in it)
            sessions_bin = action_sessions[(action_sessions['SessionStart'] < time_end) &
                                           (action_sessions['SessionEnd'] > time_start)]

            # Calculate factors for each bot (Actor 0 = Bot_L, Actor 1 = Bot_R)
            factors = {}

            for actor_id, bot_suffix in [(0, "_L"), (1, "_R")]:
                action_actor = action_bin[action_bin['Actor'] == actor_id]
                collision_actor = collision_bin[collision_bin['Actor'] == actor_id]
                position_actor = position_bin[position_bin['Actor'] == actor_id]
                sessions_actor = sessions_bin[sessions_bin['Actor'] == actor_id]

                # ----- THREAT FACTORS -----

                # 1. CollisionRatio: Hit collisions / Total collisions
                if len(collision_actor) > 0:
                    hit_collisions = len(collision_actor[
                        (collision_actor['ColTieBreaker'] == False) &
                        (collision_actor['ColActor'] == True)
                    ])
                    total_collisions = len(collision_actor)
                    collision_ratio = float(hit_collisions / total_collisions if total_collisions > 0 else np.nan)
                else:
                    collision_ratio = np.nan  # No collision data in this timebin

                # 2. AbilityRatio: (Dash + Skills) / Total actions (session-based, see ActionIntensity)
                if len(sessions_actor) > 0:
                    ability_sessions = len(sessions_actor[sessions_actor['Name'].isin(['Dash', 'SkillBoost', 'SkillStone'])])
                    total_sessions = len(sessions_actor)
                    ability_ratio = float(ability_sessions / total_sessions if total_sessions > 0 else np.nan)
                else:
                    ability_ratio = np.nan  # No action data in this timebin

                # 3. Angle: Average cosine-normalized angle from action data (matching C# api.Angle(normalized: true))
                if len(action_actor) > 0:
                    # Get bot rotation and positions from action data
                    bot_rot = action_actor['BotRot'].values  # Degrees
                    bot_x = action_actor['BotPosX'].values
                    bot_y = action_actor['BotPosY'].values
                    enemy_x = action_actor['EnemyBotPosX'].values
                    enemy_y = action_actor['EnemyBotPosY'].values

                    # Calculate bot's facing direction from rotation
                    # Unity: Quaternion.Euler(0, 0, zRot) * Vector2.up = (-sin(rot), cos(rot))
                    bot_rot_rad = np.deg2rad(bot_rot)
                    facing_x = -np.sin(bot_rot_rad)
                    facing_y = np.cos(bot_rot_rad)

                    # Calculate direction to enemy (normalized)
                    to_enemy_x = enemy_x - bot_x
                    to_enemy_y = enemy_y - bot_y
                    to_enemy_mag = np.sqrt(to_enemy_x**2 + to_enemy_y**2)
                    to_enemy_x = to_enemy_x / (to_enemy_mag + 1e-10)  # Avoid division by zero
                    to_enemy_y = to_enemy_y / (to_enemy_mag + 1e-10)

                    # Calculate cosine of angle using dot product
                    # cos(angle) = dot(facing_dir, to_enemy)
                    cos_angle = facing_x * to_enemy_x + facing_y * to_enemy_y

                    # Clamp to [0, 1] (as in C#: Mathf.Clamp01)
                    cos_angle = np.clip(cos_angle, 0.0, 1.0)

                    # Average across all actions in this timebin
                    avg_angle = float(np.mean(cos_angle[~np.isnan(cos_angle)]) if len(cos_angle) > 0 else np.nan)
                else:
                    avg_angle = np.nan  # No action data in this timebin

                # 4. SafeDistance: Distance from arena edge (normalized)
                if len(position_actor) > 0:
                    bot_x = position_actor['BotPosX'].values
                    bot_y = position_actor['BotPosY'].values

                    # Calculate distance from arena center
                    distance_from_center = np.sqrt((bot_x - arena_center[0])**2 + (bot_y - arena_center[1])**2)
                    # Calculate normalized distance from edge (negative when out of bounds)
                    safe_distances = (arena_radius - distance_from_center) / arena_radius
                    avg_safe_distance = float(np.mean(safe_distances[~np.isnan(safe_distances)]) if len(safe_distances) > 0 else np.nan)
                else:
                    avg_safe_distance = np.nan  # No collision data in this timebin

                # ----- TEMPO FACTORS -----

                # 5. ActionIntensity: sessions' overlap with this bin, in ActInterval-sized
                # decision-tick units rather than a raw session count — a continuous 5s hold
                # at ActInterval=0.1 scores ~50 (one per decision it represents) instead of a
                # flat 1, while a hold resubmitted every simulation frame (see
                # build_action_sessions) still only scores per ActInterval, not per frame.
                if len(sessions_actor) > 0:
                    overlap = np.clip(
                        np.minimum(sessions_actor['SessionEnd'].values, time_end)
                        - np.maximum(sessions_actor['SessionStart'].values, time_start),
                        0, None
                    )
                    ticks = overlap / act_interval
                    action_intensity = float(ticks.sum())
                else:
                    ticks = None
                    action_intensity = np.nan  # No action data in this timebin

                # 6. ActionDensity: Shannon entropy of the tick-weighted active-session Name
                # distribution (same weighting as ActionIntensity, so a Name held longer
                # contributes more probability mass than one merely tapped once)
                if ticks is not None and ticks.sum() > 0:
                    weighted = pd.Series(ticks).groupby(sessions_actor['Name'].values).sum()
                    name_probs = weighted / weighted.sum()
                    entropy = -np.sum(name_probs * np.log2(name_probs + 1e-10))
                    action_density = float(entropy)
                else:
                    action_density = np.nan  # No action data in this timebin

                # 7. BotsDistance: Average distance between bots (normalized, matches C#: dist / (2 * arena_radius))
                if len(position_actor) > 0:
                    bot_x = position_actor['BotPosX'].values
                    bot_y = position_actor['BotPosY'].values
                    enemy_x = position_actor['EnemyBotPosX'].values
                    enemy_y = position_actor['EnemyBotPosY'].values

                    distances = np.sqrt((bot_x - enemy_x)**2 + (bot_y - enemy_y)**2)
                    normalized_distances = distances / (2 * arena_radius)
                    avg_bots_distance = float(np.mean(normalized_distances[~np.isnan(normalized_distances)]) if len(normalized_distances) > 0 else np.nan)
                else:
                    avg_bots_distance = np.nan  # No position data in this timebin

                # 8. Velocity: Average linear velocity
                if len(position_actor) > 0:
                    velocities = position_actor['BotLinv'].values
                    avg_velocity = float(np.mean(velocities[~np.isnan(velocities)]) if len(velocities) > 0 else np.nan)
                else:
                    avg_velocity = np.nan  # No position data in this timebin

                # Normalize factors (matching C# behavior)
                # Each factor is normalized individually before storing
                factors[f'CollisionRatio{bot_suffix}'] = normalize_pacing_factor(collision_ratio, 'CollisionRatio', constraints)
                factors[f'AbilityRatio{bot_suffix}'] = normalize_pacing_factor(ability_ratio, 'AbilityRatio', constraints)
                factors[f'Angle{bot_suffix}'] = normalize_pacing_factor(avg_angle, 'Angle', constraints)
                factors[f'SafeDistance{bot_suffix}'] = normalize_pacing_factor(avg_safe_distance, 'SafeDistance', constraints)
                factors[f'ActionIntensity{bot_suffix}'] = normalize_pacing_factor(action_intensity, 'ActionIntensity', constraints)
                factors[f'ActionDensity{bot_suffix}'] = normalize_pacing_factor(action_density, 'ActionDensity', constraints)
                factors[f'BotsDistance{bot_suffix}'] = normalize_pacing_factor(avg_bots_distance, 'BotsDistance', constraints)
                factors[f'Velocity{bot_suffix}'] = normalize_pacing_factor(avg_velocity, 'Velocity', constraints)

            # Append record
            pacing_fragment_list.append({
                'GameIndex': game_idx,
                'RoundIndex': round_idx,
                'Bot_L': bot_a,
                'Bot_R': bot_b,
                'Timer': config.get('Timer'),
                'ActInterval': config.get('ActInterval'),
                'Round': config.get('Round'),
                'SkillLeft': config.get('SkillLeft'),
                'SkillRight': config.get('SkillRight'),
                'TimeBin': float(time_end),  # Use end of bin instead of start
                **factors
            })

    return pacing_fragment_list


def process_single_csv_lazy(lf, bot_a, bot_b, timer, act_interval, round_val, skill_left, skill_right):
    """
    Process a single CSV file using lazy evaluation
    Implements the same aggregation logic as process_all_games_sql
    Each CSV can contain multiple games (GameIndex), and each game may span multiple
    rounds (RoundIndex) in Best-of-N configs, each with its own StartedAt/UpdatedAt
    clock restarting near 0 (same reasoning as process_pacing_factors_timebins_single_csv).
    """

    # ===== ACTION DATA =====
    # Collapse resubmitted-every-frame holds into sessions (see build_action_sessions) so
    # action counts/durations aren't inflated for matchups that resubmit a held action every
    # simulation frame instead of once per ActInterval (e.g. Bot_LLM_ActionGPT).
    # State==2 (release) rows are kept, not filtered out: build_action_sessions needs
    # their UpdatedAt as the authoritative end-of-hold timestamp.
    action_data_lf = lf.with_row_index("_orig_idx").filter(
        pl.col("Category") == "Action"
    ).select([
        "_orig_idx", "GameIndex", "RoundIndex", "Actor", "StartedAt", "UpdatedAt", "Duration", "Name", "State"
    ])
    action_data = _dedupe_action_ticks(collect_with_gpu(action_data_lf).to_pandas())
    action_sessions = build_action_sessions(action_data)
    action_sessions["SessionDuration"] = action_sessions["SessionEnd"] - action_sessions["SessionStart"]
    # Totals are per GameIndex across all rounds, matching the existing game-level schema.
    action_sessions_lf = pl.from_pandas(
        action_sessions[["GameIndex", "Actor", "Name", "SessionDuration"]]
    ).lazy()

    # Actual durations per game/actor/action: one session's held time, summed
    action_durations = action_sessions_lf.group_by(["GameIndex", "Actor", "Name"]).agg([
        pl.col("SessionDuration").sum().alias("ActualDuration")
    ])

    # Action counts per game/actor/action: one session == one held action, regardless of
    # how many raw ticks it was fragmented across
    action_counts = action_sessions_lf.group_by(["GameIndex", "Actor", "Name"]).agg([
        pl.len().alias("action_count")
    ])

    # Collision counts per game
    collision_data = lf.filter(
        (pl.col("Category") == "Collision") & (pl.col("State").cast(pl.Int32) == 0)
    ).group_by("GameIndex").agg([
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("ColTieBreaker").cast(pl.Int32) == 0) & (pl.col("ColActor").cast(pl.Int32) == 1))
          .then(1).otherwise(0).sum().alias("collision_L"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("ColTieBreaker").cast(pl.Int32) == 0) & (pl.col("ColActor").cast(pl.Int32) == 1))
          .then(1).otherwise(0).sum().alias("collision_R"),
        pl.col("ColTieBreaker").cast(pl.Int32).fill_null(0).sum().alias("collision_tie")
    ])

    # Game metadata: winner per game, and total match duration summed across rounds
    # (each round's UpdatedAt clock restarts near 0, so max() per GameIndex alone would
    # only capture the longest single round, not the full Best-of-N match).
    game_winner = lf.group_by("GameIndex").agg([
        pl.col("GameWinner").first().alias("Winner")
    ])
    game_meta = lf.group_by(["GameIndex", "RoundIndex"]).agg([
        pl.col("UpdatedAt").max().alias("round_duration")
    ]).group_by("GameIndex").agg([
        pl.col("round_duration").sum().alias("MatchDur")
    ]).join(game_winner, on="GameIndex", how="left")

    # Now aggregate durations and counts to game level
    game_durations = action_durations.group_by("GameIndex").agg([
        pl.when(pl.col("Actor").cast(pl.Int32) == 0).then(pl.col("ActualDuration")).sum().fill_null(0).alias("Duration_L"),
        pl.when(pl.col("Actor").cast(pl.Int32) == 1).then(pl.col("ActualDuration")).sum().fill_null(0).alias("Duration_R"),

        # Per-action durations for left
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "Accelerate")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("Accelerate_Dur_L"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "TurnLeft")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("TurnLeft_Dur_L"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "TurnRight")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("TurnRight_Dur_L"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "Dash")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("Dash_Dur_L"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "SkillBoost")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("SkillBoost_Dur_L"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "SkillStone")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("SkillStone_Dur_L"),

        # Per-action durations for right
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "Accelerate")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("Accelerate_Dur_R"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "TurnLeft")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("TurnLeft_Dur_R"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "TurnRight")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("TurnRight_Dur_R"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "Dash")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("Dash_Dur_R"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "SkillBoost")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("SkillBoost_Dur_R"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "SkillStone")).then(pl.col("ActualDuration")).sum().fill_null(0).alias("SkillStone_Dur_R"),
    ])

    # Aggregate action counts to game level
    game_counts = action_counts.group_by("GameIndex").agg([
        pl.when(pl.col("Actor").cast(pl.Int32) == 0).then(pl.col("action_count")).sum().fill_null(0).alias("ActionCounts_L"),
        pl.when(pl.col("Actor").cast(pl.Int32) == 1).then(pl.col("action_count")).sum().fill_null(0).alias("ActionCounts_R"),
        pl.col("action_count").sum().fill_null(0).alias("TotalActions"),

        # Per-action counts for left
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "Accelerate")).then(pl.col("action_count")).sum().fill_null(0).alias("Accelerate_Act_L"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "TurnLeft")).then(pl.col("action_count")).sum().fill_null(0).alias("TurnLeft_Act_L"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "TurnRight")).then(pl.col("action_count")).sum().fill_null(0).alias("TurnRight_Act_L"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "Dash")).then(pl.col("action_count")).sum().fill_null(0).alias("Dash_Act_L"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "SkillBoost")).then(pl.col("action_count")).sum().fill_null(0).alias("SkillBoost_Act_L"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 0) & (pl.col("Name") == "SkillStone")).then(pl.col("action_count")).sum().fill_null(0).alias("SkillStone_Act_L"),

        # Per-action counts for right
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "Accelerate")).then(pl.col("action_count")).sum().fill_null(0).alias("Accelerate_Act_R"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "TurnLeft")).then(pl.col("action_count")).sum().fill_null(0).alias("TurnLeft_Act_R"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "TurnRight")).then(pl.col("action_count")).sum().fill_null(0).alias("TurnRight_Act_R"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "Dash")).then(pl.col("action_count")).sum().fill_null(0).alias("Dash_Act_R"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "SkillBoost")).then(pl.col("action_count")).sum().fill_null(0).alias("SkillBoost_Act_R"),
        pl.when((pl.col("Actor").cast(pl.Int32) == 1) & (pl.col("Name") == "SkillStone")).then(pl.col("action_count")).sum().fill_null(0).alias("SkillStone_Act_R"),
    ])

    # Join everything at game level
    final_metrics = game_meta.join(game_durations, on="GameIndex", how="left") \
                              .join(game_counts, on="GameIndex", how="left") \
                              .join(collision_data, on="GameIndex", how="left")

    # Fill nulls for collisions and add metadata
    final_metrics = final_metrics.with_columns([
        pl.col("collision_L").fill_null(0).alias("Collisions_L"),
        pl.col("collision_R").fill_null(0).alias("Collisions_R"),
        pl.col("collision_tie").fill_null(0).alias("Collisions_Tie"),
        pl.lit(bot_a).alias("Bot_L"),
        pl.lit(bot_b).alias("Bot_R"),
        pl.lit(timer).alias("Timer"),
        pl.lit(act_interval).alias("ActInterval"),
        pl.lit(round_val).alias("Round"),
        pl.lit(skill_left).alias("SkillLeft"),
        pl.lit(skill_right).alias("SkillRight")
    ]).drop(["collision_L", "collision_R", "collision_tie"])

    return final_metrics


def _find_main_log_files(config_path, input_format):
    """
    Find the per-config log file(s) (game_*.json converted via
    compile.log_to_parquet.convert_all_configs) in a config folder, honoring
    input_format ("csv"/"parquet"/"auto" - auto prefers parquet, falling back to
    csv). Excludes "*_pacing.parquet" - a legacy filename from before event rows and
    PacingSegment rows were merged into one file (discriminated by "Category"); any
    leftover file with that suffix predates the merge and has a different schema (no
    "Category"/"Actor"/etc. event columns), so it would break a bare "*.parquet" glob.
    """
    if input_format == "csv":
        return glob.glob(os.path.join(config_path, "*.csv"))
    parquet_files = [
        f for f in glob.glob(os.path.join(config_path, "*.parquet"))
        if not f.endswith("_pacing.parquet")
    ]
    if input_format == "parquet":
        return parquet_files
    # auto - prefer parquet, fallback to csv
    csv_files = glob.glob(os.path.join(config_path, "*.csv"))
    return parquet_files if parquet_files else csv_files


def batch_process_pacing(base_dir, batch_size=50, checkpoint_dir="batched", time_bin_size=None, bot_option="all", input_format="csv", skip_initial=0.0, config_filter=None, collision_window_n=1, name=None):
    """
    Process pacing factors in batches with bot filtering option

    Args:
        base_dir: Base directory containing simulation data
        batch_size: Number of files per batch
        checkpoint_dir: Directory to save checkpoints
        time_bin_size: Dict mapping each time bin size (seconds) to the constraints to
            normalize with at that bin size, e.g.:
                {1: constraints_default, 5: constraints_nn, 10: None}
            A None constraints value means raw (unnormalized) pacing factors for that bin size.
        bot_option: "all" for all bots, or specific bot name to filter (e.g., "BotA")
        input_format: "csv", "parquet", or "auto" to detect both
        skip_initial: Number of seconds to skip at the start to avoid initial bias (default: 0.0)
        config_filter: Dictionary to filter configurations, e.g., {"Timer": [15, 30, 45, 60], "ActInterval": [0.1, 0.2, 0.5],
                "Round": ["BestOf1", "BestOf3", "BestOf5"], "SkillLeft": ["Boost", "Stone"], "SkillRight": ["Boost", "Stone"]}
                If None, no filtering is applied (default: None)
        collision_window_n: Number of seconds to look back for collision calculation (default: 1 second)
                           e.g., at time 5s with N=3, collects collisions from [2s, 5s]
        name: Optional name for organizing output in a subfolder (default: None)
              If provided, output structure: checkpoint_dir/{name}/pacing_factors_{bin_size}/
              If None, output structure: checkpoint_dir/pacing_factors_{bin_size}/
    """
    if not time_bin_size:
        raise ValueError("time_bin_size must be a dict of {bin_size: constraints_or_None}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Find all data files grouped by matchup/config (do this once)
    all_files = []
    matchup_folders = [f for f in os.listdir(base_dir)
                       if os.path.isdir(os.path.join(base_dir, f))]

    for matchup_folder in matchup_folders:
        # Filter by bot if bot_option is not "all"
        if bot_option != "all":
            # Check if the bot is in this matchup (either left or right)
            match = re.match(r"(.+)_vs_(.+)", matchup_folder)
            if not match:
                continue
            bot_a, bot_b = match.groups()
            if bot_option not in [bot_a, bot_b]:
                continue

        matchup_path = os.path.join(base_dir, matchup_folder)
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]

        for config_folder in config_folders:
            config_path = os.path.join(matchup_path, config_folder)
            all_files.extend(_find_main_log_files(config_path, input_format))

    file_type = "Parquet" if input_format == "parquet" else "CSV/Parquet" if input_format == "auto" else "CSV"
    bot_filter_msg = f"for bot '{bot_option}'" if bot_option != "all" else "for all bots"
    print(f"Found {len(all_files)} {file_type} files to process {bot_filter_msg}")

    # Process for each time bin size, using the constraints tied to it
    for current_bin_size, current_constraints in time_bin_size.items():
        print(f"\n{'='*60}")
        print(f"Processing with time_bin_size = {current_bin_size}s"
              f" ({'raw' if current_constraints is None else 'normalized'})")
        print(f"{'='*60}")

        # Create checkpoint dir for this bin size
        # Format: pacing_factors_1, pacing_factors_3, pacing_factors_5, etc.
        bin_size_str = str(current_bin_size).replace('.', '_')  # Handle decimals: 2.5 -> 2_5

        # If name is provided, create a parent folder for it
        if name is not None:
            pacing_factors_dir = os.path.join(checkpoint_dir, name, f"pacing_factors_{bin_size_str}")
        else:
            pacing_factors_dir = os.path.join(checkpoint_dir, f"pacing_factors_{bin_size_str}")

        os.makedirs(pacing_factors_dir, exist_ok=True)

        # Determine which batches are already processed for this bin size
        processed_batches = set()
        for f in os.listdir(pacing_factors_dir):
            match = re.match(r"batch_(\d+)\.csv", f)
            if match:
                processed_batches.add(int(match.group(1)))

        # Process in batches
        total_batches = (len(all_files) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch_num = batch_idx + 1

            # Skip if already processed
            if batch_num in processed_batches:
                print(f"Skipping pacing batch {batch_num}/{total_batches} (already processed)")
                continue

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_files))
            batch_files = all_files[start_idx:end_idx]

            print(f"\nProcessing pacing batch {batch_num}/{total_batches} ({len(batch_files)} files)...")

            # Process only pacing factors
            pacing_fragment_list = []

            for csv_path in batch_files:
                # Extract bot names and config from path
                parts = csv_path.split(os.sep)
                matchup_folder = parts[-3]
                config_folder = parts[-2]

                match = re.match(r"(.+)_vs_(.+)", matchup_folder)
                if not match:
                    continue
                bot_a, bot_b = match.groups()

                # Parse config
                config = parse_config_name_cached(config_folder)

                # Apply config filter if specified
                if config_filter is not None:
                    # Check if config matches all filter criteria
                    skip_file = False
                    for key, allowed_values in config_filter.items():
                        config_value = config.get(key)
                        # Convert to string for comparison if Round or Skill fields
                        if key in ["Round", "SkillLeft", "SkillRight"]:
                            if config_value not in allowed_values:
                                skip_file = True
                                break
                        else:
                            # For numeric fields (Timer, ActInterval), check if value is in allowed list
                            if config_value not in allowed_values:
                                skip_file = True
                                break
                    if skip_file:
                        continue

                # Scan file (CSV or Parquet) with Polars lazy API
                lf = scan_file(csv_path)

                # Process pacing factors with current bin size
                pacing_tb = process_pacing_factors_timebins_single_csv(
                    lf, bot_a, bot_b, config, current_bin_size, skip_initial=skip_initial, constraints=current_constraints, collision_window_n=collision_window_n
                )
                if pacing_tb:
                    pacing_fragment_list.extend(pacing_tb)

            # Save pacing factors batch if computed
            if pacing_fragment_list:
                pacing_factors_df = pl.DataFrame(pacing_fragment_list)
                pacing_path = os.path.join(pacing_factors_dir, f"batch_{batch_num:02d}.csv")
                pacing_factors_df.write_csv(pacing_path)
                print(f"Saved pacing factors batch: {pacing_path} ({len(pacing_factors_df)} records)")


# Fallback calibration constants for logs predating metadata.json's PacingMin/PacingMax
# fields. Confirmed against Assets/Scenes/Battle.unity for the 20260803_235204_batch
# batch specifically - do not assume these hold for other batches; pass min_pacing/
# max_pacing explicitly to batch_process_pacing_segments instead.
DEFAULT_MIN_PACING = 0.0
DEFAULT_MAX_PACING = 0.474


def get_pacing_calibration(metadata: dict, min_pacing: float = None, max_pacing: float = None):
    """
    Resolve the (MinPacing, MaxPacing) constants a battle's raw Tempo/Threat/
    OverallPacing composite scores were bounded by at runtime (e.g. a simulation run
    with MaxPacing=0.6 means an OverallPacing of 0.6 is the ceiling, i.e. should
    normalize to 1.0), used to min-max normalize the Actual columns into [0, 1].

    Precedence: metadata.json's "PacingMin"/"PacingMax" keys (new logs) > the
    min_pacing/max_pacing arguments (needed for logs predating that field) > the
    module defaults (only valid for the specific old batch they were reverse-
    engineered from).
    """
    resolved_min = metadata.get("PacingMin", min_pacing if min_pacing is not None else DEFAULT_MIN_PACING)
    resolved_max = metadata.get("PacingMax", max_pacing if max_pacing is not None else DEFAULT_MAX_PACING)
    return resolved_min, resolved_max


def batch_process_pacing_segments(base_dir, batch_size=10, checkpoint_dir="batched", config_filter=None, name=None, applied_bots=None, min_pacing=None, max_pacing=None):
    """
    Gather the dynamic-pacing PacingSegment rows (Category == "PacingSegment", produced
    per config folder alongside the event rows by
    compile.log_to_parquet.convert_logs_to_parquet) into batched checkpoint files,
    tagged with matchup/config/PacingTarget/PacingConstraint so it can be visualized
    per target - to check whether the action filter's actual pacing (OverallPacing)
    tracks the predefined target curve (TargetOverallPacing) over a round.

    Structure expected: base_dir/BotA_vs_BotB/ConfigFolder/*.parquet (the same combined
    parquet batch_process_csvs reads - PacingSegment rows are filtered out of it here)

    Args:
        base_dir: Root of converted parquet files (the output_root passed to
            convert_all_configs)
        batch_size: Number of config folders per batch
        checkpoint_dir: Directory to save checkpoints
        config_filter: Dict to filter configs, same shape as batch_process_pacing's
            config_filter (Timer/ActInterval/Round/SkillLeft/SkillRight), plus optional
            "PacingTarget" / "PacingConstraint" keys (each a list of allowed values)
        name: Optional name for organizing output in a subfolder (default: None)
              If provided, output structure: checkpoint_dir/{name}/pacing_segments/
        applied_bots: Optional list of bot names that had the dynamic pacing filter
            applied to them (e.g. ["Bot_MCTS", "Bot_NN"]). The filter is tied to bot
            identity, not to a fixed Left/Right slot, so matchup folders are logged in
            both directions (BotA_vs_BotB and BotB_vs_BotA) with the applied bot
            switching sides. When provided, each row is tagged with:
              - "Bot": the actual bot name occupying that row's Side
              - "OpponentBot": the bot on the other side
              - "PacingRole": "Applied" if Bot is in applied_bots, else "Opponent"
            so downstream analysis/plots can group by the applied-bot's perspective
            instead of raw screen Side.
        min_pacing, max_pacing: Fallback calibration constants (see
            get_pacing_calibration) - the (MinPacing, MaxPacing) bounds the actual
            run used, e.g. a run with MaxPacing=0.6 means an OverallPacing of 0.6 was
            the ceiling and should normalize to 1.0. Only used when that config
            folder's own metadata.json doesn't have "PacingMin"/"PacingMax" keys.

    TargetTempo/TargetThreat/TargetOverallPacing are logged raw, bounded by the same
    [MinPacing, MaxPacing] window as the Actual columns (e.g. a "target high" curve
    tops out at MaxPacing, not 1.0) - NOT already-normalized [0, 1] values, so they
    need the same min-max rescale as Actual before the two can be compared. Each row
    is tagged with "PacingMinUsed"/"PacingMaxUsed" plus "ActualTempoScaled"/
    "ActualThreatScaled"/"ActualOverallPacingScaled" AND "TargetTempoScaled"/
    "TargetThreatScaled"/"TargetOverallPacingScaled": both sides min-max normalized
    into [0, 1] via (value - MinPacing) / (MaxPacing - MinPacing), then clipped to
    [0, 1] to guard against values that slip past the calibrated bounds - putting
    the two on the same comparable 0-1 scale instead of leaving Target in raw units
    (which would make a target sitting at/near MinPacing look "capped" near the
    floor rather than correctly rescaling down toward 0).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    pacing_segments_dir = os.path.join(checkpoint_dir, name, "pacing_segments") if name else os.path.join(checkpoint_dir, "pacing_segments")
    os.makedirs(pacing_segments_dir, exist_ok=True)

    # Find all per-config pacing parquet files, grouped by matchup/config folder
    all_config_entries = []
    matchup_folders = [f for f in os.listdir(base_dir)
                       if os.path.isdir(os.path.join(base_dir, f))]

    for matchup_folder in matchup_folders:
        match = re.match(r"(.+)_vs_(.+)", matchup_folder)
        if not match:
            continue
        bot_a, bot_b = match.groups()

        matchup_path = os.path.join(base_dir, matchup_folder)
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]

        for config_folder in config_folders:
            config_path = os.path.join(matchup_path, config_folder)
            main_files = _find_main_log_files(config_path, "parquet")
            if not main_files:
                continue
            all_config_entries.append((bot_a, bot_b, config_folder, main_files[0]))

    print(f"Found {len(all_config_entries)} config folders with pacing segment data")

    # Determine which batches are already processed
    processed_batches = set()
    for f in os.listdir(pacing_segments_dir):
        match = re.match(r"batch_(\d+)\.parquet", f)
        if match:
            processed_batches.add(int(match.group(1)))

    total_batches = (len(all_config_entries) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_num = batch_idx + 1

        if batch_num in processed_batches:
            print(f"Skipping pacing segments batch {batch_num}/{total_batches} (already processed)")
            continue

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_config_entries))
        batch_items = all_config_entries[start_idx:end_idx]

        print(f"\nProcessing pacing segments batch {batch_num}/{total_batches} ({len(batch_items)} configs)...")

        fragment_list = []
        for bot_a, bot_b, config_folder, pacing_path in batch_items:
            config = parse_config_name_cached(config_folder)
            pacing_target, pacing_constraint = parse_pacing_folder_name(config_folder)

            if config_filter is not None:
                skip_file = False
                for key, allowed_values in config_filter.items():
                    if key == "PacingTarget":
                        check_value = pacing_target
                    elif key == "PacingConstraint":
                        check_value = pacing_constraint
                    else:
                        check_value = config.get(key)
                    if check_value not in allowed_values:
                        skip_file = True
                        break
                if skip_file:
                    continue

            # The main per-config parquet holds both event rows and PacingSegment rows
            # (see compile.log_to_parquet.convert_logs_to_parquet) - keep only the latter.
            df = pl.read_parquet(pacing_path).filter(pl.col("Category") == "PacingSegment")
            if df.is_empty():
                continue

            df = df.with_columns([
                pl.lit(bot_a).alias("Bot_L"),
                pl.lit(bot_b).alias("Bot_R"),
                pl.lit(config.get("Timer")).alias("Timer"),
                pl.lit(config.get("ActInterval")).alias("ActInterval"),
                pl.lit(config.get("Round")).alias("Round"),
                pl.lit(config.get("SkillLeft")).alias("SkillLeft"),
                pl.lit(config.get("SkillRight")).alias("SkillRight"),
                pl.lit(pacing_target).alias("PacingTarget"),
                pl.lit(pacing_constraint).alias("PacingConstraint"),
                pl.lit(config_folder).alias("ConfigFolder"),
            ])

            metadata = {}
            metadata_path = os.path.join(os.path.dirname(pacing_path), "metadata.json")
            if os.path.isfile(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            resolved_min_pacing, resolved_max_pacing = get_pacing_calibration(metadata, min_pacing, max_pacing)
            pacing_span = resolved_max_pacing - resolved_min_pacing

            df = df.with_columns([
                pl.lit(resolved_min_pacing).alias("PacingMinUsed"),
                pl.lit(resolved_max_pacing).alias("PacingMaxUsed"),
            ])
            # Both Actual and Target Tempo/Threat/OverallPacing are raw composite
            # scores bounded by [MinPacing, MaxPacing] (e.g. MaxPacing=0.6 means a
            # value of 0.6 is the ceiling) - min-max normalize (rescale) *both* into
            # [0, 1] and clip only to guard against values slipping past the
            # calibrated bounds, so they land on the same comparable 0-1 scale. A
            # target sitting at/near MinPacing must be rescaled toward 0, not left
            # in raw units where it would look "capped" at the floor instead.
            df = df.with_columns([
                ((pl.col("Tempo") - resolved_min_pacing) / pacing_span).clip(0.0, 1.0).alias("ActualTempoScaled"),
                ((pl.col("Threat") - resolved_min_pacing) / pacing_span).clip(0.0, 1.0).alias("ActualThreatScaled"),
                ((pl.col("OverallPacing") - resolved_min_pacing) / pacing_span).clip(0.0, 1.0).alias("ActualOverallPacingScaled"),
                ((pl.col("TargetTempo") - resolved_min_pacing) / pacing_span).clip(0.0, 1.0).alias("TargetTempoScaled"),
                ((pl.col("TargetThreat") - resolved_min_pacing) / pacing_span).clip(0.0, 1.0).alias("TargetThreatScaled"),
                ((pl.col("TargetOverallPacing") - resolved_min_pacing) / pacing_span).clip(0.0, 1.0).alias("TargetOverallPacingScaled"),
            ])

            if applied_bots:
                df = df.with_columns([
                    pl.when(pl.col("Side") == "Left").then(pl.col("Bot_L")).otherwise(pl.col("Bot_R")).alias("Bot"),
                    pl.when(pl.col("Side") == "Left").then(pl.col("Bot_R")).otherwise(pl.col("Bot_L")).alias("OpponentBot"),
                ])
                df = df.with_columns(
                    pl.when(pl.col("Bot").is_in(applied_bots))
                    .then(pl.lit("Applied"))
                    .otherwise(pl.lit("Opponent"))
                    .alias("PacingRole")
                )

            fragment_list.append(df)

        if fragment_list:
            batch_df = pl.concat(fragment_list, how="diagonal_relaxed")
            batch_path = os.path.join(pacing_segments_dir, f"batch_{batch_num:02d}.parquet")
            batch_df.write_parquet(batch_path, compression="zstd", compression_level=3)
            print(f"Saved pacing segments batch: {batch_path} ({len(batch_df)} records)")


def generate_pacing_segments_from_batches(checkpoint_dir, output_dir, name=None):
    """
    Concatenate batched pacing-segment checkpoints (from batch_process_pacing_segments)
    into one combined Parquet file, ready to be grouped by PacingTarget for visualization.

    Args:
        checkpoint_dir: Directory containing batch checkpoints (same as passed to
            batch_process_pacing_segments)
        output_dir: Directory to save the final summary Parquet file
        name: Optional name matching the one used in batch_process_pacing_segments

    Returns:
        The combined Polars DataFrame (or None if no batches were found)
    """
    pacing_segments_dir = os.path.join(checkpoint_dir, name, "pacing_segments") if name else os.path.join(checkpoint_dir, "pacing_segments")
    batch_files = sorted(glob.glob(os.path.join(pacing_segments_dir, "batch_*.parquet")))

    if not batch_files:
        print(f"No pacing segment batches found in {pacing_segments_dir}")
        return None

    print(f"\n📂 Loading {len(batch_files)} pacing segment batch files...")
    lazy_frames = [pl.scan_parquet(f) for f in batch_files]
    combined_df = collect_with_gpu(pl.concat(lazy_frames, how="diagonal_relaxed"))
    print(f"Loaded {len(combined_df):,} pacing segment records")

    pacing_output_dir = os.path.join(output_dir, name) if name else output_dir
    os.makedirs(pacing_output_dir, exist_ok=True)

    output_path = os.path.join(pacing_output_dir, "summary_pacing_segments.parquet")
    combined_df.write_parquet(output_path, compression="zstd", compression_level=3)
    print(f"✅ Saved: {output_path} ({len(combined_df):,} rows)")

    return combined_df


# The 8 Threat/Tempo factors batch_process_pacing computes per (GameIndex,
# RoundIndex, TimeBin, Actor) - see process_pacing_factors_timebins_single_csv.
PACING_FACTOR_NAMES = [
    "CollisionRatio", "AbilityRatio", "Angle", "SafeDistance",
    "ActionIntensity", "ActionDensity", "BotsDistance", "Velocity",
]


def load_bot_pacing_factors(checkpoint_dir, name, target_bot, time_bin_size=1, group_label=None, canonical_bot=None):
    """
    Load batch_process_pacing's per-bot batch CSVs (checkpoint_dir/name/
    pacing_factors_{bin}/batch_*.csv) and fold the _L/_R-suffixed factor columns
    down to target_bot's own side, regardless of which slot (Bot_L or Bot_R) it
    occupied in a given matchup folder - unlike batch_process_pacing_segments's
    PacingSegment rows, these raw factors are recomputed offline from the same
    Action/Collision/position event rows every run logs, so they're the one pacing
    metric that exists whether or not the dynamic pacing filter was active - useful
    for comparing a filter-steered ("Applied") run against an unfiltered baseline
    run of the same bot, which never emits PacingSegment rows at all.

    Args:
        checkpoint_dir, name: Same checkpoint_dir/name passed to batch_process_pacing
        target_bot: Bot name to extract (must match the value batch_process_pacing's
            bot_option filtered on, e.g. "MCTS" or "Bot_MCTS")
        time_bin_size: Bin size used when calling batch_process_pacing (selects which
            pacing_factors_{bin} subfolder to read)
        group_label: Optional value stamped into a "Group" column (e.g. "Filtered")
        canonical_bot: Optional display name to stamp into "Bot" instead of
            target_bot, so differently-prefixed runs of the same bot (e.g. "MCTS" vs
            "Bot_MCTS") can be aligned under one name

    Returns:
        Polars DataFrame with columns GameIndex, RoundIndex, Timer, ActInterval,
        Round, SkillLeft, SkillRight, TimeBin, OpponentBot, Bot, Group, plus one
        column per PACING_FACTOR_NAMES entry - or None if no batch files were found.
    """
    bin_size_str = str(time_bin_size).replace(".", "_")
    pattern = os.path.join(checkpoint_dir, name, f"pacing_factors_{bin_size_str}", "batch_*.csv")
    batch_files = sorted(glob.glob(pattern))
    if not batch_files:
        print(f"No pacing factor batches found matching {pattern}")
        return None

    df = pl.concat([pl.scan_csv(f) for f in batch_files], how="diagonal_relaxed").collect()

    is_left = pl.col("Bot_L") == target_bot
    factor_exprs = [
        pl.when(is_left).then(pl.col(f"{factor}_L")).otherwise(pl.col(f"{factor}_R")).alias(factor)
        for factor in PACING_FACTOR_NAMES
    ]
    return df.select([
        "GameIndex", "RoundIndex", "Timer", "ActInterval", "Round", "SkillLeft", "SkillRight", "TimeBin",
        pl.when(is_left).then(pl.col("Bot_R")).otherwise(pl.col("Bot_L")).alias("OpponentBot"),
        *factor_exprs,
    ]).with_columns([
        pl.lit(canonical_bot or target_bot).alias("Bot"),
        pl.lit(group_label).alias("Group"),
    ])


def load_filtered_target_tracking(pacing_segments_dir, bots, config_filter=None):
    """
    Load the engine's own PacingSegment rows (batch_process_pacing_segments's
    checkpoint output) for the given applied bots and convert each row's
    engine-native LocalSegmentIndex into elapsed match seconds, binned to the
    nearest 1s TimeBin - the same axis load_bot_pacing_factors uses - so the
    engine's real Target/Actual pacing curves can be overlaid on the same time axis
    as the offline-recomputed factor charts.

    PacingSegment rows aren't logged on fixed 1s boundaries - how many segments a
    round gets (NumSegmentsInRound) depends on how long that round actually ran, so
    LocalSegmentIndex alone isn't comparable across rounds of different length. Each
    segment's elapsed time is instead estimated as the midpoint of its slice of the
    round: (LocalSegmentIndex + 0.5) * (RoundDuration / NumSegmentsInRound).

    Args:
        pacing_segments_dir: Dir containing batch_*.parquet (checkpoint_dir/[name/]
            pacing_segments from batch_process_pacing_segments)
        bots: Applied bot names to keep (e.g. ["MCTS", "NN"]) - rows are further
            restricted to PacingRole == "Applied" (see batch_process_pacing_segments)
            since only the applied bot's own Actual/Target values are meaningful
        config_filter: Optional dict of {column: [allowed values]}, same shape as
            batch_process_pacing_segments's config_filter (e.g. Timer/ActInterval/
            Round/SkillLeft/SkillRight)

    Returns:
        Polars DataFrame with columns Bot, PacingTarget, TimeBin, LocalSegmentIndex,
        Timer, ActualTempoScaled, ActualThreatScaled, ActualOverallPacingScaled,
        TargetTempoScaled, TargetThreatScaled, TargetOverallPacingScaled, ConfigFolder,
        GameIndex, Won - one row per original segment (not yet aggregated by TimeBin -
        aggregate downstream as needed), or None if no batch files were found.
        LocalSegmentIndex is the engine's raw per-round tick index (0-based) - the
        *Targets arrays in Resources/.../Sim_Targets/<subfolder>/<PacingTarget>.json
        are indexed by this directly (see
        plotting.pacing_filter_comparison.load_pacing_target_curves), spread evenly
        across [0, Timer] rather than TimeBin's per-round real-time estimate.
        GameIndex is only unique *within* one ConfigFolder (each config folder's own
        game_*.json files number from 1) - group by (ConfigFolder, GameIndex)
        together, never GameIndex alone, to identify one real game (see
        plotting.pacing_filter_comparison.compute_game_level_tracking_error). Won is
        whether Bot's side (Side vs GameWinner, where GameWinner is 0=Left/1=Right/
        2=Draw) won that whole game - constant across every segment row belonging to
        the same (ConfigFolder, GameIndex).
    """
    batch_files = sorted(glob.glob(os.path.join(pacing_segments_dir, "batch_*.parquet")))
    if not batch_files:
        print(f"No pacing segment batches found in {pacing_segments_dir}")
        return None

    df = pl.concat([pl.scan_parquet(f) for f in batch_files], how="diagonal_relaxed").collect()
    df = df.filter((pl.col("PacingRole") == "Applied") & pl.col("Bot").is_in(bots))

    if config_filter:
        for key, allowed in config_filter.items():
            df = df.filter(pl.col(key).is_in(allowed))

    df = df.filter(
        (pl.col("NumSegmentsInRound") > 0) & (pl.col("RoundDuration") > 0)
    ).with_columns(
        (
            (pl.col("LocalSegmentIndex") + 0.5)
            * (pl.col("RoundDuration") / pl.col("NumSegmentsInRound"))
        ).ceil().clip(lower_bound=1).alias("TimeBin"),
        (
            ((pl.col("Side") == "Left") & (pl.col("GameWinner") == 0))
            | ((pl.col("Side") == "Right") & (pl.col("GameWinner") == 1))
        ).alias("Won"),
    )

    return df.select([
        "Bot", "PacingTarget", "TimeBin", "LocalSegmentIndex", "Timer",
        "ActualTempoScaled", "ActualThreatScaled", "ActualOverallPacingScaled",
        "TargetTempoScaled", "TargetThreatScaled", "TargetOverallPacingScaled",
        "ConfigFolder", "GameIndex", "Won",
    ])


def batch_process_csvs(base_dir, batch_size=50, checkpoint_dir="batched", time_bin_size=None, compute_timebins=False, compute_pacing=False, input_format="csv"):
    """
    Process CSVs or Parquet files in batches and save checkpoints
    Similar to generator.py batch() function
    Structure: base_dir/BotA_vs_BotB/ConfigFolder/*.csv or *.parquet

    Args:
        base_dir: Base directory containing simulation data
        batch_size: Number of files per batch
        checkpoint_dir: Directory to save checkpoints
        time_bin_size: Size of time bins (only used if compute_timebins=True or compute_pacing=True)
        compute_timebins: Whether to compute time-binned data
        compute_pacing: Whether to compute pacing factors
        input_format: "csv", "parquet", or "auto" to detect both
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create separate checkpoint dirs for timebins if needed
    if compute_timebins:
        action_timebin_dir = os.path.join(checkpoint_dir, "action_timebins")
        collision_timebin_dir = os.path.join(checkpoint_dir, "collision_timebins")
        os.makedirs(action_timebin_dir, exist_ok=True)
        os.makedirs(collision_timebin_dir, exist_ok=True)

    # Create checkpoint dir for pacing factors if needed
    if compute_pacing:
        pacing_factors_dir = os.path.join(checkpoint_dir, "pacing_factors")
        os.makedirs(pacing_factors_dir, exist_ok=True)

    # Find all data files grouped by matchup/config
    all_files = []
    matchup_folders = [f for f in os.listdir(base_dir)
                       if os.path.isdir(os.path.join(base_dir, f))]

    for matchup_folder in matchup_folders:
        matchup_path = os.path.join(base_dir, matchup_folder)
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]

        for config_folder in config_folders:
            config_path = os.path.join(matchup_path, config_folder)
            all_files.extend(_find_main_log_files(config_path, input_format))

    file_type = "Parquet" if input_format == "parquet" else "CSV/Parquet" if input_format == "auto" else "CSV"
    print(f"Found {len(all_files)} {file_type} files to process")

    # Determine which batches are already processed
    processed_batches = set()
    for f in os.listdir(checkpoint_dir):
        match = re.match(r"batch_(\d+)\.csv", f)
        if match:
            processed_batches.add(int(match.group(1)))

    # Process in batches
    total_batches = (len(all_files) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_num = batch_idx + 1

        # Skip if already processed
        if batch_num in processed_batches:
            print(f"Skipping batch {batch_num}/{total_batches} (already processed)")
            continue

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_files))
        batch_files = all_files[start_idx:end_idx]

        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_files)} files)...")

        batch_df, action_timebin_df, collision_timebin_df, pacing_factors_df = process_batch_csvs(
            batch_files, checkpoint_dir, time_bin_size=time_bin_size, compute_timebins=compute_timebins, compute_pacing=compute_pacing
        )

        # Save game metrics batch
        if batch_df is not None:
            batch_path = os.path.join(checkpoint_dir, f"batch_{batch_num:02d}.csv")
            batch_df.write_csv(batch_path)
            print(f"Saved batch checkpoint: {batch_path}")

        # Save timebin batches if computed
        if compute_timebins:
            if action_timebin_df is not None:
                action_path = os.path.join(action_timebin_dir, f"batch_{batch_num:02d}.csv")
                action_timebin_df.write_csv(action_path)
                print(f"Saved action timebin batch: {action_path}")

            if collision_timebin_df is not None:
                collision_path = os.path.join(collision_timebin_dir, f"batch_{batch_num:02d}.csv")
                collision_timebin_df.write_csv(collision_path)
                print(f"Saved collision timebin batch: {collision_path}")

        # Save pacing factors batch if computed
        if compute_pacing:
            if pacing_factors_df is not None:
                pacing_path = os.path.join(pacing_factors_dir, f"batch_{batch_num:02d}.csv")
                pacing_factors_df.write_csv(pacing_path)
                print(f"Saved pacing factors batch: {pacing_path}")


def create_summary_matchup(all_games, output_dir):
    """Create matchup summary using Polars with GPU acceleration"""
    group_cols = ["Bot_L", "Bot_R", "Timer", "ActInterval", "Round", "SkillLeft", "SkillRight"]

    # Find all action-specific columns
    action_cols = [col for col in all_games.columns if any(col.endswith(suffix) for suffix in ("_Act_L", "_Act_R", "_Dur_L", "_Dur_R"))]

    # Build aggregation list
    agg_list = [
        pl.col("GameIndex").n_unique().alias("Games"),
        (pl.col("Winner") == 0).sum().alias("Winner_L"),
        (pl.col("Winner") == 1).sum().alias("Winner_R"),
        pl.col("ActionCounts_L").sum(),
        pl.col("ActionCounts_R").sum(),
        pl.col("TotalActions").sum(),
        pl.col("Duration_L").sum(),
        pl.col("Duration_R").sum(),
        pl.col("Collisions_L").sum(),
        pl.col("Collisions_R").sum(),
        pl.col("Collisions_Tie").sum(),
        pl.col("MatchDur").mean(),
    ]

    # Add all action-specific columns
    for col in action_cols:
        agg_list.append(pl.col(col).sum())

    # Use lazy frames for GPU acceleration
    matchup_summary_lazy = all_games.lazy().group_by(group_cols).agg(agg_list)

    # Add win rates
    matchup_summary_lazy = matchup_summary_lazy.with_columns([
        (pl.col("Winner_L") / pl.col("Games")).alias("WinRate_L"),
        (pl.col("Winner_R") / pl.col("Games")).alias("WinRate_R")
    ])

    matchup_summary = collect_with_gpu(matchup_summary_lazy)

    # Compute bot rankings based on overall performance
    # Aggregate left bots
    bot_summary_L_lazy = matchup_summary.lazy().group_by("Bot_L").agg([
        pl.col("Games").sum().alias("TotalGames"),
        pl.col("Winner_L").sum().alias("TotalWins"),
    ]).rename({"Bot_L": "Bot"})

    # Aggregate right bots
    bot_summary_R_lazy = matchup_summary.lazy().group_by("Bot_R").agg([
        pl.col("Games").sum().alias("TotalGames"),
        pl.col("Winner_R").sum().alias("TotalWins"),
    ]).rename({"Bot_R": "Bot"})

    # Combine and compute ranks
    bot_ranks_lazy = pl.concat([bot_summary_L_lazy, bot_summary_R_lazy]).group_by("Bot").agg([
        pl.col("TotalGames").sum(),
        pl.col("TotalWins").sum(),
    ]).with_columns([
        (pl.col("TotalWins") / pl.col("TotalGames")).alias("WinRate")
    ]).with_columns([
        pl.col("WinRate").rank(descending=True).cast(pl.Int32).alias("Rank")
    ]).select(["Bot", "Rank"])

    bot_ranks = collect_with_gpu(bot_ranks_lazy)

    # Join ranks back to matchup summary
    matchup_summary_lazy = matchup_summary.lazy().join(
        bot_ranks.lazy().rename({"Bot": "Bot_L", "Rank": "Rank_L"}),
        on="Bot_L",
        how="left"
    ).join(
        bot_ranks.lazy().rename({"Bot": "Bot_R", "Rank": "Rank_R"}),
        on="Bot_R",
        how="left"
    ).sort(["Bot_L", "Bot_R", "Timer", "ActInterval"])

    matchup_summary = collect_with_gpu(matchup_summary_lazy)

    # Save to CSV
    matchup_summary.write_csv(f"{output_dir}/summary_matchup.csv")
    print(f"Saved {output_dir}/summary_matchup.csv")

    return matchup_summary


def create_summary_bot(matchup_summary, output_dir):
    """Create bot summary using Polars with GPU acceleration"""

    # Use lazy frames for GPU acceleration
    # First, normalize the data so each row represents one bot in one game
    bot_summary_L_lazy = matchup_summary.lazy().select([
        pl.col("Bot_L").alias("Bot"),
        pl.col("Games"),
        pl.col("Winner_L").alias("Wins"),
        pl.col("Duration_L").alias("Duration"),
        pl.col("ActionCounts_L").alias("TotalActions"),
        pl.col("Collisions_L").alias("Collisions_Own"),
        pl.col("Collisions_Tie"),
    ])

    bot_summary_R_lazy = matchup_summary.lazy().select([
        pl.col("Bot_R").alias("Bot"),
        pl.col("Games"),
        pl.col("Winner_R").alias("Wins"),
        pl.col("Duration_R").alias("Duration"),
        pl.col("ActionCounts_R").alias("TotalActions"),
        pl.col("Collisions_R").alias("Collisions_Own"),
        pl.col("Collisions_Tie"),
    ])

    # Combine and calculate per-game averages, then aggregate by bot
    bot_summary_lazy = pl.concat([bot_summary_L_lazy, bot_summary_R_lazy]).with_columns([
        # Calculate per-game averages
        (pl.col("Duration") / pl.col("Games")).alias("Duration_per_game"),
        (pl.col("TotalActions") / pl.col("Games")).alias("Actions_per_game"),
        ((pl.col("Collisions_Own") + pl.col("Collisions_Tie")) / pl.col("Games")).alias("Collisions_per_game"),
        (pl.col("Wins") / pl.col("Games")).alias("WinRate_per_matchup"),
    ]).group_by("Bot").agg([
        pl.col("Games").sum().alias("TotalGames"),
        pl.col("Wins").sum().alias("TotalWins"),
        pl.col("WinRate_per_matchup").mean().alias("WinRate_mean"),
        pl.col("WinRate_per_matchup").std().alias("WinRate_std"),
        pl.col("Duration_per_game").mean().alias("Duration_mean"),
        pl.col("Duration_per_game").std().alias("Duration_std"),
        pl.col("Actions_per_game").mean().alias("Actions_mean"),
        pl.col("Actions_per_game").std().alias("Actions_std"),
        pl.col("Collisions_per_game").mean().alias("Collisions_mean"),
        pl.col("Collisions_per_game").std().alias("Collisions_std"),
    ]).with_columns([
        # Format as "mean (std)" with 2 decimal places
        (pl.col("WinRate_mean").round(2).cast(pl.Utf8) + " (" + pl.col("WinRate_std").round(2).cast(pl.Utf8) + ")").alias("Win-rate"),
        (pl.col("Duration_mean").round(2).cast(pl.Utf8) + " (" + pl.col("Duration_std").round(2).cast(pl.Utf8) + ")").alias("Action Duration"),
        (pl.col("Actions_mean").round(2).cast(pl.Utf8) + " (" + pl.col("Actions_std").round(2).cast(pl.Utf8) + ")").alias("Actions"),
        (pl.col("Collisions_mean").round(2).cast(pl.Utf8) + " (" + pl.col("Collisions_std").round(2).cast(pl.Utf8) + ")").alias("Collisions"),
    ]).with_columns([
        pl.col("WinRate_mean").rank(descending=True).cast(pl.Int32).alias("Rank"),
    ]).select([
        "Rank",
        "Bot",
        "Win-rate",
        "Action Duration",
        "Actions",
        "Collisions"
    ]).sort("Rank")

    bot_summary = collect_with_gpu(bot_summary_lazy)

    # Save
    bot_summary.write_csv(f"{output_dir}/summary_bot.csv")
    print(f"Saved {output_dir}/summary_bot.csv")

    return bot_summary


def generate_timebins_from_batches(checkpoint_dir, output_dir, pacing_ratio_percentile=5, pacing_bin_size=None, name=None):
    """
    Generate timebin summaries from batched timebin checkpoints
    Loads batch files and creates final summaries

    Args:
        checkpoint_dir: Directory containing batch checkpoints
        output_dir: Directory to save final summaries
        pacing_ratio_percentile: Percentile for pacing ratio calculation
        pacing_bin_size: Time bin size for pacing factors (e.g., 5 or "2_5" for 2.5s).
                        If None, uses default "pacing_factors" folder.
                        Can also be a list/array to process multiple bin sizes.
        name: Optional name for organizing pacing outputs in a subfolder (default: None)
              If provided, reads from checkpoint_dir/{name}/pacing_factors_{bin_size}/
              and writes to output_dir/{name}/
    """
    print("=" * 60)
    print("🚀 Generating timebin summaries from batches")
    print("=" * 60)

    # Load action timebin batches
    action_batch_files = sorted(glob.glob(f"{checkpoint_dir}/action_timebins/batch_*.csv"))
    if action_batch_files:
        print(f"\n📂 Loading {len(action_batch_files)} action timebin batch files...")
        action_lazy_frames = [scan_file(f) for f in action_batch_files]
        action_timebin_df = collect_with_gpu(pl.concat(action_lazy_frames))
        print(f"Loaded {len(action_timebin_df):,} action timebin records")

        print("\n Creating action time-bin summary...")
        summarize_action_timebins(action_timebin_df, output_dir)

    # Load collision timebin batches
    collision_batch_files = sorted(glob.glob(f"{checkpoint_dir}/collision_timebins/batch_*.csv"))
    if collision_batch_files:
        print(f"\n📂 Loading {len(collision_batch_files)} collision timebin batch files...")
        collision_lazy_frames = [scan_file(f) for f in collision_batch_files]
        collision_timebin_df = collect_with_gpu(pl.concat(collision_lazy_frames))
        print(f"Loaded {len(collision_timebin_df):,} collision timebin records")

        print("\n Creating collision time-bin summary...")
        summarize_collision_timebins(collision_timebin_df, output_dir)

    # Process pacing factors (support multiple bin sizes)
    if pacing_bin_size is not None:
        # Convert to list if single value
        if isinstance(pacing_bin_size, (int, float, str)):
            bin_sizes = [pacing_bin_size]
        else:
            bin_sizes = list(pacing_bin_size)
    else:
        # Determine base path for pacing folders (with or without name)
        base_pacing_path = os.path.join(checkpoint_dir, name) if name else checkpoint_dir

        # Try default folder first, then auto-detect folders
        default_folder = os.path.join(base_pacing_path, "pacing_factors")
        if os.path.exists(default_folder):
            bin_sizes = [None]  # Use default "pacing_factors" folder
        else:
            # Auto-detect pacing_factors_* folders
            if os.path.exists(base_pacing_path):
                pacing_dirs = [d for d in os.listdir(base_pacing_path) if d.startswith("pacing_factors_")]
                if pacing_dirs:
                    # Extract bin sizes from folder names
                    bin_sizes = [d.replace("pacing_factors_", "") for d in pacing_dirs]
                    print(f"\n📂 Auto-detected pacing bin sizes: {bin_sizes}")
                else:
                    bin_sizes = []
            else:
                bin_sizes = []

    # Load and process each bin size
    for bin_size in bin_sizes:
        if bin_size is None:
            pacing_folder = "pacing_factors"
            output_suffix = ""
        else:
            # Convert numeric bin sizes to string format
            bin_size_str = str(bin_size).replace('.', '_')
            pacing_folder = f"pacing_factors_{bin_size_str}"
            output_suffix = f"_{bin_size_str}"

        # Construct full path including name if provided
        if name:
            pacing_batch_path = os.path.join(checkpoint_dir, name, pacing_folder)
        else:
            pacing_batch_path = os.path.join(checkpoint_dir, pacing_folder)

        pacing_batch_files = sorted(glob.glob(f"{pacing_batch_path}/batch_*.csv"))

        if pacing_batch_files:
            print(f"\n📂 Loading {len(pacing_batch_files)} pacing factors batch files (bin_size={bin_size or 'default'})...")

            # Define the expected schema with all numeric fields as Float64
            schema_overrides = {
                'CollisionRatio_L': pl.Float64,
                'AbilityRatio_L': pl.Float64,
                'Angle_L': pl.Float64,
                'SafeDistance_L': pl.Float64,
                'ActionIntensity_L': pl.Float64,
                'ActionDensity_L': pl.Float64,
                'BotsDistance_L': pl.Float64,
                'Velocity_L': pl.Float64,
                'CollisionRatio_R': pl.Float64,
                'AbilityRatio_R': pl.Float64,
                'Angle_R': pl.Float64,
                'SafeDistance_R': pl.Float64,
                'ActionIntensity_R': pl.Float64,
                'ActionDensity_R': pl.Float64,
                'BotsDistance_R': pl.Float64,
                'Velocity_R': pl.Float64,
            }

            # Load each file with schema alignment
            pacing_dfs = []
            for f in pacing_batch_files:
                df = pl.read_csv(f)
                # Cast all pacing factor columns to Float64 to ensure consistent schema
                for col, dtype in schema_overrides.items():
                    if col in df.columns:
                        df = df.with_columns(pl.col(col).cast(dtype))
                pacing_dfs.append(df)

            pacing_factors_df = pl.concat(pacing_dfs)
            print(f"Loaded {len(pacing_factors_df):,} pacing factor records")

            # Create output directory for pacing (with name subfolder if provided)
            pacing_output_dir = os.path.join(output_dir, name) if name else output_dir
            os.makedirs(pacing_output_dir, exist_ok=True)

            print(f"\n Creating pacing factors summary (bin_size={bin_size or 'default'})...")
            summarize_pacing_factors(pacing_factors_df, pacing_output_dir, pacing_ratio_percentile, output_suffix)

    print("\n" + "=" * 60)
    print("🎉 Done! Created:")
    if action_batch_files:
        print("   - summary_action_timebins.csv")
    if collision_batch_files:
        print("   - summary_collision_timebins.csv")
    if bin_sizes:
        pacing_path_prefix = f"{name}/" if name else ""
        for bin_size in bin_sizes:
            suffix = "" if bin_size is None else f"_{str(bin_size).replace('.', '_')}"
            print(f"   - {pacing_path_prefix}summary_pacing_per_bot{suffix}.csv")
    print("=" * 60)


def summarize_action_timebins(time_fragment_df, output_dir):
    """
    Summarize action time fragment data with GPU acceleration.
    Computes mean counts per bot/config/timebin/action.
    """
    print(" Summarizing action time-binned data...")

    # Use lazy frames for GPU acceleration
    summary_lazy = time_fragment_df.lazy().group_by(
        ['Bot', 'Timer', 'ActInterval', 'Round', 'TimeBin', 'Action']
    ).agg([
        pl.col('Count').mean().alias('MeanCount')
    ]).sort(['Bot', 'Timer', 'ActInterval', 'Round', 'TimeBin', 'Action'])

    summary = collect_with_gpu(summary_lazy)

    # Save CSV
    summary.write_csv(f"{output_dir}/summary_action_timebins.csv")
    print(f"Saved {output_dir}/summary_action_timebins.csv")

    return summary


def summarize_collision_timebins(collision_fragment_df, output_dir):
    """
    Calculate collision time fragment data with GPU acceleration.
    Aggregates Actor, Target, Tie counts per config/timebin.
    """
    print(" Creating collision detail time-binned data...")

    # Use lazy frames for GPU acceleration
    summary_lazy = collision_fragment_df.lazy().group_by(
        ['Bot_L', 'Bot_R', 'Timer', 'ActInterval', 'Round', 'TimeBin']
    ).agg([
        pl.col('Actor_L').sum().alias('Actor_L'),
        pl.col('Actor_R').sum().alias('Actor_R'),
        pl.col('Tie').sum().alias('Tie'),
    ]).sort(['Bot_L', 'Bot_R', 'Timer', 'ActInterval', 'Round', 'TimeBin'])

    summary = collect_with_gpu(summary_lazy)

    # Save CSV
    summary.write_csv(f"{output_dir}/summary_collision_timebins.csv")
    print(f"Saved {output_dir}/summary_collision_timebins.csv")

    return summary


def summarize_pacing_factors(pacing_factors_df, output_dir, ratio_percentile=5, output_suffix=""):
    """
    Summarize pacing factors with GPU acceleration.
    Computes mean and std for each factor per bot/config/timebin.
    Also provides per-bot statistics across all matchups.

    Args:
        pacing_factors_df: DataFrame with pacing factor data
        output_dir: Output directory
        ratio_percentile: Percentile for pacing ratio calculation
        output_suffix: Suffix to append to output filenames (e.g., "_5" for 5s bin size)
    """
    print(" Creating pacing factors summary...")

    # Use lazy frames for GPU acceleration
    # Aggregate per matchup
    summary_lazy = pacing_factors_df.lazy().group_by(
        ['Bot_L', 'Bot_R', 'Timer', 'ActInterval', 'Round', 'SkillLeft', 'SkillRight', 'TimeBin']
    ).agg([
        # Threat factors
        pl.col('CollisionRatio_L').mean().alias('CollisionRatio_L_mean'),
        pl.col('CollisionRatio_R').mean().alias('CollisionRatio_R_mean'),
        pl.col('AbilityRatio_L').mean().alias('AbilityRatio_L_mean'),
        pl.col('AbilityRatio_R').mean().alias('AbilityRatio_R_mean'),
        pl.col('Angle_L').mean().alias('Angle_L_mean'),
        pl.col('Angle_R').mean().alias('Angle_R_mean'),
        pl.col('SafeDistance_L').mean().alias('SafeDistance_L_mean'),
        pl.col('SafeDistance_R').mean().alias('SafeDistance_R_mean'),
        # Tempo factors
        pl.col('ActionIntensity_L').mean().alias('ActionIntensity_L_mean'),
        pl.col('ActionIntensity_R').mean().alias('ActionIntensity_R_mean'),
        pl.col('ActionDensity_L').mean().alias('ActionDensity_L_mean'),
        pl.col('ActionDensity_R').mean().alias('ActionDensity_R_mean'),
        pl.col('BotsDistance_L').mean().alias('BotsDistance_L_mean'),
        pl.col('BotsDistance_R').mean().alias('BotsDistance_R_mean'),
        pl.col('Velocity_L').mean().alias('Velocity_L_mean'),
        pl.col('Velocity_R').mean().alias('Velocity_R_mean'),
    ]).sort(['Bot_L', 'Bot_R', 'Timer', 'ActInterval', 'Round', 'TimeBin'])

    summary = collect_with_gpu(summary_lazy)

    # Save matchup-level summary
    summary.write_csv(f"{output_dir}/summary_pacing_factors{output_suffix}.csv")
    print(f"Saved {output_dir}/summary_pacing_factors{output_suffix}.csv")

    # Create per-bot statistics (min, max, mean, std for constraint setting)
    print(" Creating per-bot per-timebin pacing statistics for constraint setting...")

    # Transform to per-bot format
    bot_stats_list = []

    for bot_side, suffix in [('Bot_L', '_L'), ('Bot_R', '_R')]:
        bot_data = pacing_factors_df.lazy().select([
            pl.col(bot_side).alias('Bot'),
            pl.col('TimeBin'),  # Include TimeBin for per-segment stats
            pl.col('Timer'),
            pl.col('ActInterval'),
            pl.col('Round'),
            pl.col(f'CollisionRatio{suffix}').alias('CollisionRatio'),
            pl.col(f'AbilityRatio{suffix}').alias('AbilityRatio'),
            pl.col(f'Angle{suffix}').alias('Angle'),
            pl.col(f'SafeDistance{suffix}').alias('SafeDistance'),
            pl.col(f'ActionIntensity{suffix}').alias('ActionIntensity'),
            pl.col(f'ActionDensity{suffix}').alias('ActionDensity'),
            pl.col(f'BotsDistance{suffix}').alias('BotsDistance'),
            pl.col(f'Velocity{suffix}').alias('Velocity'),
        ])
        bot_stats_list.append(bot_data)

    # Combine both sides
    all_bot_data = pl.concat(bot_stats_list)

    # Compute statistics per bot per timebin per timer configuration
    # For some factors, use average of lowest 5% (excluding zero) for min
    factors = ['CollisionRatio', 'AbilityRatio', 'ActionIntensity', 'ActionDensity', 'Angle', 'SafeDistance', 'BotsDistance', 'Velocity']

    agg_exprs = []

    # Factors that use 5% lowest average for min and 5% highest average for max
    for factor in factors:
        filtered = pl.col(factor).filter(pl.col(factor).is_not_nan())
        agg_exprs.extend([
            filtered.sort().head(pl.len() // (100 // ratio_percentile) + 1).mean().alias(f'{factor}_min'),
            filtered.sort(descending=True).head(pl.len() // (100 // ratio_percentile) + 1).mean().alias(f'{factor}_max'),
            filtered.mean().alias(f'{factor}_mean'),
            filtered.std().alias(f'{factor}_std'),
        ])

    bot_stats_lazy = all_bot_data.group_by(['Bot', 'Timer', 'TimeBin']).agg(agg_exprs).sort(['Bot', 'Timer', 'TimeBin'])

    bot_stats = collect_with_gpu(bot_stats_lazy)

    # Save per-bot per-timebin statistics
    bot_stats.write_csv(f"{output_dir}/summary_pacing_per_bot{output_suffix}.csv")
    print(f"Saved {output_dir}/summary_pacing_per_bot{output_suffix}.csv")
    print(f"  Use this file to set constraint ranges (min, max) for each bot per timebin segment")

    return summary, bot_stats




def generate(checkpoint_dir, output_dir):
    """
    Generate summary files from batched checkpoints
    Similar to generator.py generate() function

    Args:
        time_bin_size: Size of time bins for time-series analysis (optional)
        base_dir: Base directory for CSV files (only needed if computing time bins)
    """
    print("=" * 60)
    print("🚀 Polars GPU: Generating summaries from batches")
    print("=" * 60)

    # Load all batch files
    batch_files = sorted(glob.glob(f"{checkpoint_dir}/batch_*.csv"))

    if not batch_files:
        print(f"❌ No batch files found in '{checkpoint_dir}/' directory")
        return None, None

    print(f"\n📂 Loading {len(batch_files)} batch files...")

    # Scan and concatenate all batches using lazy API
    lazy_frames = [scan_file(f) for f in batch_files]
    all_games_lazy = pl.concat(lazy_frames)

    print("🔄 Collecting all games...")
    all_games = collect_with_gpu(all_games_lazy)

    print(f"Loaded {len(all_games):,} games")

    # Create summaries with Polars
    print("\n Creating matchup summary...")
    matchup_summary = create_summary_matchup(all_games, output_dir)

    print("\n Creating bot summary...")
    bot_summary = create_summary_bot(matchup_summary, output_dir)

    print("\n" + "=" * 60)
    print("🎉 Done! Created:")
    print("   - summary_matchup.csv")
    print("   - summary_bot.csv")
    print("=" * 60)

    return matchup_summary, bot_summary


if __name__ == "__main__":
    import sys

    base_dir = "/Users/def/Documents/Simulation"
    checkpoint_dir = "batched"
    output_dir = "result"
    timebin_size = 5
    batch_size = 2

    if len(sys.argv) > 1:
        command = sys.argv[1]

        start = time.time()
        is_valid_process = True
        if command == "batch":
            # Batch processing mode - only game metrics
            batch_process_csvs(base_dir, batch_size=batch_size, compute_timebins=False,checkpoint_dir=checkpoint_dir)

        elif command == "batch_with_timebins":
            # Batch processing mode - with timebins
            batch_process_csvs(base_dir, batch_size=batch_size,
                             time_bin_size=timebin_size, compute_timebins=True,checkpoint_dir=checkpoint_dir)

        elif command == "batch_with_pacing":
            # Batch processing mode - with pacing factors
            batch_process_csvs(base_dir, batch_size=batch_size,
                             time_bin_size=timebin_size, compute_pacing=True,checkpoint_dir=checkpoint_dir)

        elif command == "batch_with_all":
            # Batch processing mode - with timebins AND pacing factors
            batch_process_csvs(base_dir, batch_size=batch_size,
                             time_bin_size=timebin_size, compute_timebins=True, compute_pacing=True,checkpoint_dir=checkpoint_dir)

        elif command == "generate":
            # Generate summaries from batches
            generate(checkpoint_dir,output_dir)

        elif command == "generate_timebins":
            # Generate timebin summaries from timebin batches
            generate_timebins_from_batches(checkpoint_dir,output_dir)

        elif command == "batch_pacing_all":
            # Batch process pacing factors for all bots (raw, no constraints)
            batch_process_pacing(base_dir, batch_size=batch_size,
                               time_bin_size={timebin_size: None}, bot_option="all", checkpoint_dir=checkpoint_dir)

        elif command == "batch_pacing_bot":
            # Batch process pacing factors for specific bot (raw, no constraints)
            if len(sys.argv) < 3:
                print("Error: Please specify a bot name")
                print("Usage: python generator_polars_gpu.py batch_pacing_bot <bot_name>")
                is_valid_process = False
            else:
                bot_name = sys.argv[2]
                batch_process_pacing(base_dir, batch_size=batch_size,
                                   time_bin_size={timebin_size: None}, bot_option=bot_name, checkpoint_dir=checkpoint_dir)

        else:
            is_valid_process = False
            print("Unknown command:", command)
            print()
            print("Usage:")
            print("  python generator_polars_gpu.py batch")
            print("  python generator_polars_gpu.py batch_with_timebins")
            print("  python generator_polars_gpu.py batch_with_pacing")
            print("  python generator_polars_gpu.py batch_with_all")
            print("  python generator_polars_gpu.py generate")
            print("  python generator_polars_gpu.py generate_timebins")
            print("  python generator_polars_gpu.py batch_pacing_all")
            print("  python generator_polars_gpu.py batch_pacing_bot <bot_name>")

        if not is_valid_process:
            exit()
        elapsed_seconds = time.time() - start
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        processing_time = f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"
        print(f"\nProcessing Time: {processing_time}")

    else:
        print("Usage:")
        print("  python generator_polars_gpu.py batch                     # Process game metrics only")
        print("  python generator_polars_gpu.py batch_with_timebins       # Process with time bins")
        print("  python generator_polars_gpu.py batch_with_pacing         # Process with pacing factors")
        print("  python generator_polars_gpu.py batch_with_all            # Process with time bins AND pacing factors")
        print("  python generator_polars_gpu.py batch_pacing_all          # Process pacing for all bots")
        print("  python generator_polars_gpu.py batch_pacing_bot <name>   # Process pacing for specific bot")
        print("  python generator_polars_gpu.py generate                  # Generate game summaries")
        print("  python generator_polars_gpu.py generate_timebins         # Generate timebin summaries")
