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

# Check if GPU support is available
GPU_AVAILABLE = False
try:
    # Try a simple GPU operation to check availability
    pl.LazyFrame({"test": [1]}).collect(engine="gpu")
    GPU_AVAILABLE = True
    print("GPU support available - will use GPU acceleration")
except Exception:
    print("Using CPU (GPU not available)")

arena_center = np.array([0.24, 1.97])
arena_radius = 4.73485

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
    Returns list of time-binned action records
    """
    # Scan and filter for actions
    raw_data = lf.filter(
        (pl.col("Category") == "Action") & (pl.col("State").cast(pl.Int32) != 2)
    ).select([
        "GameIndex", "Actor", "UpdatedAt", "Name"
    ])

    # Add match duration per game
    match_dur_lf = lf.group_by("GameIndex").agg([
        pl.col("UpdatedAt").max().alias("match_duration")
    ])

    raw_data = raw_data.join(match_dur_lf, on="GameIndex", how="left")
    raw_data_df = collect_with_gpu(raw_data)

    time_fragment_list = []

    # Process time bins per game
    for game_idx in raw_data_df['GameIndex'].unique():
        game_df = raw_data_df.filter(pl.col('GameIndex') == game_idx)
        match_dur = game_df['match_duration'][0]

        # Use timer config to determine max time bin (cap at timer setting)
        timer_value = config.get('Timer')
        max_time = min(match_dur, timer_value) if timer_value else match_dur

        bins = np.arange(0, max_time + time_bin_size, time_bin_size)
        if len(bins) < 2:
            continue

        game_pd = game_df.to_pandas()

        for side in [0, 1]:
            actor_data = game_pd[game_pd['Actor'] == side]
            if len(actor_data) == 0:
                continue

            actor_data = actor_data.copy()
            actor_data['TimeBin'] = pd.cut(actor_data['UpdatedAt'], bins=bins,
                                           labels=bins[:-1], include_lowest=True)

            grouped = actor_data.groupby(['TimeBin', 'Name'], observed=False).size().reset_index(name='Count')

            for _, row in grouped.iterrows():
                time_fragment_list.append({
                    'GameIndex': game_idx,
                    'Bot': bot_a if side == 0 else bot_b,
                    'Timer': config.get('Timer'),
                    'ActInterval': config.get('ActInterval'),
                    'Round': config.get('Round'),
                    'SkillLeft': config.get('SkillLeft'),
                    'SkillRight': config.get('SkillRight'),
                    'TimeBin': float(row['TimeBin']),
                    'Action': row['Name'],
                    'Count': row['Count']
                })

    return time_fragment_list


def process_collision_timebins_single_csv(lf, bot_a, bot_b, config, time_bin_size):
    """
    Process collision time bins for a single CSV file
    Returns list of time-binned collision records
    """
    # Scan and filter for collisions
    # Cast State to string for consistent comparison (handles both CSV strings and Parquet types)
    raw_data = lf.filter(
        (pl.col("Category") == "Collision") & (pl.col("State").cast(pl.Utf8) == "0")
    ).select([
        "GameIndex", "Actor", "ColTieBreaker", "ColActor", "UpdatedAt"
    ])

    # Add match duration per game
    match_dur_lf = lf.group_by("GameIndex").agg([
        pl.col("UpdatedAt").max().alias("match_duration")
    ])

    raw_data = raw_data.join(match_dur_lf, on="GameIndex", how="left")
    raw_data_df = collect_with_gpu(raw_data)

    collision_fragment_list = []

    # Process collision time bins per game
    for game_idx in raw_data_df['GameIndex'].unique():
        game_df = raw_data_df.filter(pl.col('GameIndex') == game_idx)
        match_dur = game_df['match_duration'][0]

        # Use timer config to determine max time bin (cap at timer setting)
        timer_value = config.get('Timer')
        max_time = min(match_dur, timer_value) if timer_value else match_dur

        bins = np.arange(0, max_time + time_bin_size, time_bin_size)
        if len(bins) < 2:
            continue

        game_pd = game_df.to_pandas()
        game_pd['TimeBin'] = pd.cut(game_pd['UpdatedAt'], bins=bins,
                                   labels=bins[:-1], include_lowest=True)

        for time_bin, bin_data in game_pd.groupby('TimeBin', observed=False):
            actor_L_count = len(bin_data[(bin_data['Actor'] == True) &
                                        (bin_data['ColTieBreaker'] == False) &
                                        (bin_data["ColActor"] == True)])
            # print(f"actor_L_count {actor_L_count}")
            actor_R_count = len(bin_data[(bin_data['Actor'] == False) &
                                        (bin_data['ColTieBreaker'] == False) &
                                        (bin_data["ColActor"] == True)])

            tie = bin_data['ColTieBreaker'].sum() if 'ColTieBreaker' in bin_data.columns else 0

            collision_fragment_list.append({
                'GameIndex': game_idx,
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


def process_pacing_factors_timebins_single_csv(lf, bot_a, bot_b, config, time_bin_size, skip_initial=0.0):
    """
    Process pacing factors per timebin for a single CSV/Parquet file
    Calculates 8 pacing factors for both bots in each timebin:
    - Threat: CollisionRatio, AbilityRatio, Angle, SafeDistance
    - Tempo: ActionIntensity, ActionDensity, BotsDistance, Velocity

    Args:
        lf: Lazy frame of the data
        bot_a: Name of bot A
        bot_b: Name of bot B
        config: Configuration dictionary
        time_bin_size: Size of time bins in seconds
        skip_initial: Number of seconds to skip at the start (default: 0.0)

    Returns list of time-binned pacing factor records (per matchup)
    """
    # Get match duration per game
    match_dur_lf = lf.group_by("GameIndex").agg([
        pl.col("UpdatedAt").max().alias("match_duration")
    ])
    match_durations = collect_with_gpu(match_dur_lf)

    pacing_fragment_list = []

    # Process each game
    for game_idx in match_durations['GameIndex']:
        match_dur = match_durations.filter(pl.col('GameIndex') == game_idx)['match_duration'][0]

        # Use timer config to determine max time bin (cap at timer setting)
        timer_value = config.get('Timer')
        max_time = min(match_dur, timer_value) if timer_value else match_dur

        # Apply skip_initial offset
        start_time = skip_initial
        bins = np.arange(start_time, max_time + time_bin_size, time_bin_size)
        if len(bins) < 2:
            continue

        # ===== 1. ACTION DATA (for ActionIntensity, ActionDensity, AbilityRatio) =====
        action_data_lf = lf.filter(
            (pl.col("GameIndex") == game_idx) &
            (pl.col("Category") == "Action") &
            (pl.col("State").cast(pl.Int32) != 2) &  # Exclude state 2
            (pl.col("UpdatedAt") >= skip_initial)  # Skip initial period
        ).select(["Actor", "UpdatedAt", "Name"])
        action_data = collect_with_gpu(action_data_lf).to_pandas()

        # ===== 2. COLLISION DATA (for CollisionRatio, Angle, SafeDistance) =====
        collision_data_lf = lf.filter(
            (pl.col("GameIndex") == game_idx) &
            (pl.col("Category") == "Collision") &
            (pl.col("State").cast(pl.Utf8) == "0") &
            (pl.col("UpdatedAt") >= skip_initial)  # Skip initial period
        ).select([
            "Actor", "UpdatedAt", "ColTieBreaker", "ColActor",
            "BotPosX", "BotPosY", "BotRot", "BotLinv",
            "EnemyBotPosX", "EnemyBotPosY", "EnemyBotRot", "EnemyBotLinv"
        ])
        collision_data = collect_with_gpu(collision_data_lf).to_pandas()

        # ===== 3. GENERAL POSITION DATA (for BotsDistance, Velocity) =====
        # Sample from all categories to get average distance/velocity
        position_data_lf = lf.filter(
            (pl.col("GameIndex") == game_idx) &
            (pl.col("UpdatedAt") >= skip_initial)  # Skip initial period
        ).select([
            "Actor", "UpdatedAt",
            "BotPosX", "BotPosY", "BotLinv",
            "EnemyBotPosX", "EnemyBotPosY", "EnemyBotLinv"
        ])
        position_data = collect_with_gpu(position_data_lf).to_pandas()

        # Process each timebin
        for i in range(len(bins) - 1):
            time_start = bins[i]
            time_end = bins[i + 1]

            # Filter data for this timebin
            action_bin = action_data[(action_data['UpdatedAt'] >= time_start) &
                                     (action_data['UpdatedAt'] < time_end)]
            collision_bin = collision_data[(collision_data['UpdatedAt'] >= time_start) &
                                           (collision_data['UpdatedAt'] < time_end)]
            position_bin = position_data[(position_data['UpdatedAt'] >= time_start) &
                                         (position_data['UpdatedAt'] < time_end)]

            # Calculate factors for each bot (Actor 0 = Bot_L, Actor 1 = Bot_R)
            factors = {}

            for actor_id, bot_suffix in [(0, "_L"), (1, "_R")]:
                action_actor = action_bin[action_bin['Actor'] == actor_id]
                collision_actor = collision_bin[collision_bin['Actor'] == actor_id]
                position_actor = position_bin[position_bin['Actor'] == actor_id]

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

                # 2. AbilityRatio: (Dash + Skills) / Total actions
                if len(action_actor) > 0:
                    ability_actions = len(action_actor[action_actor['Name'].isin(['Dash', 'SkillBoost', 'SkillStone'])])
                    total_actions = len(action_actor)
                    ability_ratio = float(ability_actions / total_actions if total_actions > 0 else np.nan)
                else:
                    ability_ratio = np.nan  # No action data in this timebin

                # 3. Angle: Average angle between bot and opponent during collisions
                if len(collision_actor) > 0:
                    # Calculate relative angle between bots
                    bot_rot = collision_actor['BotRot'].values
                    enemy_rot = collision_actor['EnemyBotRot'].values

                    # Angle difference (absolute, normalized to 0-180)
                    angle_diff = np.abs(bot_rot - enemy_rot)
                    angle_diff = np.minimum(angle_diff, 360 - angle_diff)  # Normalize to 0-180
                    avg_angle = float(np.mean(angle_diff[~np.isnan(angle_diff)]) if len(angle_diff) > 0 else np.nan)
                else:
                    avg_angle = np.nan  # No collision data in this timebin

                # 4. SafeDistance: Distance from arena edge (normalized)
                # safedistance = abs(arena_radius - robot_distance_from_center) / arena_radius
                if len(position_actor) > 0:
                    bot_x = position_actor['BotPosX'].values
                    bot_y = position_actor['BotPosY'].values

                    # Calculate distance from arena center
                    distance_from_center = np.sqrt((bot_x - arena_center[0])**2 + (bot_y - arena_center[1])**2)
                    # Calculate normalized distance from edge
                    safe_distances = np.abs(arena_radius - distance_from_center) / arena_radius
                    avg_safe_distance = float(np.mean(safe_distances[~np.isnan(safe_distances)]) if len(safe_distances) > 0 else np.nan)
                else:
                    avg_safe_distance = np.nan  # No collision data in this timebin

                # ----- TEMPO FACTORS -----

                # 5. ActionIntensity: Number of actions
                if len(action_actor) > 0:
                    action_intensity = float(len(action_actor))
                else:
                    action_intensity = np.nan  # No action data in this timebin

                # 6. ActionDensity: Shannon entropy of action distribution
                if len(action_actor) > 0:
                    action_counts = action_actor['Name'].value_counts(normalize=True)
                    entropy = -np.sum(action_counts * np.log2(action_counts + 1e-10))
                    action_density = float(entropy)
                else:
                    action_density = np.nan  # No action data in this timebin

                # 7. BotsDistance: Average distance between bots
                if len(position_actor) > 0:
                    bot_x = position_actor['BotPosX'].values
                    bot_y = position_actor['BotPosY'].values
                    enemy_x = position_actor['EnemyBotPosX'].values
                    enemy_y = position_actor['EnemyBotPosY'].values

                    distances = np.sqrt((bot_x - enemy_x)**2 + (bot_y - enemy_y)**2)
                    avg_bots_distance = float(np.mean(distances[~np.isnan(distances)]) if len(distances) > 0 else np.nan)
                else:
                    avg_bots_distance = np.nan  # No position data in this timebin

                # 8. Velocity: Average linear velocity
                if len(position_actor) > 0:
                    velocities = position_actor['BotLinv'].values
                    avg_velocity = float(np.mean(velocities[~np.isnan(velocities)]) if len(velocities) > 0 else np.nan)
                else:
                    avg_velocity = np.nan  # No position data in this timebin

                # Store factors with suffix
                factors[f'CollisionRatio{bot_suffix}'] = collision_ratio
                factors[f'AbilityRatio{bot_suffix}'] = ability_ratio
                factors[f'Angle{bot_suffix}'] = avg_angle
                factors[f'SafeDistance{bot_suffix}'] = avg_safe_distance
                factors[f'ActionIntensity{bot_suffix}'] = action_intensity
                factors[f'ActionDensity{bot_suffix}'] = action_density
                factors[f'BotsDistance{bot_suffix}'] = avg_bots_distance
                factors[f'Velocity{bot_suffix}'] = avg_velocity

            # Append record
            pacing_fragment_list.append({
                'GameIndex': game_idx,
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
    Each CSV can contain multiple games (GameIndex)
    """

    # Filter for actions only
    action_data = lf.filter(pl.col("Category") == "Action")

    # Compute durations with window function (lag by game/actor/name)
    action_with_lag = action_data.with_columns([
        pl.col("StartedAt").shift(1).over(["GameIndex", "Actor", "Name"], order_by="UpdatedAt").alias("prev_started_at")
    ])

    # Compute actual durations per game/actor/action
    action_durations = action_with_lag.group_by(["GameIndex", "Actor", "Name"]).agg([
        pl.when((pl.col("State").cast(pl.Int32) == 2) & pl.col("prev_started_at").is_not_null())
          .then(pl.col("UpdatedAt") - pl.col("prev_started_at"))
          .otherwise(0)
          .sum()
          .alias("ActualDuration")
    ])

    # Action counts per game/actor/action
    action_counts = action_data.filter(pl.col("State").cast(pl.Int32) != 2).group_by(["GameIndex", "Actor", "Name"]).agg([
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

    # Game metadata (winner and duration per game)
    game_meta = lf.group_by("GameIndex").agg([
        pl.col("GameWinner").first().alias("Winner"),
        pl.col("UpdatedAt").max().alias("MatchDur")
    ])

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


def batch_process_pacing(base_dir, batch_size=50, checkpoint_dir="batched", time_bin_size=None, bot_option="all", input_format="csv", skip_initial=0.0):
    """
    Process pacing factors in batches with bot filtering option

    Args:
        base_dir: Base directory containing simulation data
        batch_size: Number of files per batch
        checkpoint_dir: Directory to save checkpoints
        time_bin_size: Size of time bins for pacing factors
        bot_option: "all" for all bots, or specific bot name to filter (e.g., "BotA")
        input_format: "csv", "parquet", or "auto" to detect both
        skip_initial: Number of seconds to skip at the start to avoid initial bias (default: 0.0)
    """
    if time_bin_size is None:
        raise ValueError("time_bin_size must be specified for pacing factor computation")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint dir for pacing factors
    pacing_factors_dir = os.path.join(checkpoint_dir, "pacing_factors")
    os.makedirs(pacing_factors_dir, exist_ok=True)

    # Find all data files grouped by matchup/config
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

            # Collect files based on input format
            if input_format == "csv":
                files = glob.glob(os.path.join(config_path, "*.csv"))
            elif input_format == "parquet":
                files = glob.glob(os.path.join(config_path, "*.parquet"))
            else:  # auto - prefer parquet, fallback to csv
                parquet_files = glob.glob(os.path.join(config_path, "*.parquet"))
                csv_files = glob.glob(os.path.join(config_path, "*.csv"))
                files = parquet_files if parquet_files else csv_files

            all_files.extend(files)

    file_type = "Parquet" if input_format == "parquet" else "CSV/Parquet" if input_format == "auto" else "CSV"
    bot_filter_msg = f"for bot '{bot_option}'" if bot_option != "all" else "for all bots"
    print(f"Found {len(all_files)} {file_type} files to process {bot_filter_msg}")

    # Determine which batches are already processed
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

            # Scan file (CSV or Parquet) with Polars lazy API
            lf = scan_file(csv_path)

            # Process pacing factors
            pacing_tb = process_pacing_factors_timebins_single_csv(
                lf, bot_a, bot_b, config, time_bin_size, skip_initial=skip_initial
            )
            if pacing_tb:
                pacing_fragment_list.extend(pacing_tb)

        # Save pacing factors batch if computed
        if pacing_fragment_list:
            pacing_factors_df = pl.DataFrame(pacing_fragment_list)
            pacing_path = os.path.join(pacing_factors_dir, f"batch_{batch_num:02d}.csv")
            pacing_factors_df.write_csv(pacing_path)
            print(f"Saved pacing factors batch: {pacing_path} ({len(pacing_factors_df)} records)")


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

            # Collect files based on input format
            if input_format == "csv":
                files = glob.glob(os.path.join(config_path, "*.csv"))
            elif input_format == "parquet":
                files = glob.glob(os.path.join(config_path, "*.parquet"))
            else:  # auto - prefer parquet, fallback to csv
                parquet_files = glob.glob(os.path.join(config_path, "*.parquet"))
                csv_files = glob.glob(os.path.join(config_path, "*.csv"))
                files = parquet_files if parquet_files else csv_files

            all_files.extend(files)

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


def generate_timebins_from_batches(checkpoint_dir, output_dir, pacing_ratio_percentile = 5):
    """
    Generate timebin summaries from batched timebin checkpoints
    Loads batch files and creates final summaries
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

    # Load pacing factors batches
    pacing_batch_files = sorted(glob.glob(f"{checkpoint_dir}/pacing_factors/batch_*.csv"))
    if pacing_batch_files:
        print(f"\n📂 Loading {len(pacing_batch_files)} pacing factors batch files...")

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

        print("\n Creating pacing factors summary...")
        summarize_pacing_factors(pacing_factors_df, output_dir, pacing_ratio_percentile)

    print("\n" + "=" * 60)
    print("🎉 Done! Created:")
    if action_batch_files:
        print("   - summary_action_timebins.csv")
    if collision_batch_files:
        print("   - summary_collision_timebins.csv")
    if pacing_batch_files:
        print("   - summary_pacing_factors.csv")
        print("   - summary_pacing_per_bot.csv")
    print("=" * 60)


def compute_collision_time_bins_from_csvs(base_dir, time_bin_size=5):
    """
    Compute time-binned COLLISION data from CSV files.
    """
    # Find all CSV files
    all_csvs = []
    matchup_folders = [f for f in os.listdir(base_dir)
                       if os.path.isdir(os.path.join(base_dir, f))]

    for matchup_folder in matchup_folders:
        matchup_path = os.path.join(base_dir, matchup_folder)
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]

        for config_folder in config_folders:
            config_path = os.path.join(matchup_path, config_folder)
            csv_files = glob.glob(os.path.join(config_path, "*.csv"))
            all_csvs.extend([(csv, matchup_folder, config_folder) for csv in csv_files])

    print(f" Computing time-binned collision data from {len(all_csvs)} CSV files...")

    collision_fragment_list = []

    for csv_path, matchup_folder, config_folder in all_csvs:
        match = re.match(r"(.+)_vs_(.+)", matchup_folder)
        if not match:
            continue
        bot_a, bot_b = match.groups()

        config = parse_config_name_cached(config_folder)

        # Scan and filter for collisions
        lf = scan_file(csv_path)
        raw_data = lf.filter(
            (pl.col("Category") == "Collision") & (pl.col("State").cast(pl.Utf8) == "0")
        ).select([
            "GameIndex", "Actor", "ColTieBreaker", "ColActor", "UpdatedAt"
        ])

        # Add match duration per game
        match_dur_lf = lf.group_by("GameIndex").agg([
            pl.col("UpdatedAt").max().alias("match_duration")
        ])

        raw_data = raw_data.join(match_dur_lf, on="GameIndex", how="left")
        raw_data_df = collect_with_gpu(raw_data)

        # Process collision time bins per game
        for game_idx in raw_data_df['GameIndex'].unique():
            game_df = raw_data_df.filter(pl.col('GameIndex') == game_idx)
            match_dur = game_df['match_duration'][0]

            # Use timer config to determine max time bin (cap at timer setting)
            timer_value = config.get('Timer')
            max_time = min(match_dur, timer_value) if timer_value else match_dur

            bins = np.arange(0, max_time + time_bin_size, time_bin_size)
            if len(bins) < 2:
                continue

            game_pd = game_df.to_pandas()
            game_pd['TimeBin'] = pd.cut(game_pd['UpdatedAt'], bins=bins,
                                       labels=bins[:-1], include_lowest=True)

            for time_bin, bin_data in game_pd.groupby('TimeBin', observed=False):
                actor_L_count = len(bin_data[(bin_data['Actor'] == "0") &
                                            (bin_data['ColTieBreaker'] == "0") &
                                            (bin_data["ColActor"] == "1")])
                actor_R_count = len(bin_data[(bin_data['Actor'] == "1") &
                                            (bin_data['ColTieBreaker'] == "0") &
                                            (bin_data["ColActor"] == "1")])

                tie = bin_data['ColTieBreaker'].sum() if 'ColTieBreaker' in bin_data.columns else 0

                collision_fragment_list.append({
                    'GameIndex': game_idx,
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

    collision_fragment_df = pl.DataFrame(collision_fragment_list)
    print(f"Computed {len(collision_fragment_df):,} collision time-binned records")

    return collision_fragment_df


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


def summarize_pacing_factors(pacing_factors_df, output_dir, ratio_percentile = 5):
    """
    Summarize pacing factors with GPU acceleration.
    Computes mean and std for each factor per bot/config/timebin.
    Also provides per-bot statistics across all matchups.
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
    summary.write_csv(f"{output_dir}/summary_pacing_factors.csv")
    print(f"Saved {output_dir}/summary_pacing_factors.csv")

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
    bot_stats.write_csv(f"{output_dir}/summary_pacing_per_bot.csv")
    print(f"Saved {output_dir}/summary_pacing_per_bot.csv")
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
            # Batch process pacing factors for all bots
            batch_process_pacing(base_dir, batch_size=batch_size,
                               time_bin_size=timebin_size, bot_option="all", checkpoint_dir=checkpoint_dir)

        elif command == "batch_pacing_bot":
            # Batch process pacing factors for specific bot
            if len(sys.argv) < 3:
                print("Error: Please specify a bot name")
                print("Usage: python generator_polars_gpu.py batch_pacing_bot <bot_name>")
                is_valid_process = False
            else:
                bot_name = sys.argv[2]
                batch_process_pacing(base_dir, batch_size=batch_size,
                                   time_bin_size=timebin_size, bot_option=bot_name, checkpoint_dir=checkpoint_dir)

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
