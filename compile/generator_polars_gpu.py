"""
Polars GPU-accelerated generator with batch processing
Uses scan_csv for lazy evaluation and GPU acceleration
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


def process_batch_csvs(csv_paths, batch_checkpoint_dir="batched", time_bin_size=None, compute_timebins=False):
    """
    Process a batch of CSV files and create checkpoint

    Args:
        csv_paths: List of CSV file paths to process
        batch_checkpoint_dir: Directory to save checkpoints
        time_bin_size: Size of time bins (only used if compute_timebins=True)
        compute_timebins: Whether to compute time-binned data

    Returns:
        tuple: (batch_df, action_timebin_df, collision_timebin_df) or (batch_df, None, None)
    """
    os.makedirs(batch_checkpoint_dir, exist_ok=True)

    all_games_list = []
    time_fragment_list = [] if compute_timebins else None
    collision_fragment_list = [] if compute_timebins else None

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

        # Scan CSV with Polars lazy API
        # Specify schema to ensure numeric columns are read correctly
        lf = pl.scan_csv(csv_path, infer_schema_length=10000)

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

    # Concatenate all games from this batch
    batch_df = None
    action_timebin_df = None
    collision_timebin_df = None

    if all_games_list:
        batch_df = pl.concat(all_games_list)

    if compute_timebins:
        if time_fragment_list:
            action_timebin_df = pl.DataFrame(time_fragment_list)
        if collision_fragment_list:
            collision_timebin_df = pl.DataFrame(collision_fragment_list)

    return batch_df, action_timebin_df, collision_timebin_df


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

        bins = np.arange(0, match_dur + time_bin_size, time_bin_size)
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
    raw_data = lf.filter(
        (pl.col("Category") == "Collision") & (pl.col("State") == 0)
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

        bins = np.arange(0, match_dur + time_bin_size, time_bin_size)
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


def batch_process_csvs(base_dir, batch_size=50, checkpoint_dir="batched", time_bin_size=None, compute_timebins=False):
    """
    Process CSVs in batches and save checkpoints
    Similar to generator.py batch() function
    Structure: base_dir/BotA_vs_BotB/ConfigFolder/*.csv

    Args:
        base_dir: Base directory containing simulation data
        batch_size: Number of CSV files per batch
        checkpoint_dir: Directory to save checkpoints
        time_bin_size: Size of time bins (only used if compute_timebins=True)
        compute_timebins: Whether to compute time-binned data
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create separate checkpoint dirs for timebins if needed
    if compute_timebins:
        action_timebin_dir = os.path.join(checkpoint_dir, "action_timebins")
        collision_timebin_dir = os.path.join(checkpoint_dir, "collision_timebins")
        os.makedirs(action_timebin_dir, exist_ok=True)
        os.makedirs(collision_timebin_dir, exist_ok=True)

    # Find all CSV files grouped by matchup/config
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
            all_csvs.extend(csv_files)

    print(f"Found {len(all_csvs)} CSV files to process")

    # Determine which batches are already processed
    processed_batches = set()
    for f in os.listdir(checkpoint_dir):
        match = re.match(r"batch_(\d+)\.csv", f)
        if match:
            processed_batches.add(int(match.group(1)))

    # Process in batches
    total_batches = (len(all_csvs) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_num = batch_idx + 1

        # Skip if already processed
        if batch_num in processed_batches:
            print(f"Skipping batch {batch_num}/{total_batches} (already processed)")
            continue

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_csvs))
        batch_csvs = all_csvs[start_idx:end_idx]

        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_csvs)} files)...")

        batch_df, action_timebin_df, collision_timebin_df = process_batch_csvs(
            batch_csvs, checkpoint_dir, time_bin_size=time_bin_size, compute_timebins=compute_timebins
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


def generate_timebins_from_batches(checkpoint_dir, output_dir):
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
        action_lazy_frames = [pl.scan_csv(f) for f in action_batch_files]
        action_timebin_df = collect_with_gpu(pl.concat(action_lazy_frames))
        print(f"Loaded {len(action_timebin_df):,} action timebin records")

        print("\n Creating action time-bin summary...")
        summarize_action_timebins(action_timebin_df, output_dir)

    # Load collision timebin batches
    collision_batch_files = sorted(glob.glob(f"{checkpoint_dir}/collision_timebins/batch_*.csv"))
    if collision_batch_files:
        print(f"\n📂 Loading {len(collision_batch_files)} collision timebin batch files...")
        collision_lazy_frames = [pl.scan_csv(f) for f in collision_batch_files]
        collision_timebin_df = collect_with_gpu(pl.concat(collision_lazy_frames))
        print(f"Loaded {len(collision_timebin_df):,} collision timebin records")

        print("\n Creating collision time-bin summary...")
        summarize_collision_timebins(collision_timebin_df, output_dir)

    print("\n" + "=" * 60)
    print("🎉 Done! Created:")
    if action_batch_files:
        print("   - summary_action_timebins.csv")
    if collision_batch_files:
        print("   - summary_collision_timebins.csv")
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
        lf = pl.scan_csv(csv_path)
        raw_data = lf.filter(
            (pl.col("Category") == "Collision") & (pl.col("State") == 0)
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

            bins = np.arange(0, match_dur + time_bin_size, time_bin_size)
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
    lazy_frames = [pl.scan_csv(f) for f in batch_files]
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

    base_dir = "/Users/defdef/Documents/Simulation"
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

        elif command == "generate":
            # Generate summaries from batches
            generate(checkpoint_dir,output_dir)

        elif command == "generate_timebins":
            # Generate timebin summaries from timebin batches
            generate_timebins_from_batches(checkpoint_dir,output_dir)
        else:
            is_valid_process = False
            print("Unknown command:", command)
            print()
            print("Usage:")
            print("  python generator_polars_gpu.py batch")
            print("  python generator_polars_gpu.py batch_with_timebins")
            print("  python generator_polars_gpu.py generate")
            print("  python generator_polars_gpu.py generate_timebins")

        if not is_valid_process:
            exit()
        elapsed_seconds = time.time() - start
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        processing_time = f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"
        print(f"\nProcessing Time: {processing_time}")

    else:
        print("Usage:")
        print("  python generator_polars_gpu.py batch                  # Process game metrics only")
        print("  python generator_polars_gpu.py batch_with_timebins    # Process with time bins")
        print("  python generator_polars_gpu.py generate               # Generate game summaries")
        print("  python generator_polars_gpu.py generate_timebins      # Generate timebin summaries")
