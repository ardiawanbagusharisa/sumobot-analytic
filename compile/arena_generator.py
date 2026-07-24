import time
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import glob
from tqdm import tqdm
import json

# Check if GPU support is available
GPU_AVAILABLE = False
try:
    # Try a simple GPU operation to check availability
    pl.LazyFrame({"test": [1]}).collect(engine="gpu")
    GPU_AVAILABLE = True
    print("✅ GPU support available - will use GPU acceleration")
except Exception:
    print("✅ Using CPU")


def collect_with_gpu(lf, streaming=True):
    """Helper to collect LazyFrame with GPU if available, otherwise uses CPU with streaming"""
    if GPU_AVAILABLE:
        try:
            return lf.collect(engine="gpu", streaming=streaming)
        except Exception:
            # Fallback to CPU if GPU collection fails
            return lf.collect(streaming=streaming)
    else:
        return lf.collect(streaming=streaming)

# =====================
# Config - Load from unified config.json
# =====================
_config_path = Path(__file__).parent.parent / "config.json"
with open(_config_path, 'r') as f:
    _config = json.load(f)

arena_center = np.array(_config['arena']['center'])
arena_radius = _config['arena']['radius']
tile_size = _config['visualization']['tile_size']
# arrow_size = 50   # Larger = longer arrows

def scan_data_file(file_path):
    """
    Scan a data file (CSV or Parquet) and return a LazyFrame

    Args:
        file_path: Path to CSV or Parquet file

    Returns:
        Polars LazyFrame
    """
    if file_path.endswith('.parquet'):
        return pl.scan_parquet(file_path)
    elif file_path.endswith('.csv'):
        return pl.scan_csv(file_path, ignore_errors=True, rechunk=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Only .csv and .parquet are supported.")

def load_data_chunked(csv_path, chunksize=50000, actor_filter=None):
    """
    Load CSV or Parquet data using Polars with GPU acceleration and streaming

    Args:
        csv_path: Path to CSV or Parquet file
        chunksize: Number of rows per chunk (ignored for Polars, kept for API compatibility)
        actor_filter: Filter for specific actor (0 for left, 1 for right, None for both)
    """
    # Scan file (CSV or Parquet) without schema enforcement
    # Use ignore_errors for CSV to handle inconsistent column types across files
    # rechunk=False reduces memory overhead by avoiding unnecessary rechunking
    lf = scan_data_file(csv_path)

    # Select ONLY required columns to drastically reduce memory usage
    # This is critical for 135GB files - we only load what we need
    lf = lf.select([
        "GameIndex",     # For grouping by game
        "RoundIndex",    # Rounds in Best-of-N matches restart their own UpdatedAt clock near 0
        "UpdatedAt",     # For time-based analysis
        "Actor",         # For filtering by bot
        "BotPosX",       # X position
        "BotPosY",       # Y position
        "BotRot"         # Rotation (used for null checking)
    ])

    # Filter by actor if specified, casting Actor inline for comparison
    # IMPORTANT: Do this BEFORE collect to reduce memory usage
    if actor_filter is not None:
        lf = lf.filter(pl.col("Actor").cast(pl.Int64) == actor_filter)

    # Drop invalid entries BEFORE collecting to reduce memory footprint
    lf = lf.drop_nulls(subset=["BotPosX", "BotPosY", "BotRot"])

    # Collect with GPU acceleration and streaming enabled
    # streaming=True processes data in batches to avoid OOM
    df = collect_with_gpu(lf, streaming=True)

    return df

def split_into_phases(df, num_phases=3):
    """
    Split game data into phases based on UpdatedAt time, computed independently PER
    (GameIndex, RoundIndex). Best-of-N rounds each restart their own UpdatedAt clock near
    0, so phases must be computed per round: splitting per GameIndex alone would blend a
    round that ends quickly (e.g. a fast KO) with a longer round in the same game, letting
    the longer round's time range dictate where the short round's early/mid/late
    boundaries fall.

    Args:
        df: Polars DataFrame with game data (must have GameIndex and UpdatedAt columns;
            RoundIndex is used for per-round splitting when present)
        num_phases: Number of phases to split into (default: 3 for early/mid/late)

    Returns:
        List of Polars DataFrames, one per phase (aggregated across all games/rounds)
    """
    if df.is_empty():
        return [pl.DataFrame()] * num_phases

    # Initialize phase containers
    phases = [[] for _ in range(num_phases)]

    group_cols = ["GameIndex", "RoundIndex"] if "RoundIndex" in df.columns else ["GameIndex"]

    # Process each (game, round) independently
    for group_key in df.select(group_cols).unique().rows():
        group_df = df.filter(
            pl.all_horizontal([pl.col(c) == v for c, v in zip(group_cols, group_key)])
        )

        if group_df.is_empty():
            continue

        # Calculate time boundaries for THIS game/round
        min_time = group_df["UpdatedAt"].min()
        max_time = group_df["UpdatedAt"].max()
        time_range = max_time - min_time

        # Avoid division by zero for games with no time range
        if time_range == 0:
            # Put all data in the first phase
            phases[0].append(group_df)
            continue

        phase_size = time_range / num_phases

        # Split this game/round into phases
        for i in range(num_phases):
            phase_start = min_time + (i * phase_size)
            phase_end = min_time + ((i + 1) * phase_size)

            if i == num_phases - 1:
                # Include the last timestamp in the final phase
                phase_df = group_df.filter(
                    (pl.col("UpdatedAt") >= phase_start) & (pl.col("UpdatedAt") <= phase_end)
                )
            else:
                phase_df = group_df.filter(
                    (pl.col("UpdatedAt") >= phase_start) & (pl.col("UpdatedAt") < phase_end)
                )

            if not phase_df.is_empty():
                phases[i].append(phase_df)

    # Concatenate all games for each phase
    result_phases = []
    for phase_data in phases:
        if phase_data:
            result_phases.append(pl.concat(phase_data, how="vertical_relaxed"))
        else:
            result_phases.append(pl.DataFrame())

    return result_phases

def create_heatmap_data(x, y, tile_size):
    """Create heatmap data from position coordinates"""
    if len(x) == 0:
        return None, None, None

    xrange = np.arange(x.min(), x.max() + tile_size, tile_size)
    yrange = np.arange(y.min(), y.max() + tile_size, tile_size)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[xrange, yrange])

    return heatmap, xedges, yedges

def plot_phase_heatmap(ax, phase_df, phase_name):
    """Plot contour density heatmap for a single phase"""
    if phase_df.is_empty():
        ax.text(0.5, 0.5, f"No data for {phase_name}",
                ha='center', va='center', transform=ax.transAxes)
        return

    x = phase_df["BotPosX"].to_numpy() - arena_center[0]
    y = phase_df["BotPosY"].to_numpy() - arena_center[1]  # Shift by arena center

    # Create 2D kernel density estimation for smooth contours
    if len(x) > 1:
        from scipy.stats import gaussian_kde

        # Create KDE
        try:
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)

            # Create grid for evaluation (data shifted by arena_center, so center is at origin)
            x_min, x_max = 0 - arena_radius - 1, 0 + arena_radius + 1
            y_min, y_max = 0 - arena_radius - 1, 0 + arena_radius + 1

            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = np.reshape(kde(positions).T, xx.shape)

            # Plot filled contours (density heatmap)
            ax.contourf(xx, yy, density, levels=15, cmap="Greens", alpha=0.8, zorder=1)

            # Optionally add contour lines for better definition
            ax.contour(xx, yy, density, levels=5, colors='darkgreen', alpha=0.3, linewidths=0.5, zorder=2)

        except Exception as e:
            # Fallback to scatter if KDE fails
            print(f"Warning: KDE failed for {phase_name}, using scatter plot. Error: {e}")
            ax.scatter(x, y, alpha=0.1, s=1, c='green', zorder=1)

    # Draw arena boundary AFTER contours so it appears on top (data shifted by arena_center)
    arena_center_shifted = np.array([0, 0])  # Center is at origin after shift
    circle = plt.Circle(arena_center_shifted, arena_radius,
                       fill=False, edgecolor="red",
                       linewidth=2, linestyle="--", zorder=3)
    ax.add_artist(circle)

    # Labels & Arena Bounds
    ax.set_title(f"{phase_name}\n(n={len(phase_df):,} samples)")
    ax.set_xlabel("BotPosX")
    ax.set_ylabel("BotPosY")
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(0 - arena_radius - 1, 0 + arena_radius + 1)
    ax.set_ylim(0 - arena_radius - 1, 0 + arena_radius + 1)

    # Add grid
    ax.grid(True, alpha=0.3, zorder=0)


def _write_cache_csv(df, cache_path, max_samples=50000, use_parquet=True):
    """
    Write a small Polars DataFrame of chart-ready data to cache_path, creating parent dirs.
    Downsamples to max_samples if the DataFrame is larger to keep cache files small and fast.

    Args:
        df: Polars DataFrame to cache
        cache_path: Path to write the CSV cache
        max_samples: Maximum number of samples to store (default: 50000)
        use_parquet: Use Parquet format instead of CSV (much faster, default: True)
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Downsample if needed to keep cache small and fast
    if len(df) > max_samples:
        # Random sample to max_samples
        df = df.sample(n=max_samples, shuffle=True, seed=42)
        print(f"    (downsampled {len(df):,} → {max_samples:,} samples for cache)")

    # Use Parquet for much faster I/O (10-100x faster than CSV)
    if use_parquet:
        cache_path = cache_path.replace('.csv', '.parquet')
        df.write_parquet(cache_path, compression='zstd')
    else:
        df.write_csv(cache_path)


def _cached_files(cache_dir_path, pattern):
    """
    Return sorted cache file paths (CSV or Parquet) matching pattern under cache_dir_path,
    or [] if the directory doesn't exist. Prefers Parquet over CSV when both exist.
    Purely an existence check - it does NOT verify the cache was produced with the same
    skip_initial/actor_position/etc params as the current call.
    """
    if not cache_dir_path or not os.path.isdir(cache_dir_path):
        return []

    # Check for both CSV and Parquet files
    csv_files = sorted(glob.glob(os.path.join(cache_dir_path, pattern)))
    parquet_pattern = pattern.replace('.csv', '.parquet')
    parquet_files = sorted(glob.glob(os.path.join(cache_dir_path, parquet_pattern)))

    # Prefer Parquet over CSV (faster)
    if parquet_files:
        return parquet_files

    return csv_files


def plot_position_distribution(df_combined, bot_name, actor_position="both", cache_path=None):
    """
    Plot X and Y position distributions in a single frame (overlaid histograms)
    Y values are shifted by -2 since the game starts at y=2

    Args:
        df_combined: Combined Polars DataFrame with bot position data
        bot_name: Name of the bot
        actor_position: Position filter text for title
        cache_path: If set, write the raw BotPosX/BotPosY samples used for this chart to
            this CSV path, so the chart can be redrawn later via
            plot_position_distribution_from_cache() without reloading/refiltering the
            underlying simulation data (useful for iterating on titles/labels/colors when
            the source data is very large).

    Returns:
        matplotlib figure
    """
    if df_combined.is_empty():
        return None

    bot_x = df_combined["BotPosX"].to_numpy()
    bot_y = df_combined["BotPosY"].to_numpy()

    if cache_path:
        _write_cache_csv(pl.DataFrame({"BotPosX": bot_x, "BotPosY": bot_y}), cache_path)

    return _render_position_distribution(bot_x - arena_center[0], bot_y - arena_center[1], bot_name, actor_position, len(df_combined))


def plot_position_distribution_from_cache(cache_path, bot_name, actor_position="both"):
    """
    Redraw the position distribution chart from a CSV previously written by
    plot_position_distribution(..., cache_path=...), skipping the expensive data load.
    """
    df = pl.read_csv(cache_path)
    if df.is_empty():
        return None

    x = df["BotPosX"].to_numpy() - arena_center[0]
    y = df["BotPosY"].to_numpy() - arena_center[1]
    return _render_position_distribution(x, y, bot_name, actor_position, len(df))


def _render_position_distribution(x, y, bot_name, actor_position, n_samples):
    """Pure rendering step for plot_position_distribution - only touches small arrays,
    so this is the part to edit when iterating on chart design (title/labels/colors)."""
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot X distribution
    ax.hist(x, bins=30, alpha=0.7, color='green', edgecolor='darkgreen',
            label=f'{bot_name} X', linewidth=0.5)

    # Plot Y distribution (overlaid, shifted)
    ax.hist(y, bins=30, alpha=0.7, color='red', edgecolor='darkred',
            label=f'{bot_name} Y', linewidth=0.5)

    # Customize plot
    position_text = f" ({actor_position} side)" if actor_position != "both" else ""
    ax.set_title(f"Distribution of {bot_name} Positions{position_text}\n(n={n_samples:,} samples)",
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_joint_heatmap_with_distributions(phase_df, phase_name, bot_name="", actor_position="both", cache_path=None):
    """
    Create a joint plot with contour heatmap and marginal distributions (like seaborn jointplot)
    Y values are shifted by -2 since the game starts at y=2

    Args:
        phase_df: Polars DataFrame with position data for a specific phase
        phase_name: Name of the phase (e.g., "Early Game")
        bot_name: Name of the bot
        actor_position: Position filter text for title
        cache_path: If set, write the raw BotPosX/BotPosY samples used for this chart to
            this CSV path, so the chart can be redrawn later via
            plot_joint_heatmap_from_cache() without reloading/refiltering/re-KDE'ing the
            underlying simulation data (useful for iterating on titles/labels/colors when
            the source data is very large - this is the most expensive plot to regenerate).

    Returns:
        matplotlib figure
    """
    if phase_df.is_empty():
        return None

    bot_x = phase_df["BotPosX"].to_numpy()
    bot_y = phase_df["BotPosY"].to_numpy()

    if cache_path:
        _write_cache_csv(pl.DataFrame({"BotPosX": bot_x, "BotPosY": bot_y}), cache_path)

    return _render_joint_heatmap(bot_x - arena_center[0], bot_y - arena_center[1], phase_name, bot_name, actor_position, len(phase_df))


def plot_joint_heatmap_from_cache(cache_path, phase_name, bot_name="", actor_position="both"):
    """
    Redraw the joint heatmap (contour + marginal distributions) from a cached file previously
    written by plot_joint_heatmap_with_distributions(..., cache_path=...), skipping the
    expensive data load/filter step. The KDE itself is still recomputed here (cheap on the
    small cached array), so bin/bandwidth-independent design changes (title, labels,
    colors, colormap) can be iterated on quickly.

    Supports both CSV and Parquet formats (Parquet is 10-100x faster).
    """
    # Support both Parquet and CSV
    if cache_path.endswith('.csv'):
        # Check if Parquet version exists (faster)
        parquet_path = cache_path.replace('.csv', '.parquet')
        if os.path.exists(parquet_path):
            cache_path = parquet_path

    # Read cache file
    if cache_path.endswith('.parquet'):
        df = pl.read_parquet(cache_path)
    else:
        df = pl.read_csv(cache_path)

    if df.is_empty():
        return None

    x = df["BotPosX"].to_numpy() - arena_center[0]
    y = df["BotPosY"].to_numpy() - arena_center[1]
    return _render_joint_heatmap(x, y, phase_name, bot_name, actor_position, len(df))


def _render_joint_heatmap(x, y, phase_name, bot_name, actor_position, n_samples):
    """Pure rendering step for plot_joint_heatmap_with_distributions - only touches the
    small x/y arrays, so this is the part to edit when iterating on chart design."""
    # Create figure with GridSpec for joint plot layout
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)

    # Main central plot (contour heatmap)
    ax_main = fig.add_subplot(gs[1:4, 0:3])

    # Top marginal (X distribution)
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)

    # Right marginal (Y distribution)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # Set white background for main axis
    ax_main.set_facecolor('white')

    # Plot contour heatmap on main axis
    if len(x) > 1:
        from scipy.stats import gaussian_kde
        from matplotlib.colors import LinearSegmentedColormap

        try:
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)

            # Create grid for evaluation (data shifted by arena_center, so center is at origin)
            x_min, x_max = 0 - arena_radius - 1, 0 + arena_radius + 1
            y_min, y_max = 0 - arena_radius - 1, 0 + arena_radius + 1

            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = np.reshape(kde(positions).T, xx.shape)

            # Mask low-density areas to keep background white
            threshold = np.percentile(density, 40)  # Mask bottom n% of density
            density_masked = np.ma.masked_where(density < threshold, density)

            # Create custom colormap: white -> light green -> strong dark green (more layers)
            colors_list = [ '#E0FFE0', '#C0FFC0', '#90EE90', '#66DD66',
                          '#32CD32', '#2AAA2A', '#228B22', '#1A6B1A', '#006400']
            n_bins = 256
            cmap = LinearSegmentedColormap.from_list('green_gradient', colors_list, N=n_bins)

            # Plot filled contours with masked data - only areas above threshold
            ax_main.contourf(xx, yy, density_masked, levels=10, cmap=cmap, zorder=1)
            ax_main.contour(xx, yy, density_masked, levels=10, colors='darkgreen', alpha=0.4,
                           linewidths=0.5, zorder=2)

        except Exception as e:
            print(f"Warning: KDE failed for {phase_name}, using scatter plot. Error: {e}")
            ax_main.scatter(x, y, alpha=0.1, s=1, c='green', zorder=1)

    # Draw arena boundary (Y shifted by -2)
    arena_center_shifted = np.array([0, 0])  # Center is at origin after shift
    circle = plt.Circle(arena_center_shifted, arena_radius,
                       fill=False, edgecolor="red",
                       linewidth=2, linestyle="--", zorder=3)
    ax_main.add_artist(circle)

    # Configure main axis
    ax_main.set_xlabel("X Position", fontsize=12)
    ax_main.set_ylabel("Y Position", fontsize=12)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.set_xlim(0 - arena_radius - 1, 0 + arena_radius + 1)
    ax_main.set_ylim(0 - arena_radius - 1, 0 + arena_radius + 1)
    ax_main.grid(True, alpha=0.3, zorder=0)

    # Plot marginal distributions
    # Top: X distribution (histogram with KDE line)
    ax_top.hist(x, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5, density=True)

    # Add KDE line for X
    if len(x) > 1:
        try:
            from scipy.stats import gaussian_kde
            kde_x = gaussian_kde(x)
            x_range = np.linspace(x.min(), x.max(), 200)
            ax_top.plot(x_range, kde_x(x_range), 'darkblue', linewidth=2)
        except Exception:
            pass

    ax_top.set_ylabel("Density", fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['top'].set_visible(False)

    # Right: Y distribution (histogram with KDE line, rotated)
    ax_right.hist(y, bins=50, color='steelblue', alpha=0.7, edgecolor='black',
                  linewidth=0.5, orientation='horizontal', density=True)

    # Add KDE line for Y
    if len(y) > 1:
        try:
            from scipy.stats import gaussian_kde
            kde_y = gaussian_kde(y)
            y_range = np.linspace(y.min(), y.max(), 200)
            ax_right.plot(kde_y(y_range), y_range, 'darkblue', linewidth=2)
        except Exception:
            pass

    ax_right.set_xlabel("Density", fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['top'].set_visible(False)

    # Add title
    position_text = f" ({actor_position} side)" if actor_position != "both" else ""
    title = f"{bot_name}{position_text} - {phase_name}\n(n={n_samples:,} samples)"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    return fig

def create_phased_heatmap(csv_path, output_path=None, chunksize=50000):
    """
    Create a 3-phase heatmap visualization (early, mid, late game)

    Args:
        csv_path: Path to the game log CSV file
        output_path: Path to save the output image (optional)
        chunksize: Size of chunks for reading large CSV files
    """
    print(f"Loading data from {csv_path}...")
    df = load_data_chunked(csv_path, chunksize)

    if df.is_empty():
        print("No valid data found in the CSV file.")
        return

    print(f"Total samples: {len(df):,}")
    print(f"Time range: {df['UpdatedAt'].min():.2f} - {df['UpdatedAt'].max():.2f}")

    # Split into phases
    print("Splitting into phases...")
    phases = split_into_phases(df, num_phases=3)
    phase_names = ["Early Game", "Mid Game", "Late Game"]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Plot each phase
    for ax, phase_df, phase_name in zip(axes, phases, phase_names):
        print(f"Plotting {phase_name}...")
        plot_phase_heatmap(ax, phase_df, phase_name)

    plt.suptitle(f"Sumobot Arena Heatmap - Phased Analysis\n{Path(csv_path).name}",
                 fontsize=16, y=0.98)
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

def extract_timer_from_config(config_folder):
    """
    Extract Timer value from config folder name
    e.g., "Timer_15__ActInterval_0.1" -> 15

    Args:
        config_folder: Config folder name

    Returns:
        Timer value as float or None if not found
    """
    import re
    match = re.search(r'Timer_(\d+(?:\.\d+)?)', config_folder)
    if match:
        return float(match.group(1))
    return None


def load_bot_data_from_simulation(base_dir, bot_name, actor_position="left", chunksize=50000, max_configs=None, group_by_timer=False, also_load_distance=False):
    """
    Load all CSV or Parquet data for a specific bot from the simulation directory

    Args:
        base_dir: Base simulation directory
        bot_name: Name of the bot (e.g., "Bot_BT", "Bot_NN", "Bot_Primitive")
        actor_position: "left" (Actor 0) or "right" (Actor 1) or "both"
        chunksize: Chunk size for reading files (ignored for Parquet)
        max_configs: Maximum number of config folders to process (None for all)
        group_by_timer: If True, return dict of {timer_value: DataFrame}, else return combined DataFrame
        also_load_distance: If True, also return timer-grouped distance data

    Returns:
        Combined DataFrame with all bot data, or dict of DataFrames grouped by Timer
        If also_load_distance=True, returns tuple: (bot_data, distance_data)

    Note:
        Prefers Parquet files over CSV if both exist in the config folder
    """
    all_data = []
    timer_grouped_data = {}  # {timer_value: [dataframes]}
    timer_distance_data = {}  # {timer_value: [distance dataframes]}

    # Find all matchup folders containing this bot
    matchup_folders = [f for f in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, f)) and bot_name in f]

    print(f"Found {len(matchup_folders)} matchup folders for {bot_name}")

    total_csvs = 0
    for matchup_folder in matchup_folders:
        matchup_path = os.path.join(base_dir, matchup_folder)

        # Determine actor filter based on bot position in matchup
        # Bot_A_vs_Bot_B: Bot_A is actor 0 (left), Bot_B is actor 1 (right)
        parts = matchup_folder.split("_vs_")
        if len(parts) == 2:
            left_bot = parts[0]
            is_left_bot = (bot_name == left_bot)

            if actor_position == "left" and is_left_bot:
                actor_filter = 0
            elif actor_position == "left" and not is_left_bot:
                continue  # Skip this matchup
            elif actor_position == "right" and not is_left_bot:
                actor_filter = 1
            elif actor_position == "right" and is_left_bot:
                continue  # Skip this matchup
            elif actor_position == "both":
                actor_filter = 0 if is_left_bot else 1
            else:
                continue
        else:
            continue

        # Get all config folders
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]

        if max_configs:
            config_folders = config_folders[:max_configs]

        print(f"  {matchup_folder}: {len(config_folders)} configs")

        # Process each config folder
        for config_folder in tqdm(config_folders, desc=f"  Loading {matchup_folder}", leave=False):
            config_path = os.path.join(matchup_path, config_folder)

            # Find data file (prefer Parquet, fallback to CSV) in this config folder
            parquet_files = glob.glob(os.path.join(config_path, "*.parquet"))
            csv_files = glob.glob(os.path.join(config_path, "*.csv"))
            data_files = parquet_files if parquet_files else csv_files

            if data_files:
                data_path = data_files[0]  # Should only be 1 file per config
                df = load_data_chunked(data_path, chunksize, actor_filter=actor_filter)

                if not df.is_empty():
                    # Also load distance data if requested
                    if also_load_distance:
                        df_all_actors = load_data_chunked(data_path, chunksize, actor_filter=None)
                        if not df_all_actors.is_empty():
                            dist_df = calculate_distance_between_bots(df_all_actors)
                            if not dist_df.is_empty():
                                timer = extract_timer_from_config(config_folder)
                                if timer is not None:
                                    if timer not in timer_distance_data:
                                        timer_distance_data[timer] = []
                                    timer_distance_data[timer].append(dist_df)

                    if group_by_timer:
                        # Extract timer value and group
                        timer = extract_timer_from_config(config_folder)
                        if timer is not None:
                            if timer not in timer_grouped_data:
                                timer_grouped_data[timer] = []
                            timer_grouped_data[timer].append(df)
                    else:
                        all_data.append(df)
                    total_csvs += 1

    if group_by_timer:
        # Return dict of combined DataFrames per timer
        if not timer_grouped_data:
            print("No valid data found.")
            if also_load_distance:
                return {}, {}
            return {}

        print(f"\nLoaded {total_csvs} CSV files")
        result = {}
        for timer, dfs in timer_grouped_data.items():
            print(f"Combining data for Timer={timer}...")
            result[timer] = pl.concat(dfs, how="vertical_relaxed")
            print(f"  Timer {timer}: {len(result[timer]):,} samples")

        if also_load_distance:
            return result, timer_distance_data
        return result
    else:
        # Return combined DataFrame
        if not all_data:
            print("No valid data found.")
            if also_load_distance:
                return pl.DataFrame(), {}
            return pl.DataFrame()

        print(f"\nLoaded {total_csvs} CSV files")
        print("Combining all data...")
        df_combined = pl.concat(all_data, how="vertical_relaxed")

        print(f"Total samples: {len(df_combined):,}")

        if also_load_distance:
            return df_combined, timer_distance_data
        return df_combined


def load_all_bots_data_from_simulation(base_dir, chunksize=50000, max_configs=None, group_by_timer=False, also_load_distance=False, input_format="auto", filter_matchups=None):
    """
    Load position data across every matchup in the simulation directory, pooling both
    bots' data together (bot identity is not preserved). Used to build an aggregate
    "all bots combined" view alongside the per-bot ones from load_bot_data_from_simulation.

    Args:
        base_dir: Base simulation directory
        chunksize: Chunk size for reading files (ignored for Parquet)
        max_configs: Maximum number of config folders to process per matchup (None for all)
        group_by_timer: If True, return dict of {timer_value: DataFrame}, else return combined DataFrame
        also_load_distance: If True, also return timer-grouped distance-between-bots data
        input_format: "csv", "parquet", or "auto" (default: "auto" prefers parquet)
        filter_matchups: Optional list of matchup folder names to restrict to

    Returns:
        Combined DataFrame with all position data (both actors, every matchup), or dict of
        DataFrames grouped by Timer. If also_load_distance=True, returns tuple:
        (position_data, distance_data)
    """
    all_data = []
    timer_grouped_data = {}
    timer_distance_data = {}

    matchup_folders = [f for f in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, f)) and "_vs_" in f]
    if filter_matchups:
        matchup_folders = [m for m in matchup_folders if m in filter_matchups]

    print(f"Found {len(matchup_folders)} matchup folders")

    total_files = 0
    for matchup_folder in matchup_folders:
        matchup_path = os.path.join(base_dir, matchup_folder)
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]
        if max_configs:
            config_folders = config_folders[:max_configs]

        for config_folder in tqdm(config_folders, desc=f"  Loading {matchup_folder}", leave=False):
            config_path = os.path.join(matchup_path, config_folder)

            if input_format == "parquet":
                data_files = glob.glob(os.path.join(config_path, "*.parquet"))
            elif input_format == "csv":
                data_files = glob.glob(os.path.join(config_path, "*.csv"))
            else:  # "auto" - prefer parquet
                parquet_files = glob.glob(os.path.join(config_path, "*.parquet"))
                csv_files = glob.glob(os.path.join(config_path, "*.csv"))
                data_files = parquet_files if parquet_files else csv_files

            if not data_files:
                continue

            data_path = data_files[0]
            # Load WITHOUT actor filter - we pool both bots' data together
            df = load_data_chunked(data_path, chunksize, actor_filter=None)

            if df.is_empty():
                continue

            if also_load_distance:
                dist_df = calculate_distance_between_bots(df)
                if not dist_df.is_empty():
                    timer = extract_timer_from_config(config_folder)
                    if timer is not None:
                        timer_distance_data.setdefault(timer, []).append(dist_df)

            if group_by_timer:
                timer = extract_timer_from_config(config_folder)
                if timer is not None:
                    timer_grouped_data.setdefault(timer, []).append(df)
            else:
                all_data.append(df)
            total_files += 1

    if group_by_timer:
        if not timer_grouped_data:
            print("No valid data found.")
            return ({}, {}) if also_load_distance else {}

        print(f"\nLoaded {total_files} files")
        result = {}
        for timer, dfs in timer_grouped_data.items():
            result[timer] = pl.concat(dfs, how="vertical_relaxed")
            print(f"  Timer {timer}: {len(result[timer]):,} samples")

        return (result, timer_distance_data) if also_load_distance else result
    else:
        if not all_data:
            print("No valid data found.")
            return (pl.DataFrame(), {}) if also_load_distance else pl.DataFrame()

        print(f"\nLoaded {total_files} files")
        df_combined = pl.concat(all_data, how="vertical_relaxed")
        print(f"Total samples: {len(df_combined):,}")

        return (df_combined, timer_distance_data) if also_load_distance else df_combined


def create_phased_heatmap_for_bot(base_dir, bot_name, actor_position="left", output_path=None, chunksize=50000, max_configs=None, use_timer=True):
    """
    Create heatmaps for a specific bot from simulation directory
    Can use either phases (early/mid/late) or Timer values from config

    Args:
        base_dir: Base simulation directory
        bot_name: Name of the bot (e.g., "Bot_BT", "Bot_NN")
        actor_position: "left" or "right" or "both" (which side to analyze)
        output_path: Path to save the output image
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of configs to process per matchup (None for all)
        use_timer: If True, group by Timer values instead of phases
    """
    print("=" * 60)
    mode_text = "Timer-based" if use_timer else "Phase-based"
    print(f"Creating {mode_text} heatmap for {bot_name} (position: {actor_position})")
    print("=" * 60)

    if use_timer:
        # Load data grouped by timer
        timer_data = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs, group_by_timer=True)

        if not timer_data:
            print("No data to plot.")
            return

        # Create plots for each timer value
        for timer in sorted(timer_data.keys()):
            df = timer_data[timer]
            print(f"\nProcessing Timer={timer}...")
            print(f"  Samples: {len(df):,}")
            print(f"  Time range: {df['UpdatedAt'].min():.2f} - {df['UpdatedAt'].max():.2f}")

            label = f"Timer {int(timer)}s" if timer == int(timer) else f"Timer {timer}s"
            fig = plot_joint_heatmap_with_distributions(df, label, bot_name, actor_position)

            if fig is not None:
                # Determine output path for this timer
                if output_path:
                    base_name = output_path.rsplit('.', 1)[0]
                    ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'png'
                    timer_output = f"{base_name}_timer_{int(timer) if timer == int(timer) else timer}.{ext}"
                    plt.savefig(timer_output, dpi=150, bbox_inches='tight')
                    print(f"  Saved to {timer_output}")
                    plt.close(fig)
                else:
                    plt.show()

        print(f"\n✅ Completed all timers for {bot_name}")

    else:
        # Phase-based mode (original behavior)
        # Load all data for this bot
        df_combined = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs, group_by_timer=False)

        if df_combined.is_empty():
            print("No data to plot.")
            return

        print(f"Time range: {df_combined['UpdatedAt'].min():.2f} - {df_combined['UpdatedAt'].max():.2f}")

        # Split into phases
        print("\nSplitting into phases...")
        phases = split_into_phases(df_combined, num_phases=3)
        phase_names = ["Early Game", "Mid Game", "Late Game"]

        # Create separate joint plots for each phase
        for idx, (phase_df, phase_name) in enumerate(zip(phases, phase_names)):
            print(f"Creating {phase_name} joint heatmap...")

            if phase_df.is_empty():
                print(f"  No data for {phase_name}, skipping...")
                continue

            fig = plot_joint_heatmap_with_distributions(phase_df, phase_name, bot_name, actor_position)

            if fig is not None:
                # Determine output path for this phase
                if output_path:
                    base_name = output_path.rsplit('.', 1)[0]
                    ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'png'
                    phase_output = f"{base_name}_{phase_name.replace(' ', '_').lower()}.{ext}"
                    plt.savefig(phase_output, dpi=150, bbox_inches='tight')
                    print(f"  Saved to {phase_output}")
                    plt.close(fig)
                else:
                    plt.show()

        print(f"\n✅ Completed all phases for {bot_name}")


def create_phased_heatmap_combined(csv_paths, bot_name, output_path=None, chunksize=50000):
    """
    Create a 3-phase heatmap from multiple CSV files combined

    Args:
        csv_paths: List of paths to CSV files
        bot_name: Name of the bot for the title
        output_path: Path to save the output image
        chunksize: Size of chunks for reading large CSV files
    """
    all_data = []

    print(f"Loading {len(csv_paths)} CSV files...")
    for i, csv_path in enumerate(csv_paths):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(csv_paths)} files...")

        df = load_data_chunked(csv_path, chunksize, actor_filter=0)
        if not df.is_empty():
            all_data.append(df)

    if not all_data:
        print("No valid data found.")
        return

    print("Combining all data...")
    df_combined = pl.concat(all_data)

    print(f"Total samples: {len(df_combined):,}")
    print(f"Time range: {df_combined['UpdatedAt'].min():.2f} - {df_combined['UpdatedAt'].max():.2f}")

    # Split into phases
    print("Splitting into phases...")
    phases = split_into_phases(df_combined, num_phases=3)
    phase_names = ["Early Game", "Mid Game", "Late Game"]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Plot each phase
    for ax, phase_df, phase_name in zip(axes, phases, phase_names):
        print(f"Plotting {phase_name}...")
        plot_phase_heatmap(ax, phase_df, phase_name)

    plt.suptitle(f"Sumobot Arena Heatmap - Phased Analysis: {bot_name}\n({len(csv_paths)} matches, {len(df_combined):,} total samples)",
                 fontsize=16, y=0.98)
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

def get_bot_heatmap_figures(base_dir, bot_name, actor_position="both", chunksize=50000, max_configs=None):
    """
    Generate matplotlib figures for bot heatmaps (for use in Streamlit/web display)

    Args:
        base_dir: Base simulation directory
        bot_name: Name of the bot (e.g., "Bot_BT", "Bot_NN")
        actor_position: "left", "right", or "both"
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of configs to process per matchup

    Returns:
        List of 3 matplotlib figures [early_fig, mid_fig, late_fig]
    """
    print(f"Loading data for {bot_name}...")

    # Load all data for this bot
    df_combined = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs)

    if df_combined.is_empty():
        print(f"No data found for {bot_name}")
        return [None, None, None]

    print(f"Total samples: {len(df_combined):,}")

    # Split into phases
    print("Splitting into phases...")
    phases = split_into_phases(df_combined, num_phases=3)
    phase_names = ["Early Game", "Mid Game", "Late Game"]

    # Create figures for each phase
    figures = []
    for phase_df, phase_name in zip(phases, phase_names):
        if phase_df.is_empty():
            figures.append(None)
            continue

        # Create single figure for this phase
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plot_phase_heatmap(ax, phase_df, phase_name)

        position_text = f" ({actor_position} side)" if actor_position != "both" else ""
        plt.suptitle(f"{bot_name}{position_text} - {phase_name}\n({len(phase_df):,} samples)",
                    fontsize=16, y=0.98)
        plt.tight_layout()

        figures.append(fig)

    return figures


def calculate_distance_between_bots(df):
    """
    Calculate distance between Bot 1 (Actor 0) and Bot 2 (Actor 1) for each game frame.

    Joins on RoundIndex in addition to GameIndex/UpdatedAt when available. Best-of-N
    rounds each restart their own UpdatedAt clock near 0, and the simulation ticks at a
    fixed rate from each round's start, so two different rounds of the same game can share
    identical UpdatedAt values — an inner join on (GameIndex, UpdatedAt) alone would then
    cross-match bot1's position in one round with bot2's position in a different round.

    Args:
        df: Polars DataFrame with columns including Actor, BotPosX, BotPosY, GameIndex,
            RoundIndex, UpdatedAt

    Returns:
        Polars DataFrame with distance between bots for each frame
    """
    join_cols = ["GameIndex", "RoundIndex", "UpdatedAt"] if "RoundIndex" in df.columns else ["GameIndex", "UpdatedAt"]
    select_cols = join_cols + ["BotPosX", "BotPosY"]

    # Split data by actor - cast Actor inline for filtering
    bot1_df = df.filter(pl.col("Actor").cast(pl.Int64) == 0).select(select_cols).rename(
        {"BotPosX": "Bot1_X", "BotPosY": "Bot1_Y"}
    )

    bot2_df = df.filter(pl.col("Actor").cast(pl.Int64) == 1).select(select_cols).rename(
        {"BotPosX": "Bot2_X", "BotPosY": "Bot2_Y"}
    )

    # Merge on GameIndex (+ RoundIndex) and UpdatedAt to align frames
    merged = bot1_df.join(bot2_df, on=join_cols, how="inner")

    # Calculate Euclidean distance
    merged = merged.with_columns([
        (((pl.col("Bot1_X") - pl.col("Bot2_X"))**2 +
          (pl.col("Bot1_Y") - pl.col("Bot2_Y"))**2).sqrt()).alias("Distance")
    ])

    return merged

def calculate_distance_from_center(df):
    """
    Calculate distance from arena center for each bot

    Args:
        df: Polars DataFrame with columns including Actor, BotPosX, BotPosY

    Returns:
        Polars DataFrame with distance from center for each bot
    """
    # Calculate distance from center for each position
    df = df.with_columns([
        (((pl.col("BotPosX") - arena_center[0])**2 +
          (pl.col("BotPosY") - arena_center[1])**2).sqrt()).alias("DistanceFromCenter")
    ])

    return df

def plot_distance_histogram_from_data(distance_data, bot_name, output_path=None, cache_path=None):
    """
    Plot histogram of distance between bot and all opponents

    Args:
        distance_data: Dict of {timer: [list of distance dataframes]}
        bot_name: Name of the bot to analyze
        output_path: Path to save the figure
        cache_path: If set, write the flattened (timer, Distance) samples used for this
            chart to this CSV path, so it can be redrawn later via
            plot_distance_histogram_from_cache() without reloading the underlying data.

    Returns:
        matplotlib figure
    """
    if not distance_data:
        print("No valid distance data found")
        return None

    # Combine all distance data across all timers and opponents
    all_distances = []
    cache_parts = [] if cache_path else None
    for timer, dfs in distance_data.items():
        combined_df = pl.concat(dfs, how="vertical_relaxed")
        dist_arr = combined_df["Distance"].to_numpy()
        all_distances.append(dist_arr)
        if cache_path:
            cache_parts.append(pl.DataFrame({"timer": [timer] * len(dist_arr), "Distance": dist_arr}))

    # Concatenate all distances
    distances = np.concatenate(all_distances)

    if cache_path:
        _write_cache_csv(pl.concat(cache_parts), cache_path)

    return _render_distance_histogram(distances, bot_name, output_path)


def plot_distance_histogram_from_cache(cache_path, bot_name, output_path=None):
    """
    Redraw the distance-between-bots histogram from a CSV previously written by
    plot_distance_histogram_from_data(..., cache_path=...), skipping the expensive data load.
    """
    df = pl.read_csv(cache_path)
    if df.is_empty():
        print("No valid distance data found")
        return None
    return _render_distance_histogram(df["Distance"].to_numpy(), bot_name, output_path)


def _render_distance_histogram(distances, bot_name, output_path=None):
    """Pure rendering step for plot_distance_histogram_from_data - only touches the small
    distances array, so this is the part to edit when iterating on chart design."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    ax.hist(distances, bins=30, color='steelblue', edgecolor='black', alpha=0.7, linewidth=0.5)

    # Customize plot
    ax.set_xlabel("Distance Between Bots", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Distribution of Distance Between Bots\n{bot_name} vs All Opponents\n(n={len(distances):,} samples)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add statistics text
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    std_dist = np.std(distances)
    stats_text = f"Mean: {mean_dist:.2f}\nMedian: {median_dist:.2f}\nStd: {std_dist:.2f}"
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')

    plt.tight_layout()

    # Save or return
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved distance histogram to {output_path}")

    return fig


def plot_distance_from_center_histogram(bot_data, bot_name, output_path=None, cache_path=None):
    """
    Plot histogram of distance from center for a specific bot

    Args:
        bot_data: DataFrame or dict of DataFrames with bot position data
        bot_name: Name of the bot to analyze
        output_path: Path to save the figure
        cache_path: If set, write the DistanceFromCenter samples used for this chart to
            this CSV path, so it can be redrawn later via
            plot_distance_from_center_histogram_from_cache() without reloading the
            underlying data.

    Returns:
        matplotlib figure
    """
    # Handle both single DataFrame and dict of DataFrames
    if isinstance(bot_data, dict):
        # Combine all timer data
        all_dfs = []
        for timer, df in bot_data.items():
            all_dfs.append(df)
        combined_df = pl.concat(all_dfs, how="vertical_relaxed")
    else:
        combined_df = bot_data

    if combined_df.is_empty():
        print("No valid data found")
        return None

    # Calculate distance from center
    df_with_center_dist = calculate_distance_from_center(combined_df)
    distances = df_with_center_dist["DistanceFromCenter"].to_numpy()

    if cache_path:
        _write_cache_csv(pl.DataFrame({"DistanceFromCenter": distances}), cache_path)

    return _render_distance_from_center_histogram(distances, bot_name, output_path)


def plot_distance_from_center_histogram_from_cache(cache_path, bot_name, output_path=None):
    """
    Redraw the distance-from-center histogram from a CSV previously written by
    plot_distance_from_center_histogram(..., cache_path=...), skipping the expensive data load.
    """
    df = pl.read_csv(cache_path)
    if df.is_empty():
        print("No valid data found")
        return None
    return _render_distance_from_center_histogram(df["DistanceFromCenter"].to_numpy(), bot_name, output_path)


def _render_distance_from_center_histogram(distances, bot_name, output_path=None):
    """Pure rendering step for plot_distance_from_center_histogram - only touches the
    small distances array, so this is the part to edit when iterating on chart design."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    ax.hist(distances, bins=30, color='green', edgecolor='darkgreen', alpha=0.7, linewidth=0.5)

    # Add arena radius line
    ax.axvline(arena_radius, color='red', linestyle='--', linewidth=2, label=f'Arena Radius ({arena_radius:.2f})')

    # Customize plot
    ax.set_xlabel("Distance from Center", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Distribution of Distance from Center\n{bot_name}\n(n={len(distances):,} samples)",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add statistics text
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    std_dist = np.std(distances)
    stats_text = f"Mean: {mean_dist:.2f}\nMedian: {median_dist:.2f}\nStd: {std_dist:.2f}"
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')

    plt.tight_layout()

    # Save or return
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved distance from center histogram to {output_path}")

    return fig


def plot_distance_over_time_from_data(timer_data, bot_name, output_path=None, cache_path=None):
    """
    Plot mean distance over time from pre-loaded timer-grouped data

    Args:
        timer_data: Dict of {timer: [list of distance dataframes]}
        bot_name: Name of the bot to analyze
        output_path: Path to save the figure
        cache_path: If set, write the flattened (timer, UpdatedAt, Distance) samples used
            for this chart to this CSV path, so it can be redrawn later via
            plot_distance_over_time_from_cache() without reloading the underlying data.

    Returns:
        matplotlib figure
    """
    if not timer_data:
        print("No valid data found")
        return None

    if cache_path:
        cache_parts = []
        for timer, dfs in timer_data.items():
            combined_df = pl.concat(dfs, how="vertical_relaxed").select(["UpdatedAt", "Distance"])
            cache_parts.append(combined_df.with_columns(pl.lit(timer).alias("timer")))
        _write_cache_csv(pl.concat(cache_parts), cache_path)

    return _render_distance_over_time(timer_data, bot_name, output_path)


def plot_distance_over_time_from_cache(cache_path, bot_name, output_path=None):
    """
    Redraw the distance-over-time chart from a CSV previously written by
    plot_distance_over_time_from_data(..., cache_path=...), skipping the expensive data load.
    """
    df = pl.read_csv(cache_path)
    if df.is_empty():
        print("No valid data found")
        return None
    timer_data = {timer_key[0]: [group.drop("timer")] for timer_key, group in df.group_by("timer")}
    return _render_distance_over_time(timer_data, bot_name, output_path)


def _render_distance_over_time(timer_data, bot_name, output_path=None):
    """Pure rendering step for plot_distance_over_time_from_data - only touches the small
    per-timer arrays, so this is the part to edit when iterating on chart design."""
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Process each timer value
    colors = plt.cm.tab10(range(len(timer_data)))

    for idx, (timer, dfs) in enumerate(sorted(timer_data.items())):
        # Combine all games for this timer (across all opponents)
        combined_df = pl.concat(dfs, how="vertical_relaxed")

        print(f"  Timer {timer}s: {len(combined_df):,} data points")

        # Calculate mean distance over time bins
        # Bin UpdatedAt into time intervals, but only up to the Timer value
        time_bins = 50  # Number of bins
        # Use the Timer value as the max time for this specific config
        max_time = timer  # Cut at the Timer config value
        bin_size = max_time / time_bins

        # Create time bins and calculate mean distance per bin
        time_points = []
        mean_distances = []
        std_distances = []

        for i in range(time_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size

            bin_data = combined_df.filter(
                (pl.col('UpdatedAt') >= bin_start) &
                (pl.col('UpdatedAt') < bin_end)
            )

            if not bin_data.is_empty():
                time_points.append((bin_start + bin_end) / 2)
                mean_distances.append(bin_data['Distance'].mean())
                # Handle None for std (when only 1 data point)
                std_val = bin_data['Distance'].std()
                std_distances.append(std_val if std_val is not None else 0.0)

        # Convert to numpy for plotting
        time_points = np.array(time_points)
        mean_distances = np.array(mean_distances)
        std_distances = np.array(std_distances)

        # Plot line with markers
        timer_label = f"Timer {int(timer)}s" if timer == int(timer) else f"Timer {timer}s"
        ax.plot(time_points, mean_distances, marker='o', markersize=4,
                linewidth=2, label=timer_label, color=colors[idx], alpha=0.8)

        # Add confidence interval (mean ± std)
        ax.fill_between(time_points,
                        mean_distances - std_distances,
                        mean_distances + std_distances,
                        alpha=0.2, color=colors[idx])

    # Customize plot
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Mean Distance Between Bots", fontsize=12)
    ax.set_title(f"Mean Distance Over Time (vs All Opponents)\n{bot_name}",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Limit x-axis to the maximum Timer value found
    max_timer = max(timer_data.keys())
    ax.set_xlim(0, max_timer)

    plt.tight_layout()

    # Save or return
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved distance over time to {output_path}")

    return fig


def plot_distance_distribution(between_series, from_center_series, name, output_path=None, cache_path=None, is_pooled=False):
    """
    Plot the 2-subplot "distance_distribution" chart: distance between bots (top) and
    distance from center (bottom), for one bot (averaged across all its matchups) or
    pooled across every bot.

    Args:
        between_series: Series/array of distance-between-bots samples
        from_center_series: Series/array of distance-from-center samples
        name: Bot name, or "All Bots" for the pooled aggregate
        output_path: Path to save the figure
        cache_path: If set, write the (kind, value) samples used for this chart to this CSV
            path, so it can be redrawn later via plot_distance_distribution_from_cache()
            without reloading/rejoining the underlying data.
        is_pooled: True for the pooled "All Bots" aggregate, which uses slightly different
            title wording than the per-bot chart.

    Returns:
        matplotlib figure
    """
    between_numpy = between_series.to_numpy() if hasattr(between_series, "to_numpy") else np.asarray(between_series)
    from_center_numpy = from_center_series.to_numpy() if hasattr(from_center_series, "to_numpy") else np.asarray(from_center_series)

    if cache_path:
        cache_df = pl.concat([
            pl.DataFrame({"kind": ["between"] * len(between_numpy), "value": between_numpy}),
            pl.DataFrame({"kind": ["from_center"] * len(from_center_numpy), "value": from_center_numpy}),
        ])
        _write_cache_csv(cache_df, cache_path)

    return _render_distance_distribution(between_numpy, from_center_numpy, name, output_path, is_pooled)


def plot_distance_distribution_from_cache(cache_path, name, output_path=None, is_pooled=False):
    """
    Redraw the distance_distribution chart from a CSV previously written by
    plot_distance_distribution(..., cache_path=...), skipping the expensive data load.
    """
    df = pl.read_csv(cache_path)
    between_numpy = df.filter(pl.col("kind") == "between")["value"].to_numpy()
    from_center_numpy = df.filter(pl.col("kind") == "from_center")["value"].to_numpy()
    return _render_distance_distribution(between_numpy, from_center_numpy, name, output_path, is_pooled)


def _render_distance_distribution(between_numpy, from_center_numpy, name, output_path=None, is_pooled=False):
    """Pure rendering step for plot_distance_distribution - only touches the small
    between/from_center arrays, so this is the part to edit when iterating on chart design."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Distance between bots (averaged across all matchups)
    ax1.hist(between_numpy, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    between_title = "Distance Between Bots (All Bots, All Matchups)" if is_pooled else f"Distance Between {name} and Opponents (All Matchups)"
    ax1.set_title(between_title, fontsize=14, fontweight='bold')
    ax1.set_xlabel("Distance Between Bots", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.text(0.98, 0.98, f"n={len(between_numpy):,}",
            transform=ax1.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Distance from center
    ax2.hist(from_center_numpy, bins=30, color='green', edgecolor='black', alpha=0.7)
    from_center_title = "Distance from Center: All Bots" if is_pooled else f"Distance from Center: {name}"
    ax2.set_title(from_center_title, fontsize=14, fontweight='bold')
    ax2.set_xlabel("Distance from Center", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Add arena radius reference line
    ax2.axvline(x=arena_radius, color='red', linestyle='--', linewidth=2,
               label=f'Arena Radius ({arena_radius:.2f})', alpha=0.8)
    ax2.legend(loc='upper right', fontsize=10)

    ax2.text(0.98, 0.98, f"n={len(from_center_numpy):,}",
            transform=ax2.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {output_path}")

    return fig


def plot_distance_over_time_by_timer_per_bot(base_dir, bot_name, output_path=None, chunksize=50000, max_configs=None):
    """
    Plot mean distance over time for a bot against ALL other bots, grouped by Timer configuration
    Shows how distance changes throughout the match for different Timer values

    Args:
        base_dir: Base simulation directory
        bot_name: Name of the bot to analyze
        output_path: Path to save the figure
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of configs to process

    Returns:
        matplotlib figure
    """
    # Find all matchup folders containing this bot
    matchup_folders = [f for f in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, f)) and bot_name in f and "_vs_" in f]

    if not matchup_folders:
        print(f"No matchup folders found for {bot_name}")
        return None

    print(f"Found {len(matchup_folders)} matchup folders for {bot_name}")

    # Group configs by Timer value
    timer_data = {}  # {timer: [list of distance dataframes]}

    for matchup_folder in matchup_folders:
        matchup_path = os.path.join(base_dir, matchup_folder)
        print(f"Processing {matchup_folder}...")

        # Get all config folders
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]

        if max_configs:
            config_folders = config_folders[:max_configs]

        for config_folder in tqdm(config_folders, desc=f"  {matchup_folder}", leave=False):
            # Extract timer value
            timer = extract_timer_from_config(config_folder)
            if timer is None:
                continue

            config_path = os.path.join(matchup_path, config_folder)
            csv_files = glob.glob(os.path.join(config_path, "*.csv"))

            if csv_files:
                csv_path = csv_files[0]
                # Load data (need both actors)
                df = load_data_chunked(csv_path, chunksize, actor_filter=None)

                if not df.is_empty():
                    # Calculate distance between bots
                    dist_df = calculate_distance_between_bots(df)

                    if not dist_df.is_empty():
                        if timer not in timer_data:
                            timer_data[timer] = []
                        timer_data[timer].append(dist_df)

    if not timer_data:
        print("No valid data found")
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Process each timer value
    colors = plt.cm.tab10(range(len(timer_data)))

    for idx, (timer, dfs) in enumerate(sorted(timer_data.items())):
        # Combine all games for this timer (across all opponents)
        combined_df = pl.concat(dfs, how="vertical_relaxed")

        print(f"\nTimer {timer}s: {len(combined_df):,} data points")

        # Calculate mean distance over time bins
        # Bin UpdatedAt into time intervals, but only up to the Timer value
        time_bins = 50  # Number of bins
        # Use the Timer value as the max time for this specific config
        max_time = timer  # Cut at the Timer config value
        bin_size = max_time / time_bins

        # Create time bins and calculate mean distance per bin
        time_points = []
        mean_distances = []
        std_distances = []

        for i in range(time_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size

            bin_data = combined_df.filter(
                (pl.col('UpdatedAt') >= bin_start) &
                (pl.col('UpdatedAt') < bin_end)
            )

            if not bin_data.is_empty():
                time_points.append((bin_start + bin_end) / 2)
                mean_distances.append(bin_data['Distance'].mean())
                # Handle None for std (when only 1 data point)
                std_val = bin_data['Distance'].std()
                std_distances.append(std_val if std_val is not None else 0.0)

        # Convert to numpy for plotting
        time_points = np.array(time_points)
        mean_distances = np.array(mean_distances)
        std_distances = np.array(std_distances)

        # Plot line with markers
        timer_label = f"Timer {int(timer)}s" if timer == int(timer) else f"Timer {timer}s"
        ax.plot(time_points, mean_distances, marker='o', markersize=4,
                linewidth=2, label=timer_label, color=colors[idx], alpha=0.8)

        # Add confidence interval (mean ± std)
        ax.fill_between(time_points,
                        mean_distances - std_distances,
                        mean_distances + std_distances,
                        alpha=0.2, color=colors[idx])

    # Customize plot
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Mean Distance Between Bots", fontsize=12)
    ax.set_title(f"Mean Distance Over Time (vs All Opponents)\n{bot_name}",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Limit x-axis to the maximum Timer value found
    max_timer = max(timer_data.keys())
    ax.set_xlim(0, max_timer)

    plt.tight_layout()

    # Save or return
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {output_path}")

    return fig


def plot_distance_distributions(df, bot1_name="Bot 1", bot2_name="Bot 2", output_path=None):
    """
    Create combined distance distribution plots:
    1. Distance between bots
    2. Distance from center for each bot (stacked histogram)

    Args:
        df: Polars DataFrame with game data (must have Actor, BotPosX, BotPosY, GameIndex, UpdatedAt)
        bot1_name: Name of Bot 1 (Actor 0 / left bot)
        bot2_name: Name of Bot 2 (Actor 1 / right bot)
        output_path: Path to save the figure (optional)

    Returns:
        matplotlib figure
    """
    if df.is_empty():
        print("No data to plot")
        return None

    # Calculate distances
    print("Calculating distance between bots...")
    dist_between = calculate_distance_between_bots(df)

    print("Calculating distance from center...")
    df_with_center_dist = calculate_distance_from_center(df)

    # Split by actor for center distance - use numeric comparison
    bot1_center_dist = df_with_center_dist.filter(
        pl.col("Actor") == 0
    )["DistanceFromCenter"].to_numpy()
    bot2_center_dist = df_with_center_dist.filter(
        pl.col("Actor") == 1
    )["DistanceFromCenter"].to_numpy()

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Distance between bots
    ax1.hist(dist_between["Distance"].to_numpy(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_title("Distribution of Distance Between Bots", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Distance Between Bots", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.text(0.98, 0.98, f"n={len(dist_between):,}",
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Distance from center (stacked histogram)
    bins = np.linspace(
        min(np.min(bot1_center_dist), np.min(bot2_center_dist)),
        max(np.max(bot1_center_dist), np.max(bot2_center_dist)),
        100
    )

    ax2.hist([bot1_center_dist, bot2_center_dist], bins=bins,
             label=[f'{bot1_name} Distance from Center', f'{bot2_name} Distance from Center'],
             color=['green', 'red'], edgecolor='black', alpha=0.6, stacked=False)
    ax2.set_title("Distribution of Bot Distance from Center", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Distance from Center", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Add arena radius reference line
    ax2.axvline(x=arena_radius, color='red', linestyle='--', linewidth=2,
                label=f'Arena Radius ({arena_radius:.2f})', alpha=0.8)
    ax2.legend(loc='upper right', fontsize=10)

    plt.suptitle(f"Distributions of distance between bots (use mean) & distance to center. If possible, combine those 2, 1 bot 1 frame.",
                 fontsize=12, y=0.995)
    plt.tight_layout()

    # Save or return
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")

    return fig

def load_all_game_data(base_dir, bot1_name=None, bot2_name=None, chunksize=50000, max_configs=None, input_format="auto"):
    """
    Load all game data from simulation directory, optionally filtered by bot matchup

    Args:
        base_dir: Base simulation directory
        bot1_name: Name of bot 1 (optional filter)
        bot2_name: Name of bot 2 (optional filter)
        chunksize: Chunk size for reading CSV/Parquet files
        max_configs: Maximum number of configs to process (None for all)
        input_format: "csv", "parquet", or "auto" (default: "auto" prefers parquet)

    Returns:
        DataFrame with all game data including both actors
    """
    all_data = []

    # Find matchup folders
    if bot1_name and bot2_name:
        # Specific matchup
        matchup_folder = f"{bot1_name}_vs_{bot2_name}"
        matchup_folders = [matchup_folder] if os.path.exists(os.path.join(base_dir, matchup_folder)) else []
    else:
        # All matchups
        matchup_folders = [f for f in os.listdir(base_dir)
                          if os.path.isdir(os.path.join(base_dir, f)) and "_vs_" in f]

    print(f"Found {len(matchup_folders)} matchup folders")

    total_csvs = 0
    for matchup_folder in matchup_folders:
        matchup_path = os.path.join(base_dir, matchup_folder)

        # Get all config folders
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]

        if max_configs:
            config_folders = config_folders[:max_configs]

        print(f"  {matchup_folder}: {len(config_folders)} configs")

        # Process each config folder
        for config_folder in tqdm(config_folders, desc=f"  Loading {matchup_folder}", leave=False):
            config_path = os.path.join(matchup_path, config_folder)

            # Find data file (prefer Parquet, fallback to CSV based on input_format)
            parquet_files = glob.glob(os.path.join(config_path, "*.parquet"))
            csv_files = glob.glob(os.path.join(config_path, "*.csv"))

            if input_format == "parquet":
                data_files = parquet_files
            elif input_format == "csv":
                data_files = csv_files
            else:  # "auto" - prefer parquet
                data_files = parquet_files if parquet_files else csv_files

            if data_files:
                data_path = data_files[0]
                # Load WITHOUT actor filter (we need both bots)
                df = load_data_chunked(data_path, chunksize, actor_filter=None)

                if not df.is_empty():
                    all_data.append(df)
                    total_csvs += 1

    if not all_data:
        print("No valid data found.")
        return pl.DataFrame()

    print(f"\nLoaded {total_csvs} CSV files")
    print("Combining all data...")
    df_combined = pl.concat(all_data, how="vertical_relaxed")

    print(f"Total samples: {len(df_combined):,}")

    return df_combined

def create_distance_over_time_all_bots(base_dir, output_dir="arena_heatmaps", chunksize=50000, max_configs=None, input_format="auto"):
    """
    Create distance over time line plots for all bots (vs all opponents, grouped by Timer)
    Saves plots in each bot's directory within the output_dir

    Args:
        base_dir: Base simulation directory
        output_dir: Base output directory (plots will be saved in bot subdirectories)
        chunksize: Chunk size for reading CSV/Parquet files
        max_configs: Maximum number of configs to process
        input_format: "csv", "parquet", or "auto" (default: "auto" prefers parquet)
    """
    # Find all unique bot names from matchup folders
    matchup_folders = [f for f in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, f)) and "_vs_" in f]

    bot_names = set()
    for matchup in matchup_folders:
        parts = matchup.split("_vs_")
        if len(parts) == 2:
            bot_names.add(parts[0])
            bot_names.add(parts[1])

    bot_names = sorted(bot_names)
    print(f"Found {len(bot_names)} unique bots: {bot_names}")

    # Process each bot
    for bot_name in bot_names:
        print("\n" + "=" * 60)
        print(f"Processing {bot_name}")
        print("=" * 60)

        # Create bot-specific directory if it doesn't exist
        bot_dir = os.path.join(output_dir, bot_name)
        os.makedirs(bot_dir, exist_ok=True)

        # Create distance over time plot (vs all opponents)
        output_path = os.path.join(bot_dir, "distance_over_time.png")
        fig = plot_distance_over_time_by_timer_per_bot(base_dir, bot_name, output_path, chunksize, max_configs)

        if fig is not None:
            plt.close(fig)

    print("\n" + "=" * 60)
    print(f"✅ Completed! All distance over time plots saved in bot directories")
    print("=" * 60)


def create_distance_distributions_all_matchups(base_dir, output_dir="arena_heatmaps", chunksize=50000, max_configs=None, skip_initial=0.0, input_format="auto"):
    """
    Create distance distribution plots per bot (averaged across all matchups).
    Saves to {output_dir}/{bot_name}/distance_distribution.png

    Args:
        base_dir: Base simulation directory
        output_dir: Output directory (should be arena_heatmaps folder)
        chunksize: Chunk size for reading CSV/Parquet files
        max_configs: Maximum number of configs to process per matchup
        skip_initial: Skip initial N seconds of data to remove spawn point bias (default: 0.0)
        input_format: "csv", "parquet", or "auto" (default: "auto" prefers parquet)
    """
    # Find all matchup folders
    matchup_folders = [f for f in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, f)) and "_vs_" in f]

    print(f"Found {len(matchup_folders)} matchup folders")

    # Collect data per bot (across all matchups)
    bot_distance_data = {}  # {bot_name: [distance_between_series, distance_from_center_series]}

    # Process each matchup
    for matchup_folder in matchup_folders:
        print("\n" + "=" * 60)
        print(f"Processing {matchup_folder}")
        print("=" * 60)

        # Extract bot names
        parts = matchup_folder.split("_vs_")
        if len(parts) != 2:
            print(f"  Skipping invalid matchup folder name: {matchup_folder}")
            continue

        bot1_name, bot2_name = parts[0], parts[1]

        # Load data for this matchup
        df = load_all_game_data(base_dir, bot1_name, bot2_name, chunksize, max_configs, input_format)

        if df.is_empty():
            print(f"  No data found for {matchup_folder}, skipping...")
            continue

        # Apply skip_initial filter if specified (per game)
        if skip_initial > 0:
            print(f"  ⏩ Skipping initial {skip_initial}s of data per game to remove spawn bias...")
            df = df.filter(
                pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over(["GameIndex", "RoundIndex"]) + skip_initial
            )
            if df.is_empty():
                print(f"  No data remaining after skipping initial {skip_initial}s, skipping matchup...")
                continue
            print(f"  Samples after filter: {len(df):,}")

        # Calculate distance between bots
        print("  Calculating distance between bots...")
        dist_between = calculate_distance_between_bots(df)

        # Calculate distance from center for each bot
        print("  Calculating distance from center...")
        df_with_center_dist = calculate_distance_from_center(df)

        # Split by actor - bot1 is actor 0, bot2 is actor 1
        bot1_center_dist = df_with_center_dist.filter(pl.col("Actor").cast(pl.Int64) == 0)["DistanceFromCenter"]
        bot2_center_dist = df_with_center_dist.filter(pl.col("Actor").cast(pl.Int64) == 1)["DistanceFromCenter"]

        # Store data for each bot
        if bot1_name not in bot_distance_data:
            bot_distance_data[bot1_name] = {"between": [], "from_center": []}
        if bot2_name not in bot_distance_data:
            bot_distance_data[bot2_name] = {"between": [], "from_center": []}

        # Add distance between for both bots (it's the same data)
        bot_distance_data[bot1_name]["between"].append(dist_between["Distance"])
        bot_distance_data[bot2_name]["between"].append(dist_between["Distance"])

        # Add distance from center for each bot
        bot_distance_data[bot1_name]["from_center"].append(bot1_center_dist)
        bot_distance_data[bot2_name]["from_center"].append(bot2_center_dist)

    # Create distance distribution plot for each bot
    for bot_name, data in bot_distance_data.items():
        print("\n" + "=" * 60)
        print(f"Creating distance distribution for {bot_name}...")
        print("=" * 60)

        # Concatenate all data for this bot
        combined_between = pl.concat(data["between"])
        combined_from_center = pl.concat(data["from_center"])

        between_numpy = combined_between.to_numpy()
        from_center_numpy = combined_from_center.to_numpy()

        # Create 2-subplot figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot 1: Distance between bots (averaged across all matchups)
        ax1.hist(between_numpy, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_title(f"Distance Between {bot_name} and Opponents (All Matchups)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Distance Between Bots", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.text(0.98, 0.98, f"n={len(between_numpy):,}",
                transform=ax1.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 2: Distance from center
        ax2.hist(from_center_numpy, bins=30, color='green', edgecolor='black', alpha=0.7)
        ax2.set_title(f"Distance from Center: {bot_name}", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Distance from Center", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Add arena radius reference line
        ax2.axvline(x=arena_radius, color='red', linestyle='--', linewidth=2,
                   label=f'Arena Radius ({arena_radius:.2f})', alpha=0.8)
        ax2.legend(loc='upper right', fontsize=10)

        ax2.text(0.98, 0.98, f"n={len(from_center_numpy):,}",
                transform=ax2.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save to bot's folder
        bot_output_dir = os.path.join(output_dir, bot_name)
        os.makedirs(bot_output_dir, exist_ok=True)
        output_path = os.path.join(bot_output_dir, "distance_distribution.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {output_path}")
        plt.close(fig)

    print("\n" + "=" * 60)
    print(f"✅ Completed! Distance distribution plots saved in bot folders")
    print("=" * 60)

def create_phased_heatmaps_all_bots(base_dir, output_dir="arena_heatmap", actor_position="both", chunksize=50000, max_configs=None, mode="all", use_timer=False, use_time_windows=False, include_distance_over_time=True, skip_initial=0.0, input_format="auto", filter_bots=None, filter_matchups=None, bot_scope="all", cache_dir=None, use_cache_if_exists=False):
    """
    Create heatmaps and position distribution plots for all bots in the simulation directory
    Saves individual phase/timer images for each bot, plus a pooled "All_Bots_Combined"
    aggregate (heatmap, position distribution, distance distributions) across every bot.

    Args:
        base_dir: Base simulation directory
        output_dir: Output directory for heatmaps (default: "arena_heatmap")
        actor_position: "left", "right", or "both"
        chunksize: Chunk size for reading CSV/Parquet files
        max_configs: Maximum number of configs to process per matchup
        mode: What to generate - "heatmap", "position", or "all" (default: "all")
        use_timer: If True, group by Timer values instead of phases
        use_time_windows: If True, group by fixed time windows [0-15s, 15-30s, 30-45s, 45-60s]
        include_distance_over_time: If True, also generate distance over time plot (default: True)
        skip_initial: Skip initial N seconds of data to remove spawn point bias (default: 0.0)
        input_format: "csv", "parquet", or "auto" (default: "auto" prefers parquet)
        filter_bots: Optional list of bot names to process (e.g., ["Bot_BT", "Bot_GA"]). If None, process all bots.
        filter_matchups: Optional list of matchup names to process (e.g., ["Bot_BT_vs_Bot_GA"]). If None, process all matchups.
        bot_scope: "all" (default) generates both per-bot charts and the pooled
            All_Bots_Combined aggregate; "aggregate_only" skips the per-bot loop entirely
            and only generates the pooled aggregate (faster if you only want the overall
            view); "per_bot_only" skips the aggregate and only generates per-bot charts.
        cache_dir: Optional directory to mirror output_dir's structure with a small CSV of
            the raw sample arrays behind each chart (BotPosX/BotPosY, Distance, etc). When
            set, every chart also writes its cache CSV alongside the PNG. Once populated,
            pass this directory to render_charts_from_cache() to redraw all charts (e.g.
            after tweaking titles/labels/colors) without rescanning the simulation data.
        use_cache_if_exists: If True and cache_dir already has a chart's CSV, skip the
            expensive load/compute for that chart and render straight from the cached CSV
            instead - so calling this function again with the same cache_dir is fast rather
            than always re-scanning the simulation data. This is a pure existence check: it
            does NOT verify the cache was produced with the same skip_initial/actor_position/
            use_timer/use_time_windows/etc as the current call. If you change any of those,
            delete the stale cache_dir (or the affected bot subfolders) first, or you'll get
            charts rendered from mismatched cached data with no warning.
    """
    if bot_scope not in ("all", "aggregate_only", "per_bot_only"):
        raise ValueError(f"bot_scope must be 'all', 'aggregate_only', or 'per_bot_only', got {bot_scope!r}")

    # Find all unique bot names from matchup folders
    matchup_folders = [f for f in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, f)) and "_vs_" in f]

    # Apply matchup filter if specified
    if filter_matchups:
        matchup_folders = [m for m in matchup_folders if m in filter_matchups]
        print(f"Filtering to {len(matchup_folders)} matchups: {filter_matchups}")

    if not matchup_folders:
        print("No matchup folders found!")
        return

    bot_names = set()
    for matchup in matchup_folders:
        parts = matchup.split("_vs_")
        if len(parts) == 2:
            bot_names.add(parts[0])
            bot_names.add(parts[1])

    bot_names = sorted(bot_names)

    # Apply bot filter if specified
    if filter_bots:
        bot_names = [b for b in bot_names if b in filter_bots]
        print(f"Filtering to {len(bot_names)} bots: {bot_names}")
    else:
        print(f"Found {len(bot_names)} unique bots: {bot_names}")

    if bot_scope != "aggregate_only" and not bot_names:
        print("No bots to process after filtering!")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each bot
    total_bots = 0 if bot_scope == "aggregate_only" else len(bot_names)
    for bot_idx, bot_name in enumerate(([] if bot_scope == "aggregate_only" else bot_names), start=1):
        print("\n" + "=" * 60)
        print(f"Processing {bot_name} ({bot_idx}/{total_bots}) - {bot_idx/total_bots*100:.1f}% overall")
        print("=" * 60)

        # Create bot-specific directory
        bot_dir = os.path.join(output_dir, bot_name)
        os.makedirs(bot_dir, exist_ok=True)
        cache_bot_dir = os.path.join(cache_dir, bot_name) if cache_dir else None

        # Generate heatmaps if requested
        if mode in ["heatmap", "all"]:
            # Check timer cache once
            cached_timers = _cached_files(cache_bot_dir, "timer_*.csv") if (use_timer and use_cache_if_exists) else []

            if use_timer and cached_timers:
                print(f"\n♻️  use_cache_if_exists=True: rendering {bot_name} timer heatmaps from {len(cached_timers)} cached CSV(s) instead of reloading...")
                for cache_idx, cache_path in enumerate(cached_timers, start=1):
                    print(f"  📊 Rendering timer {cache_idx}/{len(cached_timers)} from cache...", end='\r')
                    output_path = os.path.join(bot_dir, f"{Path(cache_path).stem}.png")
                    _render_single_cached_chart(cache_path, bot_name, output_path, actor_position)
                print(f"  ✅ Rendered {len(cached_timers)} timer heatmaps from cache" + " " * 20)

                if include_distance_over_time:
                    for stem in ("distance_over_time", "distance_histogram", "distance_from_center_histogram"):
                        cp = os.path.join(cache_bot_dir, f"{stem}.csv")
                        if os.path.exists(cp):
                            output_path = os.path.join(bot_dir, f"{stem}.png")
                            _render_single_cached_chart(cp, bot_name, output_path, actor_position)

            elif use_timer:
                # Timer-based mode - load data with distance if needed
                print("\n📥 Loading data grouped by Timer...")
                if include_distance_over_time:
                    timer_data, distance_data = load_bot_data_from_simulation(
                        base_dir, bot_name, actor_position, chunksize, max_configs,
                        group_by_timer=True, also_load_distance=True
                    )
                else:
                    timer_data = load_bot_data_from_simulation(
                        base_dir, bot_name, actor_position, chunksize, max_configs,
                        group_by_timer=True
                    )

                if not timer_data:
                    print(f"No data found for {bot_name}, skipping...")
                    continue

                # Apply skip_initial filter if specified
                if skip_initial > 0:
                    print(f"\n⏩ Skipping initial {skip_initial}s of data per game to remove spawn bias...")
                    filtered_timer_data = {}
                    for timer, df in timer_data.items():
                        # Filter out data where UpdatedAt < (min_UpdatedAt_for_that_game + skip_initial) per game
                        df_filtered = df.filter(
                            pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over(["GameIndex", "RoundIndex"]) + skip_initial
                        )
                        if not df_filtered.is_empty():
                            filtered_timer_data[timer] = df_filtered
                            print(f"  Timer {timer}: {len(df):,} -> {len(df_filtered):,} samples")
                    timer_data = filtered_timer_data

                # Create plots for each timer value
                for timer in sorted(timer_data.keys()):
                    df = timer_data[timer]
                    print(f"\nProcessing Timer={timer}...")
                    print(f"  Samples: {len(df):,}")
                    print(f"  Time range: {df['UpdatedAt'].min():.2f} - {df['UpdatedAt'].max():.2f}")

                    label = f"Timer {int(timer)}s" if timer == int(timer) else f"Timer {timer}s"
                    timer_str = f"{int(timer)}" if timer == int(timer) else f"{timer}"
                    cache_path = os.path.join(cache_bot_dir, f"timer_{timer_str}.csv") if cache_bot_dir else None
                    fig = plot_joint_heatmap_with_distributions(df, label, bot_name, actor_position, cache_path=cache_path)

                    if fig is not None:
                        # Save with timer in filename
                        output_path = os.path.join(bot_dir, f"timer_{timer_str}.png")
                        plt.savefig(output_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved to {output_path}")
                        plt.close(fig)

                # Generate distance plots if requested and data is available
                if include_distance_over_time and distance_data:
                    print(f"\nGenerating distance over time plot...")
                    output_path = os.path.join(bot_dir, "distance_over_time.png")
                    cache_path = os.path.join(cache_bot_dir, "distance_over_time.csv") if cache_bot_dir else None
                    fig = plot_distance_over_time_from_data(distance_data, bot_name, output_path, cache_path=cache_path)
                    if fig is not None:
                        plt.close(fig)

                    print(f"Generating distance histogram...")
                    output_path = os.path.join(bot_dir, "distance_histogram.png")
                    cache_path = os.path.join(cache_bot_dir, "distance_histogram.csv") if cache_bot_dir else None
                    fig = plot_distance_histogram_from_data(distance_data, bot_name, output_path, cache_path=cache_path)
                    if fig is not None:
                        plt.close(fig)

                    print(f"Generating distance from center histogram...")
                    output_path = os.path.join(bot_dir, "distance_from_center_histogram.png")
                    cache_path = os.path.join(cache_bot_dir, "distance_from_center_histogram.csv") if cache_bot_dir else None
                    fig = plot_distance_from_center_histogram(timer_data, bot_name, output_path, cache_path=cache_path)
                    if fig is not None:
                        plt.close(fig)

            # Check time windows cache once
            cached_windows = _cached_files(cache_bot_dir, "window_*.csv") if (use_time_windows and use_cache_if_exists) else []

            if use_time_windows and cached_windows:
                print(f"\n♻️  use_cache_if_exists=True: rendering {bot_name} time-window heatmaps from {len(cached_windows)} cached CSV(s) instead of reloading...")
                for cache_idx, cache_path in enumerate(cached_windows, start=1):
                    print(f"  📊 Rendering window {cache_idx}/{len(cached_windows)} from cache...", end='\r')
                    output_path = os.path.join(bot_dir, f"{Path(cache_path).stem}.png")
                    _render_single_cached_chart(cache_path, bot_name, output_path, actor_position)
                print(f"  ✅ Rendered {len(cached_windows)} time-window heatmaps from cache" + " " * 20)

            elif use_time_windows:
                # Time window mode - fixed time windows [0-15s, 15-30s, 30-45s, 45-60s]
                print("\n📥 Loading all data for time window grouping...")
                df_combined = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs, group_by_timer=False)
                if not df_combined.is_empty():
                    print(f"  ✅ Loaded {len(df_combined):,} position samples")

                if df_combined.is_empty():
                    print(f"No data found for {bot_name}, skipping...")
                    continue

                # Apply skip_initial filter if specified (per game)
                if skip_initial > 0:
                    print(f"\n⏩ Skipping initial {skip_initial}s of data per game to remove spawn bias...")
                    original_count = len(df_combined)
                    df_combined = df_combined.filter(
                        pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over(["GameIndex", "RoundIndex"]) + skip_initial
                    )
                    print(f"  Filtered: {original_count:,} -> {len(df_combined):,} samples")

                    if df_combined.is_empty():
                        print(f"No data remaining after filtering for {bot_name}, skipping...")
                        continue

                # Define time windows: [0-15s], [15-30s], [30-45s], [45-60s]
                time_windows = [
                    (skip_initial, 15, f"{skip_initial}-15s") if skip_initial > 0 else (0, 15, "0-15s"),
                    (15, 30, "15-30s"),
                    (30, 45, "30-45s"),
                    (45, 60, "45-60s")
                ]

                print(f"\nSplitting data into fixed time windows...")
                # Create plots for each time window
                for start, end, window_name in time_windows:
                    # Filter data for this time window
                    window_df = df_combined.filter(
                        (pl.col("UpdatedAt") >= start) & (pl.col("UpdatedAt") < end)
                    )

                    if window_df.is_empty():
                        print(f"  No data for {window_name}, skipping...")
                        continue

                    print(f"\nProcessing {window_name}...")
                    print(f"  Samples: {len(window_df):,}")
                    print(f"  Time range: {window_df['UpdatedAt'].min():.2f} - {window_df['UpdatedAt'].max():.2f}")

                    # Create joint plot
                    cache_path = os.path.join(cache_bot_dir, f"window_{start}-{end}s.csv") if cache_bot_dir else None
                    fig = plot_joint_heatmap_with_distributions(window_df, window_name, bot_name, actor_position, cache_path=cache_path)

                    if fig is not None:
                        # Save with window name in filename
                        output_path = os.path.join(bot_dir, f"window_{start}-{end}s.png")
                        plt.savefig(output_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved to {output_path}")
                        plt.close(fig)

            # Check phase cache once
            cached_phases = _cached_files(cache_bot_dir, "[0-2].csv") if use_cache_if_exists else []

            if cached_phases:
                print(f"\n♻️  use_cache_if_exists=True: rendering {bot_name} phase heatmaps from {len(cached_phases)} cached CSV(s) instead of reloading...")
                for cache_idx, cache_path in enumerate(cached_phases, start=1):
                    print(f"  📊 Rendering phase {cache_idx}/{len(cached_phases)} from cache...", end='\r')
                    output_path = os.path.join(bot_dir, f"{Path(cache_path).stem}.png")
                    _render_single_cached_chart(cache_path, bot_name, output_path, actor_position)
                print(f"  ✅ Rendered {len(cached_phases)} phase heatmaps from cache" + " " * 20)

            else:
                # Phase-based mode (original)
                print("\n📥 Loading all data...")
                df_combined = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs, group_by_timer=False)
                if not df_combined.is_empty():
                    print(f"  ✅ Loaded {len(df_combined):,} position samples")

                if df_combined.is_empty():
                    print(f"No data found for {bot_name}, skipping...")
                    continue

                # Apply skip_initial filter if specified (per game)
                if skip_initial > 0:
                    print(f"\n⏩ Skipping initial {skip_initial}s of data per game to remove spawn bias...")
                    original_count = len(df_combined)
                    df_combined = df_combined.filter(
                        pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over(["GameIndex", "RoundIndex"]) + skip_initial
                    )
                    print(f"  Filtered: {original_count:,} -> {len(df_combined):,} samples")

                    if df_combined.is_empty():
                        print(f"No data remaining after filtering for {bot_name}, skipping...")
                        continue

                print(f"Time range: {df_combined['UpdatedAt'].min():.2f} - {df_combined['UpdatedAt'].max():.2f}")

                # Split into phases
                print("\nSplitting into phases...")
                phases = split_into_phases(df_combined, num_phases=3)
                phase_names = ["Early Game", "Mid Game", "Late Game"]

                # Create and save individual heatmaps for each phase
                for idx, (phase_df, phase_name) in enumerate(zip(phases, phase_names)):
                    print(f"Creating {phase_name} joint heatmap with marginal distributions...")

                    if phase_df.is_empty():
                        print(f"  No data for {phase_name}, skipping...")
                        continue

                    # Create joint plot with marginal distributions
                    cache_path = os.path.join(cache_bot_dir, f"{idx}.csv") if cache_bot_dir else None
                    fig = plot_joint_heatmap_with_distributions(phase_df, phase_name, bot_name, actor_position, cache_path=cache_path)

                    if fig is not None:
                        # Save
                        output_path = os.path.join(bot_dir, f"{idx}.png")
                        plt.savefig(output_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved to {output_path}")
                        plt.close(fig)

        # Generate position distribution if requested
        if mode in ["position", "all"]:
            position_cache_path = os.path.join(cache_bot_dir, "position_distribution.csv") if cache_bot_dir else None

            if use_cache_if_exists and position_cache_path and os.path.exists(position_cache_path):
                print(f"\n♻️  use_cache_if_exists=True: rendering {bot_name} position distribution from cached CSV...")
                dist_path = os.path.join(bot_dir, "position_distribution.png")
                _render_single_cached_chart(position_cache_path, bot_name, dist_path, actor_position)
                print(f"  ✅ Rendered position distribution from cache")
            else:
                # Load combined data if not already loaded (needed for position distribution).
                # Also reload if df_combined never got set - e.g. the phase-mode heatmap
                # section above was skipped via use_cache_if_exists, or mode="position" ran
                # without the heatmap section ever executing.
                if use_timer or use_time_windows or 'df_combined' not in locals():
                    print("\nLoading combined data for position distribution...")
                    df_combined = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs, group_by_timer=False)

                    # Apply skip_initial filter if specified (per game)
                    if skip_initial > 0 and not df_combined.is_empty():
                        print(f"\n⏩ Skipping initial {skip_initial}s of data per game to remove spawn bias...")
                        original_count = len(df_combined)
                        df_combined = df_combined.filter(
                            pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over(["GameIndex", "RoundIndex"]) + skip_initial
                        )
                        print(f"  Filtered: {original_count:,} -> {len(df_combined):,} samples")

                # Check if we have data
                if 'df_combined' in locals() and not df_combined.is_empty():
                    # Create position distribution plot
                    print(f"Creating position distribution plot...")
                    fig_dist = plot_position_distribution(df_combined, bot_name, actor_position, cache_path=position_cache_path)

                    if fig_dist is not None:
                        dist_path = os.path.join(bot_dir, "position_distribution.png")
                        fig_dist.savefig(dist_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved to {dist_path}")
                        plt.close(fig_dist)
                else:
                    print(f"No data available for position distribution")

    # ========== Generate aggregate "All Bots" heatmap + position distribution ==========
    # Pools every bot's position samples together (bot identity dropped), so it needs its
    # own data load via load_all_bots_data_from_simulation rather than reusing per-bot data.
    if bot_scope != "per_bot_only" and mode in ["heatmap", "position", "all"]:
        print("\n" + "=" * 60)
        print("Processing All Bots (pooled across every bot)")
        print("=" * 60)

        agg_label = "All Bots"
        agg_dir = os.path.join(output_dir, "All_Bots_Combined")
        os.makedirs(agg_dir, exist_ok=True)
        agg_cache_dir = os.path.join(cache_dir, "All_Bots_Combined") if cache_dir else None

        agg_df_combined = None

        if mode in ["heatmap", "all"]:
            # Check aggregate timer cache once
            agg_cached_timers = _cached_files(agg_cache_dir, "timer_*.csv") if (use_timer and use_cache_if_exists) else []

            if use_timer and agg_cached_timers:
                print(f"\n♻️  use_cache_if_exists=True: rendering pooled timer heatmaps from {len(agg_cached_timers)} cached CSV(s)...")
                for cache_idx, cache_path in enumerate(agg_cached_timers, start=1):
                    print(f"  📊 Rendering aggregate timer {cache_idx}/{len(agg_cached_timers)} from cache...", end='\r')
                    output_path = os.path.join(agg_dir, f"{Path(cache_path).stem}.png")
                    _render_single_cached_chart(cache_path, agg_label, output_path, actor_position)
                print(f"  ✅ Rendered {len(agg_cached_timers)} aggregate timer heatmaps from cache" + " " * 20)

                if include_distance_over_time:
                    for stem in ("distance_over_time", "distance_histogram", "distance_from_center_histogram"):
                        cp = os.path.join(agg_cache_dir, f"{stem}.csv")
                        if os.path.exists(cp):
                            output_path = os.path.join(agg_dir, f"{stem}.png")
                            _render_single_cached_chart(cp, agg_label, output_path, actor_position)

            elif use_timer:
                print("\n📥 Loading pooled data grouped by Timer (all bots)...")
                if include_distance_over_time:
                    agg_timer_data, agg_distance_data = load_all_bots_data_from_simulation(
                        base_dir, chunksize, max_configs, group_by_timer=True,
                        also_load_distance=True, input_format=input_format, filter_matchups=filter_matchups
                    )
                else:
                    agg_timer_data = load_all_bots_data_from_simulation(
                        base_dir, chunksize, max_configs, group_by_timer=True,
                        input_format=input_format, filter_matchups=filter_matchups
                    )
                    agg_distance_data = None

                if not agg_timer_data:
                    print("No pooled data found, skipping aggregate heatmap.")
                else:
                    if skip_initial > 0:
                        print(f"\n⏩ Skipping initial {skip_initial}s of data per round to remove spawn bias...")
                        filtered = {}
                        for timer, df in agg_timer_data.items():
                            df_filtered = df.filter(
                                pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over(["GameIndex", "RoundIndex"]) + skip_initial
                            )
                            if not df_filtered.is_empty():
                                filtered[timer] = df_filtered
                        agg_timer_data = filtered

                    for timer in sorted(agg_timer_data.keys()):
                        df = agg_timer_data[timer]
                        label = f"Timer {int(timer)}s" if timer == int(timer) else f"Timer {timer}s"
                        timer_str = f"{int(timer)}" if timer == int(timer) else f"{timer}"
                        cache_path = os.path.join(agg_cache_dir, f"timer_{timer_str}.csv") if agg_cache_dir else None
                        fig = plot_joint_heatmap_with_distributions(df, label, agg_label, actor_position, cache_path=cache_path)
                        if fig is not None:
                            output_path = os.path.join(agg_dir, f"timer_{timer_str}.png")
                            plt.savefig(output_path, dpi=150, bbox_inches='tight')
                            print(f"  Saved to {output_path}")
                            plt.close(fig)

                    if include_distance_over_time and agg_distance_data:
                        print("\nGenerating pooled distance over time plot...")
                        cache_path = os.path.join(agg_cache_dir, "distance_over_time.csv") if agg_cache_dir else None
                        fig = plot_distance_over_time_from_data(agg_distance_data, agg_label, os.path.join(agg_dir, "distance_over_time.png"), cache_path=cache_path)
                        if fig is not None:
                            plt.close(fig)

                        print("Generating pooled distance histogram...")
                        cache_path = os.path.join(agg_cache_dir, "distance_histogram.csv") if agg_cache_dir else None
                        fig = plot_distance_histogram_from_data(agg_distance_data, agg_label, os.path.join(agg_dir, "distance_histogram.png"), cache_path=cache_path)
                        if fig is not None:
                            plt.close(fig)

                        print("Generating pooled distance from center histogram...")
                        cache_path = os.path.join(agg_cache_dir, "distance_from_center_histogram.csv") if agg_cache_dir else None
                        fig = plot_distance_from_center_histogram(agg_timer_data, agg_label, os.path.join(agg_dir, "distance_from_center_histogram.png"), cache_path=cache_path)
                        if fig is not None:
                            plt.close(fig)

            # Check aggregate time windows cache once
            agg_cached_windows = _cached_files(agg_cache_dir, "window_*.csv") if (use_time_windows and use_cache_if_exists) else []

            if use_time_windows and agg_cached_windows:
                print(f"\n♻️  use_cache_if_exists=True: rendering pooled time-window heatmaps from {len(agg_cached_windows)} cached CSV(s)...")
                for cache_idx, cache_path in enumerate(agg_cached_windows, start=1):
                    print(f"  📊 Rendering aggregate window {cache_idx}/{len(agg_cached_windows)} from cache...", end='\r')
                    output_path = os.path.join(agg_dir, f"{Path(cache_path).stem}.png")
                    _render_single_cached_chart(cache_path, agg_label, output_path, actor_position)
                print(f"  ✅ Rendered {len(agg_cached_windows)} aggregate time-window heatmaps from cache" + " " * 20)

            elif use_time_windows:
                print("\n📥 Loading pooled data for time window grouping (all bots)...")
                agg_df_combined = load_all_bots_data_from_simulation(
                    base_dir, chunksize, max_configs, group_by_timer=False,
                    input_format=input_format, filter_matchups=filter_matchups
                )

                if agg_df_combined.is_empty():
                    print("No pooled data found, skipping aggregate heatmap.")
                else:
                    if skip_initial > 0:
                        agg_df_combined = agg_df_combined.filter(
                            pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over(["GameIndex", "RoundIndex"]) + skip_initial
                        )

                    time_windows = [
                        (skip_initial, 15, f"{skip_initial}-15s") if skip_initial > 0 else (0, 15, "0-15s"),
                        (15, 30, "15-30s"),
                        (30, 45, "30-45s"),
                        (45, 60, "45-60s")
                    ]
                    for start, end, window_name in time_windows:
                        window_df = agg_df_combined.filter((pl.col("UpdatedAt") >= start) & (pl.col("UpdatedAt") < end))
                        if window_df.is_empty():
                            continue
                        cache_path = os.path.join(agg_cache_dir, f"window_{start}-{end}s.csv") if agg_cache_dir else None
                        fig = plot_joint_heatmap_with_distributions(window_df, window_name, agg_label, actor_position, cache_path=cache_path)
                        if fig is not None:
                            output_path = os.path.join(agg_dir, f"window_{start}-{end}s.png")
                            plt.savefig(output_path, dpi=150, bbox_inches='tight')
                            print(f"  Saved to {output_path}")
                            plt.close(fig)

            # Check aggregate phase cache once
            agg_cached_phases = _cached_files(agg_cache_dir, "[0-2].csv") if use_cache_if_exists else []

            if agg_cached_phases:
                print(f"\n♻️  use_cache_if_exists=True: rendering pooled phase heatmaps from {len(agg_cached_phases)} cached CSV(s)...")
                for cache_idx, cache_path in enumerate(agg_cached_phases, start=1):
                    print(f"  📊 Rendering aggregate phase {cache_idx}/{len(agg_cached_phases)} from cache...", end='\r')
                    output_path = os.path.join(agg_dir, f"{Path(cache_path).stem}.png")
                    _render_single_cached_chart(cache_path, agg_label, output_path, actor_position)
                print(f"  ✅ Rendered {len(agg_cached_phases)} aggregate phase heatmaps from cache" + " " * 20)

            else:
                print("\n📥 Loading pooled data (all bots)...")
                agg_df_combined = load_all_bots_data_from_simulation(
                    base_dir, chunksize, max_configs, group_by_timer=False,
                    input_format=input_format, filter_matchups=filter_matchups
                )

                if agg_df_combined.is_empty():
                    print("No pooled data found, skipping aggregate heatmap.")
                else:
                    if skip_initial > 0:
                        agg_df_combined = agg_df_combined.filter(
                            pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over(["GameIndex", "RoundIndex"]) + skip_initial
                        )

                    print("\nSplitting pooled data into phases...")
                    phases = split_into_phases(agg_df_combined, num_phases=3)
                    phase_names = ["Early Game", "Mid Game", "Late Game"]
                    for idx, (phase_df, phase_name) in enumerate(zip(phases, phase_names)):
                        if phase_df.is_empty():
                            continue
                        cache_path = os.path.join(agg_cache_dir, f"{idx}.csv") if agg_cache_dir else None
                        fig = plot_joint_heatmap_with_distributions(phase_df, phase_name, agg_label, actor_position, cache_path=cache_path)
                        if fig is not None:
                            output_path = os.path.join(agg_dir, f"{idx}.png")
                            plt.savefig(output_path, dpi=150, bbox_inches='tight')
                            print(f"  Saved to {output_path}")
                            plt.close(fig)

        if mode in ["position", "all"]:
            agg_position_cache_path = os.path.join(agg_cache_dir, "position_distribution.csv") if agg_cache_dir else None

            if use_cache_if_exists and agg_position_cache_path and os.path.exists(agg_position_cache_path):
                print(f"\n♻️  use_cache_if_exists=True: rendering pooled position distribution from cached CSV instead of reloading...")
                dist_path = os.path.join(agg_dir, "position_distribution.png")
                _render_single_cached_chart(agg_position_cache_path, agg_label, dist_path, actor_position)

            else:
                if agg_df_combined is None or agg_df_combined.is_empty():
                    print("\nLoading pooled data for position distribution...")
                    agg_df_combined = load_all_bots_data_from_simulation(
                        base_dir, chunksize, max_configs, group_by_timer=False,
                        input_format=input_format, filter_matchups=filter_matchups
                    )
                    if not agg_df_combined.is_empty() and skip_initial > 0:
                        agg_df_combined = agg_df_combined.filter(
                            pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over(["GameIndex", "RoundIndex"]) + skip_initial
                        )

                if not agg_df_combined.is_empty():
                    print("Creating pooled position distribution plot...")
                    fig_dist = plot_position_distribution(agg_df_combined, agg_label, actor_position, cache_path=agg_position_cache_path)
                    if fig_dist is not None:
                        dist_path = os.path.join(agg_dir, "position_distribution.png")
                        fig_dist.savefig(dist_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved to {dist_path}")
                        plt.close(fig_dist)
                else:
                    print("No pooled data available for position distribution")

    # ========== Generate distance distributions per bot ==========
    print("\n" + "=" * 60)
    print("Generating distance distributions for each bot (across all matchups)...")
    print("=" * 60)

    # This section computes every bot's distance_distribution.csv in a single pass over
    # matchup_folders (data isn't per-bot separable up front), so the cache check needs to
    # confirm ALL expected outputs already exist before it can skip that pass entirely.
    required_distance_dist_caches = []
    if cache_dir:
        if bot_scope != "aggregate_only":
            required_distance_dist_caches += [os.path.join(cache_dir, b, "distance_distribution.csv") for b in bot_names]
        if bot_scope != "per_bot_only":
            required_distance_dist_caches.append(os.path.join(cache_dir, "All_Bots_Combined", "distance_distribution.csv"))

    if use_cache_if_exists and required_distance_dist_caches and all(os.path.exists(p) for p in required_distance_dist_caches):
        print(f"\n♻️  use_cache_if_exists=True: rendering {len(required_distance_dist_caches)} distance_distribution chart(s) from cache...")
        for cache_idx, cache_path in enumerate(required_distance_dist_caches, start=1):
            print(f"  📊 Rendering distance distribution {cache_idx}/{len(required_distance_dist_caches)} from cache...", end='\r')
            cached_bot_name = os.path.basename(os.path.dirname(cache_path))
            name = "All Bots" if cached_bot_name == "All_Bots_Combined" else cached_bot_name
            output_path = os.path.join(output_dir, cached_bot_name, "distance_distribution.png")
            _render_single_cached_chart(cache_path, name, output_path, actor_position)
        print(f"  ✅ Rendered {len(required_distance_dist_caches)} distance distributions from cache" + " " * 20)

    else:
        # Collect data per bot (across all matchups)
        bot_distance_data = {}  # {bot_name: [distance_between_series, distance_from_center_series]}
        # Distance-between is symmetric and gets appended to BOTH bot1's and bot2's entries in
        # bot_distance_data above, so pooling across bot_distance_data would double-count every
        # matchup's series. Collect it once per matchup here instead, for the aggregate below.
        all_between_distances = []

        # Process each matchup
        for matchup_folder in matchup_folders:
            print("\n" + "=" * 60)
            print(f"Processing matchup: {matchup_folder}")
            print("=" * 60)

            # Extract bot names
            parts = matchup_folder.split("_vs_")
            if len(parts) != 2:
                print(f"  Skipping invalid matchup folder name: {matchup_folder}")
                continue

            bot1_name, bot2_name = parts[0], parts[1]

            # Load data for this matchup
            df = load_all_game_data(base_dir, bot1_name, bot2_name, chunksize, max_configs, input_format)

            if df.is_empty():
                print(f"  No data found for {matchup_folder}, skipping...")
                continue

            # Apply skip_initial filter if specified (per game)
            if skip_initial > 0:
                print(f"  ⏩ Skipping initial {skip_initial}s of data per game to remove spawn bias...")
                df = df.filter(
                    pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over(["GameIndex", "RoundIndex"]) + skip_initial
                )
                if df.is_empty():
                    print(f"  No data remaining after skipping initial {skip_initial}s, skipping matchup...")
                    continue
                print(f"  Samples after filter: {len(df):,}")

            # Calculate distance between bots
            print("  Calculating distance between bots...")
            dist_between = calculate_distance_between_bots(df)
            all_between_distances.append(dist_between["Distance"])

            # Calculate distance from center for each bot
            print("  Calculating distance from center...")
            df_with_center_dist = calculate_distance_from_center(df)

            # Split by actor - bot1 is actor 0, bot2 is actor 1
            bot1_center_dist = df_with_center_dist.filter(pl.col("Actor").cast(pl.Int64) == 0)["DistanceFromCenter"]
            bot2_center_dist = df_with_center_dist.filter(pl.col("Actor").cast(pl.Int64) == 1)["DistanceFromCenter"]

            # Store data for each bot
            if bot1_name not in bot_distance_data:
                bot_distance_data[bot1_name] = {"between": [], "from_center": []}
            if bot2_name not in bot_distance_data:
                bot_distance_data[bot2_name] = {"between": [], "from_center": []}

            # Add distance between for both bots (it's the same data)
            bot_distance_data[bot1_name]["between"].append(dist_between["Distance"])
            bot_distance_data[bot2_name]["between"].append(dist_between["Distance"])

            # Add distance from center for each bot
            bot_distance_data[bot1_name]["from_center"].append(bot1_center_dist)
            bot_distance_data[bot2_name]["from_center"].append(bot2_center_dist)

        # Create distance distribution plot for each bot
        bot_dist_list = list(({} if bot_scope == "aggregate_only" else bot_distance_data).items())
        total_bot_dists = len(bot_dist_list)
        for bot_dist_idx, (bot_name, data) in enumerate(bot_dist_list, start=1):
            print(f"\n  📊 Creating distance distribution for {bot_name} ({bot_dist_idx}/{total_bot_dists})...")

            # Concatenate all data for this bot
            combined_between = pl.concat(data["between"])
            combined_from_center = pl.concat(data["from_center"])

            # Save to bot's folder
            bot_output_dir = os.path.join(output_dir, bot_name)
            os.makedirs(bot_output_dir, exist_ok=True)
            output_path = os.path.join(bot_output_dir, "distance_distribution.png")
            cache_bot_dir = os.path.join(cache_dir, bot_name) if cache_dir else None
            cache_path = os.path.join(cache_bot_dir, "distance_distribution.csv") if cache_bot_dir else None

            fig = plot_distance_distribution(combined_between, combined_from_center, bot_name, output_path, cache_path=cache_path)
            if fig is not None:
                plt.close(fig)

        # Create pooled distance distribution across all bots
        if bot_scope != "per_bot_only" and all_between_distances and bot_distance_data:
            print(f"\n  📊 Creating pooled distance distribution for All Bots Combined...")

            combined_between = pl.concat(all_between_distances)
            combined_from_center = pl.concat([
                series for data in bot_distance_data.values() for series in data["from_center"]
            ])

            agg_output_dir = os.path.join(output_dir, "All_Bots_Combined")
            os.makedirs(agg_output_dir, exist_ok=True)
            output_path = os.path.join(agg_output_dir, "distance_distribution.png")
            agg_cache_dir = os.path.join(cache_dir, "All_Bots_Combined") if cache_dir else None
            cache_path = os.path.join(agg_cache_dir, "distance_distribution.csv") if agg_cache_dir else None

            fig = plot_distance_distribution(combined_between, combined_from_center, "All Bots", output_path, cache_path=cache_path, is_pooled=True)
            if fig is not None:
                plt.close(fig)

    print("\n" + "=" * 60)
    print(f"✅ COMPLETED! All visualizations saved to: {output_dir}")
    print("=" * 60)
    if bot_scope != "aggregate_only":
        print(f"  📊 Processed {total_bots} bots")
    if bot_scope != "per_bot_only":
        print(f"  📊 Generated aggregate 'All Bots Combined' charts")
    print("=" * 60)


_PHASE_NAMES = ["Early Game", "Mid Game", "Late Game"]


def _render_single_cached_chart(cache_path, name, output_path, actor_position="both"):
    """
    Dispatch a single cache CSV (named by create_phased_heatmaps_all_bots' cache_path
    convention: timer_*.csv, window_*.csv, {0,1,2}.csv, position_distribution.csv,
    distance_histogram.csv, distance_from_center_histogram.csv, distance_over_time.csv,
    distance_distribution.csv) to the matching *_from_cache render function and save the
    result to output_path. Shared by render_charts_from_cache() and
    create_phased_heatmaps_all_bots(use_cache_if_exists=True).

    Args:
        cache_path: Path to the cached CSV
        name: Bot name, or "All Bots" for the pooled aggregate (used in chart titles)
        output_path: Where to save the regenerated PNG
        actor_position: Position filter text to show in heatmap/position chart titles

    Returns:
        True if a chart was rendered and saved, False if the filename wasn't recognized
        or the cached data was empty.
    """
    stem = Path(cache_path).stem
    is_pooled = os.path.basename(os.path.dirname(cache_path)) == "All_Bots_Combined"
    needs_manual_save = False

    if stem.startswith("timer_"):
        label = f"Timer {stem[len('timer_'):]}s"
        fig = plot_joint_heatmap_from_cache(cache_path, label, name, actor_position)
        needs_manual_save = True
    elif stem.startswith("window_"):
        label = stem[len("window_"):]
        fig = plot_joint_heatmap_from_cache(cache_path, label, name, actor_position)
        needs_manual_save = True
    elif stem in ("0", "1", "2"):
        fig = plot_joint_heatmap_from_cache(cache_path, _PHASE_NAMES[int(stem)], name, actor_position)
        needs_manual_save = True
    elif stem == "position_distribution":
        fig = plot_position_distribution_from_cache(cache_path, name, actor_position)
        needs_manual_save = True
    elif stem == "distance_histogram":
        fig = plot_distance_histogram_from_cache(cache_path, name, output_path)
    elif stem == "distance_from_center_histogram":
        fig = plot_distance_from_center_histogram_from_cache(cache_path, name, output_path)
    elif stem == "distance_over_time":
        fig = plot_distance_over_time_from_cache(cache_path, name, output_path)
    elif stem == "distance_distribution":
        fig = plot_distance_distribution_from_cache(cache_path, name, output_path, is_pooled=is_pooled)
    else:
        print(f"  Skipping unrecognized cache file: {cache_path}")
        return False

    if fig is None:
        return False

    if needs_manual_save:
        # The heatmap/position variants return a figure without saving it themselves;
        # the distance_* variants above already saved via output_path.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {output_path}")
    plt.close(fig)
    return True


def render_charts_from_cache(cache_dir, output_dir, actor_position="both", filter_bots=None):
    """
    Regenerate chart PNGs from the CSVs written by
    create_phased_heatmaps_all_bots(cache_dir=...), without touching the raw simulation
    data. Use this to iterate on chart design (titles, labels, colors, bins) - edit the
    _render_* functions in this file, then rerun this instead of the full pipeline.

    Args:
        cache_dir: Directory previously passed as cache_dir to create_phased_heatmaps_all_bots
        output_dir: Where to write the regenerated PNGs (mirrors cache_dir's bot-folder structure)
        actor_position: Position filter text to show in chart titles ("left"/"right"/"both")
        filter_bots: Optional list of cache subfolder names to restrict to (e.g. ["Bot_BT"]).
            Subfolder names include "All_Bots_Combined" for the pooled aggregate.

    Returns:
        Number of charts regenerated.
    """
    if not os.path.isdir(cache_dir):
        print(f"Cache directory not found: {cache_dir}")
        return 0

    bot_dirs = sorted(d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d)))
    if filter_bots:
        bot_dirs = [d for d in bot_dirs if d in filter_bots]

    count = 0
    for bot_name in bot_dirs:
        cache_bot_dir = os.path.join(cache_dir, bot_name)
        out_bot_dir = os.path.join(output_dir, bot_name)
        name = "All Bots" if bot_name == "All_Bots_Combined" else bot_name

        for fname in sorted(os.listdir(cache_bot_dir)):
            if not fname.endswith(".csv"):
                continue
            cache_path = os.path.join(cache_bot_dir, fname)
            output_path = os.path.join(out_bot_dir, f"{Path(fname).stem}.png")
            if _render_single_cached_chart(cache_path, name, output_path, actor_position):
                count += 1

    print(f"\n✅ Regenerated {count} charts from cache into {output_dir}")
    return count


if __name__ == "__main__":
    default_base_dir = "/Users/user_name/Library/Application Support/DefaultCompany/Sumobot/Simulation"
    parser = argparse.ArgumentParser(
        description="Create phased heatmap visualizations for sumobot arena data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single CSV file
  python detailed_analyzer.py single game_log.csv -o output.png

  # Single bot analysis (phase-based)
  python detailed_analyzer.py bot Bot_BT

  # Single bot analysis (timer-based, groups by Timer from config)
  python detailed_analyzer.py bot Bot_BT --use-timer

  # Generate ALL visualizations for all bots (heatmaps + position plots)
  python detailed_analyzer.py all

  # Generate only heatmaps for all bots (phase-based)
  python detailed_analyzer.py all heatmap

  # Generate only heatmaps for all bots (timer-based)
  python detailed_analyzer.py all heatmap --use-timer

  # Generate only position distribution plots for all bots
  python detailed_analyzer.py all position

  # All visualizations with custom path and limited configs
  python detailed_analyzer.py all all "/custom/path" --max-configs 10

  # Generate distance distribution plots for all matchups
  python detailed_analyzer.py distance

  # Distance distributions with custom path and limited configs
  python detailed_analyzer.py distance "/custom/path" -o distance_output --max-configs 5

  # Generate distance over time plots (grouped by Timer) for all matchups
  python detailed_analyzer.py distance-time

  # Distance over time with custom path
  python detailed_analyzer.py distance-time "/custom/path" -o distance_time_output

  # Run ALL analyses at once (heatmaps, position distributions, distance plots)
  python detailed_analyzer.py all

  # All analyses with Timer grouping and skip initial 0.5s spawn data
  python detailed_analyzer.py all --use-timer --skip-initial=0.5

  # All analyses with fixed time windows [0-15s, 15-30s, 30-45s, 45-60s]
  python detailed_analyzer.py all --use-time-windows

  # Run all analyses with timer-based grouping
  python detailed_analyzer.py all --use-timer

  # Test mode: process only 1 config per matchup (default)
  python detailed_analyzer.py all --test --use-timer

  # Test mode: process 5 configs per matchup
  python detailed_analyzer.py all --test=5 --use-timer

  # All analyses, also caching each chart's raw sample data as CSV for fast re-rendering
  python detailed_analyzer.py all --cache-dir arena_heatmaps_cache

  # Redraw all charts from that cache (e.g. after tweaking a title/label/color in the
  # _render_* functions) without rescanning the (potentially huge) simulation data
  python detailed_analyzer.py render-cache arena_heatmaps_cache -o arena_heatmaps_v2
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Analysis mode")

    # Single file mode
    single_parser = subparsers.add_parser("single", help="Analyze a single CSV file")
    single_parser.add_argument("csv_path", help="Path to CSV file")
    single_parser.add_argument("-o", "--output", help="Output path for the image")
    single_parser.add_argument("-c", "--chunksize", type=int, default=50000,
                              help="Chunk size for reading CSV (default: 50000)")

    # Bot analysis mode
    bot_parser = subparsers.add_parser("bot", help="Analyze a specific bot from simulation directory")
    bot_parser.add_argument("bot_name", help="Bot name (e.g., Bot_BT, Bot_NN, Bot_Primitive)")
    bot_parser.add_argument("base_dir", nargs='?', default=default_base_dir,
                           help=f"Base simulation directory (default: {default_base_dir})")
    bot_parser.add_argument("-o", "--output", help="Output path for the image")
    bot_parser.add_argument("-p", "--position", choices=["left", "right", "both"], default="both",
                           help="Analyze bot when on left side, right side, or both (default: both)")
    bot_parser.add_argument("-c", "--chunksize", type=int, default=50000,
                           help="Chunk size for reading CSV files (default: 50000)")
    bot_parser.add_argument("--max-configs", type=int,
                           help="Maximum number of config folders to process per matchup (for testing)")
    bot_parser.add_argument("--use-timer", action="store_true",
                           help="Group by Timer values from config instead of phases (early/mid/late)")

    # All bots mode
    all_parser = subparsers.add_parser("all", help="Run ALL analyses: heatmaps, position distributions, distance distributions")
    all_parser.add_argument("base_dir", nargs='?', default=default_base_dir,
                           help=f"Base simulation directory (default: {default_base_dir})")
    all_parser.add_argument("-o", "--output", default="arena_heatmaps",
                           help="Base output directory for all visualizations (default: arena_heatmaps)")
    all_parser.add_argument("-p", "--position", choices=["left", "right", "both"], default="both",
                           help="Analyze bot when on left side, right side, or both (default: both)")
    all_parser.add_argument("-c", "--chunksize", type=int, default=50000,
                           help="Chunk size for reading CSV files (default: 50000)")
    all_parser.add_argument("--max-configs", type=int,
                           help="Maximum number of config folders to process per matchup (for testing)")
    all_parser.add_argument("--use-timer", action="store_true",
                           help="Group by Timer values from config instead of phases (early/mid/late)")
    all_parser.add_argument("--use-time-windows", action="store_true",
                           help="Group by fixed time windows: [0-15s], [15-30s], [30-45s], [45-60s]")
    all_parser.add_argument("--skip-initial", type=float, default=0.0,
                           help="Skip initial N seconds of data to remove spawn point bias (default: 0.0)")
    all_parser.add_argument("--test", type=int, nargs='?', const=1, default=None,
                           help="Test mode: process only N configs per matchup for quick testing (default: 1 if flag is used)")
    all_parser.add_argument("--cache-dir",
                           help="Also write a small CSV of each chart's raw sample data under this directory. "
                                "Pass it to `render-cache` afterward to redraw charts (title/label/color tweaks) "
                                "without rescanning the simulation data.")
    all_parser.add_argument("--use-cache-if-exists", action="store_true",
                           help="With --cache-dir: skip the expensive reload for any chart whose cache CSV "
                                "already exists there, rendering straight from cache instead. Existence-only "
                                "check - does not validate the cache matches these params, so delete stale "
                                "cache_dir contents first if skip-initial/position/etc changed since it was written.")

    # Render charts from a previously written data cache (fast, no raw data access)
    render_cache_parser = subparsers.add_parser("render-cache",
        help="Redraw chart PNGs from a --cache-dir written by 'all', without touching raw simulation data")
    render_cache_parser.add_argument("cache_dir", help="Cache directory previously passed as --cache-dir to 'all'")
    render_cache_parser.add_argument("-o", "--output", default="arena_heatmaps_rendered",
                                    help="Output directory for regenerated PNGs (default: arena_heatmaps_rendered)")
    render_cache_parser.add_argument("-p", "--position", choices=["left", "right", "both"], default="both",
                                    help="Position filter text shown in chart titles (default: both)")

    # Distance distributions mode
    distance_parser = subparsers.add_parser("distance", help="Generate distance distribution plots per bot (averaged across matchups)")
    distance_parser.add_argument("base_dir", nargs='?', default=default_base_dir,
                                help=f"Base simulation directory (default: {default_base_dir})")
    distance_parser.add_argument("-o", "--output", default="distance_distributions",
                                help="Output directory for distance plots - creates bot subfolders (default: distance_distributions)")
    distance_parser.add_argument("-c", "--chunksize", type=int, default=50000,
                                help="Chunk size for reading CSV files (default: 50000)")
    distance_parser.add_argument("--max-configs", type=int,
                                help="Maximum number of config folders to process per matchup (for testing)")

    # Distance over time mode (grouped by Timer)
    distance_time_parser = subparsers.add_parser("distance-time", help="Generate distance over time line plots (grouped by Timer) for all matchups")
    distance_time_parser.add_argument("base_dir", nargs='?', default=default_base_dir,
                                     help=f"Base simulation directory (default: {default_base_dir})")
    distance_time_parser.add_argument("-o", "--output", default="distance_over_time",
                                     help="Output directory for distance over time plots (default: distance_over_time)")
    distance_time_parser.add_argument("-c", "--chunksize", type=int, default=50000,
                                     help="Chunk size for reading CSV files (default: 50000)")
    distance_time_parser.add_argument("--max-configs", type=int,
                                     help="Maximum number of config folders to process per matchup (for testing)")

    args = parser.parse_args()

    if args.command == "single":
        create_phased_heatmap(args.csv_path, args.output, args.chunksize)

    elif args.command == "bot":
        output = args.output or f"phased_heatmap_{args.bot_name}_{args.position}.png"
        create_phased_heatmap_for_bot(
            args.base_dir,
            args.bot_name,
            args.position,
            output,
            args.chunksize,
            args.max_configs,
            args.use_timer
        )

    elif args.command == "all":
        start = time.time()
        # Validate that only one grouping mode is selected
        if args.use_timer and args.use_time_windows:
            print("❌ Error: Cannot use both --use-timer and --use-time-windows at the same time")
            print("   Please choose only one grouping mode:")
            print("   - --use-timer: Group by Timer config values")
            print("   - --use-time-windows: Group by fixed time windows [0-15s, 15-30s, 30-45s, 45-60s]")
            print("   - (default): Group by phases (early/mid/late)")
            exit(1)

        # Handle test mode
        if args.test is not None:
            max_configs = args.test
            mode_text = f"🧪 TEST MODE ({args.test} config(s) per matchup)"
        else:
            max_configs = args.max_configs
            mode_text = "🚀 Running ALL Analyses"

        print("=" * 60)
        print(mode_text)
        print("=" * 60)

        base_output = args.output

        # Generate all visualizations (heatmaps, position distributions, distance distributions)
        print("\n" + "=" * 60)
        print("Generating all visualizations...")
        print("=" * 60)
        heatmap_dir = os.path.join(base_output)
        create_phased_heatmaps_all_bots(
            args.base_dir,
            heatmap_dir,
            args.position,
            args.chunksize,
            max_configs,  # Use test value if --test flag is set
            "all",  # Generate both heatmaps and position distributions
            args.use_timer,
            args.use_time_windows,
            include_distance_over_time=True,  # Generate distance plots (only with --use-timer)
            skip_initial=args.skip_initial,
            cache_dir=args.cache_dir,
            use_cache_if_exists=args.use_cache_if_exists
        )

        print("\n" + "=" * 60)
        print("ALL ANALYSES COMPLETED!")
        print("=" * 60)
        print(f"All outputs saved to: {base_output}")
        print("\nGenerated in each bot folder:")
        print(f"  - Arena heatmaps")
        print(f"  - Position distributions")
        print(f"  - Distance distributions (distance_distribution.png)")
        if args.use_timer:
            print(f"  - Distance over time plots")
        print("=" * 60)
        
        elapsed_seconds = time.time() - start
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        processing_time = f"{int(hours):02d}:{int(minutes):02d}:{seconds:.2f}"
        print(f"\nProcessing Time: {processing_time}")

    elif args.command == "distance":
        create_distance_distributions_all_matchups(
            args.base_dir,
            args.output,
            args.chunksize,
            args.max_configs
        )

    elif args.command == "distance-time":
        create_distance_over_time_all_bots(
            args.base_dir,
            args.output,
            args.chunksize,
            args.max_configs
        )

    elif args.command == "render-cache":
        render_charts_from_cache(
            args.cache_dir,
            args.output,
            args.position
        )

    else:
        parser.print_help()
