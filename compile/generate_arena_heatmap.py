import time
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import glob
from tqdm import tqdm

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
# Config
# =====================
arena_center = np.array([0.24, 1.97])
arena_radius = 4.73485

# Adjustable parameters
tile_size = 0.7   # Larger = bigger heatmap tiles (lower resolution)
# arrow_size = 50   # Larger = longer arrows

def load_data_chunked(csv_path, chunksize=50000, actor_filter=None):
    """
    Load CSV data using Polars with GPU acceleration and streaming

    Args:
        csv_path: Path to CSV file
        chunksize: Number of rows per chunk (ignored for Polars, kept for API compatibility)
        actor_filter: Filter for specific actor (0 for left, 1 for right, None for both)
    """
    # Scan CSV without schema enforcement - let Polars infer naturally
    # Use ignore_errors to handle inconsistent column types across files
    # rechunk=False reduces memory overhead by avoiding unnecessary rechunking
    lf = pl.scan_csv(csv_path, ignore_errors=True, rechunk=False)

    # Select ONLY required columns to drastically reduce memory usage
    # This is critical for 135GB files - we only load what we need
    lf = lf.select([
        "GameIndex",     # For grouping by game
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
    Split game data into phases based on UpdatedAt time PER GAME.
    Each game (GameIndex) is split into early/mid/late phases independently.

    Args:
        df: Polars DataFrame with game data (must have GameIndex and UpdatedAt columns)
        num_phases: Number of phases to split into (default: 3 for early/mid/late)

    Returns:
        List of Polars DataFrames, one per phase (aggregated across all games)
    """
    if df.is_empty():
        return [pl.DataFrame()] * num_phases

    # Initialize phase containers
    phases = [[] for _ in range(num_phases)]

    # Process each game independently
    for game_idx in df["GameIndex"].unique().to_list():
        game_df = df.filter(pl.col("GameIndex") == game_idx)

        if game_df.is_empty():
            continue

        # Calculate time boundaries for THIS game
        min_time = game_df["UpdatedAt"].min()
        max_time = game_df["UpdatedAt"].max()
        time_range = max_time - min_time

        # Avoid division by zero for games with no time range
        if time_range == 0:
            # Put all data in the first phase
            phases[0].append(game_df)
            continue

        phase_size = time_range / num_phases

        # Split this game into phases
        for i in range(num_phases):
            phase_start = min_time + (i * phase_size)
            phase_end = min_time + ((i + 1) * phase_size)

            if i == num_phases - 1:
                # Include the last timestamp in the final phase
                phase_df = game_df.filter(
                    (pl.col("UpdatedAt") >= phase_start) & (pl.col("UpdatedAt") <= phase_end)
                )
            else:
                phase_df = game_df.filter(
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
    ax.set_xlabel("BotPosX (shifted)")
    ax.set_ylabel("BotPosY (shifted)")
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(0 - arena_radius - 1, 0 + arena_radius + 1)
    ax.set_ylim(0 - arena_radius - 1, 0 + arena_radius + 1)

    # Add grid
    ax.grid(True, alpha=0.3, zorder=0)


def plot_position_distribution(df_combined, bot_name, actor_position="both"):
    """
    Plot X and Y position distributions in a single frame (overlaid histograms)
    Y values are shifted by -2 since the game starts at y=2

    Args:
        df_combined: Combined Polars DataFrame with bot position data
        bot_name: Name of the bot
        actor_position: Position filter text for title

    Returns:
        matplotlib figure
    """
    if df_combined.is_empty():
        return None

    x = df_combined["BotPosX"].to_numpy() - arena_center[0]
    y = df_combined["BotPosY"].to_numpy() - arena_center[1]  # Shift by arena center

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
    ax.set_title(f"Distribution of {bot_name} Positions (Overlayed){position_text}\n(n={len(df_combined):,} samples)",
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Position (shifted by arena center)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_joint_heatmap_with_distributions(phase_df, phase_name, bot_name="", actor_position="both"):
    """
    Create a joint plot with contour heatmap and marginal distributions (like seaborn jointplot)
    Y values are shifted by -2 since the game starts at y=2

    Args:
        phase_df: Polars DataFrame with position data for a specific phase
        phase_name: Name of the phase (e.g., "Early Game")
        bot_name: Name of the bot
        actor_position: Position filter text for title

    Returns:
        matplotlib figure
    """
    if phase_df.is_empty():
        return None

    x = phase_df["BotPosX"].to_numpy() - arena_center[0]
    y = phase_df["BotPosY"].to_numpy() - arena_center[1]  # Shift by arena center

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
    ax_main.set_xlabel("X Position (shifted)", fontsize=12)
    ax_main.set_ylabel("Y Position (shifted)", fontsize=12)
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
    title = f"Contour Heatmap with Marginal Distributions\n{bot_name}{position_text} - {phase_name}\n(n={len(phase_df):,} samples)"
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
    Load all CSV data for a specific bot from the simulation directory

    Args:
        base_dir: Base simulation directory
        bot_name: Name of the bot (e.g., "Bot_BT", "Bot_NN", "Bot_Primitive")
        actor_position: "left" (Actor 0) or "right" (Actor 1) or "both"
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of config folders to process (None for all)
        group_by_timer: If True, return dict of {timer_value: DataFrame}, else return combined DataFrame
        also_load_distance: If True, also return timer-grouped distance data

    Returns:
        Combined DataFrame with all bot data, or dict of DataFrames grouped by Timer
        If also_load_distance=True, returns tuple: (bot_data, distance_data)
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

            # Find CSV file in this config folder
            csv_files = glob.glob(os.path.join(config_path, "*.csv"))

            if csv_files:
                csv_path = csv_files[0]  # Should only be 1 CSV per config
                df = load_data_chunked(csv_path, chunksize, actor_filter=actor_filter)

                if not df.is_empty():
                    # Also load distance data if requested
                    if also_load_distance:
                        df_all_actors = load_data_chunked(csv_path, chunksize, actor_filter=None)
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
    Calculate distance between Bot 1 (Actor 0) and Bot 2 (Actor 1) for each game frame

    Args:
        df: Polars DataFrame with columns including Actor, BotPosX, BotPosY, GameIndex, UpdatedAt

    Returns:
        Polars DataFrame with distance between bots for each frame
    """
    # Split data by actor - cast Actor inline for filtering
    bot1_df = df.filter(pl.col("Actor").cast(pl.Int64) == 0).select([
        "GameIndex", "UpdatedAt", "BotPosX", "BotPosY"
    ]).rename({"BotPosX": "Bot1_X", "BotPosY": "Bot1_Y"})

    bot2_df = df.filter(pl.col("Actor").cast(pl.Int64) == 1).select([
        "GameIndex", "UpdatedAt", "BotPosX", "BotPosY"
    ]).rename({"BotPosX": "Bot2_X", "BotPosY": "Bot2_Y"})

    # Merge on GameIndex and UpdatedAt to align frames
    merged = bot1_df.join(bot2_df, on=["GameIndex", "UpdatedAt"], how="inner")

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

def plot_distance_histogram_from_data(distance_data, bot_name, output_path=None):
    """
    Plot histogram of distance between bot and all opponents

    Args:
        distance_data: Dict of {timer: [list of distance dataframes]}
        bot_name: Name of the bot to analyze
        output_path: Path to save the figure

    Returns:
        matplotlib figure
    """
    if not distance_data:
        print("No valid distance data found")
        return None

    # Combine all distance data across all timers and opponents
    all_distances = []
    for timer, dfs in distance_data.items():
        combined_df = pl.concat(dfs, how="vertical_relaxed")
        all_distances.append(combined_df["Distance"].to_numpy())

    # Concatenate all distances
    distances = np.concatenate(all_distances)

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


def plot_distance_from_center_histogram(bot_data, bot_name, output_path=None):
    """
    Plot histogram of distance from center for a specific bot

    Args:
        bot_data: DataFrame or dict of DataFrames with bot position data
        bot_name: Name of the bot to analyze
        output_path: Path to save the figure

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


def plot_distance_over_time_from_data(timer_data, bot_name, output_path=None):
    """
    Plot mean distance over time from pre-loaded timer-grouped data

    Args:
        timer_data: Dict of {timer: [list of distance dataframes]}
        bot_name: Name of the bot to analyze
        output_path: Path to save the figure

    Returns:
        matplotlib figure
    """
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

def load_all_game_data(base_dir, bot1_name=None, bot2_name=None, chunksize=50000, max_configs=None):
    """
    Load all game data from simulation directory, optionally filtered by bot matchup

    Args:
        base_dir: Base simulation directory
        bot1_name: Name of bot 1 (optional filter)
        bot2_name: Name of bot 2 (optional filter)
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of configs to process (None for all)

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

            # Find CSV file
            csv_files = glob.glob(os.path.join(config_path, "*.csv"))

            if csv_files:
                csv_path = csv_files[0]
                # Load WITHOUT actor filter (we need both bots)
                df = load_data_chunked(csv_path, chunksize, actor_filter=None)

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

def create_distance_over_time_all_bots(base_dir, output_dir="arena_heatmaps", chunksize=50000, max_configs=None):
    """
    Create distance over time line plots for all bots (vs all opponents, grouped by Timer)
    Saves plots in each bot's directory within the output_dir

    Args:
        base_dir: Base simulation directory
        output_dir: Base output directory (plots will be saved in bot subdirectories)
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of configs to process
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


def create_distance_distributions_all_matchups(base_dir, output_dir="arena_heatmaps", chunksize=50000, max_configs=None, skip_initial=0.0):
    """
    Create distance distribution plots per bot (averaged across all matchups).
    Saves to {output_dir}/{bot_name}/distance_distribution.png

    Args:
        base_dir: Base simulation directory
        output_dir: Output directory (should be arena_heatmaps folder)
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of configs to process per matchup
        skip_initial: Skip initial N seconds of data to remove spawn point bias (default: 0.0)
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
        df = load_all_game_data(base_dir, bot1_name, bot2_name, chunksize, max_configs)

        if df.is_empty():
            print(f"  No data found for {matchup_folder}, skipping...")
            continue

        # Apply skip_initial filter if specified (per game)
        if skip_initial > 0:
            print(f"  ⏩ Skipping initial {skip_initial}s of data per game to remove spawn bias...")
            df = df.filter(
                pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over("GameIndex") + skip_initial
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

def create_phased_heatmaps_all_bots(base_dir, output_dir="arena_heatmap", actor_position="both", chunksize=50000, max_configs=None, mode="all", use_timer=False, use_time_windows=False, include_distance_over_time=True, skip_initial=0.0):
    """
    Create heatmaps and position distribution plots for all bots in the simulation directory
    Saves individual phase/timer images for each bot

    Args:
        base_dir: Base simulation directory
        output_dir: Output directory for heatmaps (default: "arena_heatmap")
        actor_position: "left", "right", or "both"
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of configs to process per matchup
        mode: What to generate - "heatmap", "position", or "all" (default: "all")
        use_timer: If True, group by Timer values instead of phases
        use_time_windows: If True, group by fixed time windows [0-15s, 15-30s, 30-45s, 45-60s]
        include_distance_over_time: If True, also generate distance over time plot (default: True)
        skip_initial: Skip initial N seconds of data to remove spawn point bias (default: 0.0)
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

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each bot
    for bot_name in bot_names:
        print("\n" + "=" * 60)
        print(f"Processing {bot_name}")
        print("=" * 60)

        # Create bot-specific directory
        bot_dir = os.path.join(output_dir, bot_name)
        os.makedirs(bot_dir, exist_ok=True)

        # Generate heatmaps if requested
        if mode in ["heatmap", "all"]:
            if use_timer:
                # Timer-based mode - load data with distance if needed
                print("\nLoading data grouped by Timer...")
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
                            pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over("GameIndex") + skip_initial
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
                    fig = plot_joint_heatmap_with_distributions(df, label, bot_name, actor_position)

                    if fig is not None:
                        # Save with timer in filename
                        timer_str = f"{int(timer)}" if timer == int(timer) else f"{timer}"
                        output_path = os.path.join(bot_dir, f"timer_{timer_str}.png")
                        plt.savefig(output_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved to {output_path}")
                        plt.close(fig)

                # Generate distance plots if requested and data is available
                if include_distance_over_time and distance_data:
                    print(f"\nGenerating distance over time plot...")
                    output_path = os.path.join(bot_dir, "distance_over_time.png")
                    fig = plot_distance_over_time_from_data(distance_data, bot_name, output_path)
                    if fig is not None:
                        plt.close(fig)

                    print(f"Generating distance histogram...")
                    output_path = os.path.join(bot_dir, "distance_histogram.png")
                    fig = plot_distance_histogram_from_data(distance_data, bot_name, output_path)
                    if fig is not None:
                        plt.close(fig)

                    print(f"Generating distance from center histogram...")
                    output_path = os.path.join(bot_dir, "distance_from_center_histogram.png")
                    fig = plot_distance_from_center_histogram(timer_data, bot_name, output_path)
                    if fig is not None:
                        plt.close(fig)

            elif use_time_windows:
                # Time window mode - fixed time windows [0-15s, 15-30s, 30-45s, 45-60s]
                print("\nLoading all data for time window grouping...")
                df_combined = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs, group_by_timer=False)

                if df_combined.is_empty():
                    print(f"No data found for {bot_name}, skipping...")
                    continue

                # Apply skip_initial filter if specified (per game)
                if skip_initial > 0:
                    print(f"\n⏩ Skipping initial {skip_initial}s of data per game to remove spawn bias...")
                    original_count = len(df_combined)
                    df_combined = df_combined.filter(
                        pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over("GameIndex") + skip_initial
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
                    fig = plot_joint_heatmap_with_distributions(window_df, window_name, bot_name, actor_position)

                    if fig is not None:
                        # Save with window name in filename
                        output_path = os.path.join(bot_dir, f"window_{start}-{end}s.png")
                        plt.savefig(output_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved to {output_path}")
                        plt.close(fig)

            else:
                # Phase-based mode (original)
                print("\nLoading all data...")
                df_combined = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs, group_by_timer=False)

                if df_combined.is_empty():
                    print(f"No data found for {bot_name}, skipping...")
                    continue

                # Apply skip_initial filter if specified (per game)
                if skip_initial > 0:
                    print(f"\n⏩ Skipping initial {skip_initial}s of data per game to remove spawn bias...")
                    original_count = len(df_combined)
                    df_combined = df_combined.filter(
                        pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over("GameIndex") + skip_initial
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
                    fig = plot_joint_heatmap_with_distributions(phase_df, phase_name, bot_name, actor_position)

                    if fig is not None:
                        # Save
                        output_path = os.path.join(bot_dir, f"{idx}.png")
                        plt.savefig(output_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved to {output_path}")
                        plt.close(fig)

        # Generate position distribution if requested
        if mode in ["position", "all"]:
            # Load combined data if not already loaded (needed for position distribution)
            if use_timer or use_time_windows:
                print("\nLoading combined data for position distribution...")
                df_combined = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs, group_by_timer=False)

                # Apply skip_initial filter if specified (per game)
                if skip_initial > 0 and not df_combined.is_empty():
                    print(f"\n⏩ Skipping initial {skip_initial}s of data per game to remove spawn bias...")
                    original_count = len(df_combined)
                    df_combined = df_combined.filter(
                        pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over("GameIndex") + skip_initial
                    )
                    print(f"  Filtered: {original_count:,} -> {len(df_combined):,} samples")

            # Check if we have data
            if 'df_combined' in locals() and not df_combined.is_empty():
                # Create position distribution plot
                print(f"Creating position distribution plot...")
                fig_dist = plot_position_distribution(df_combined, bot_name, actor_position)

                if fig_dist is not None:
                    dist_path = os.path.join(bot_dir, "position_distribution.png")
                    fig_dist.savefig(dist_path, dpi=150, bbox_inches='tight')
                    print(f"  Saved to {dist_path}")
                    plt.close(fig_dist)
            else:
                print(f"No data available for position distribution")

    # ========== Generate distance distributions per bot ==========
    print("\n" + "=" * 60)
    print("Generating distance distributions for each bot (across all matchups)...")
    print("=" * 60)

    # Collect data per bot (across all matchups)
    bot_distance_data = {}  # {bot_name: [distance_between_series, distance_from_center_series]}

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
        df = load_all_game_data(base_dir, bot1_name, bot2_name, chunksize, max_configs)

        if df.is_empty():
            print(f"  No data found for {matchup_folder}, skipping...")
            continue

        # Apply skip_initial filter if specified (per game)
        if skip_initial > 0:
            print(f"  ⏩ Skipping initial {skip_initial}s of data per game to remove spawn bias...")
            df = df.filter(
                pl.col("UpdatedAt") >= pl.col("UpdatedAt").min().over("GameIndex") + skip_initial
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
    print(f"✅ Completed! All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    default_base_dir = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"
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
            skip_initial=args.skip_initial
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

    else:
        parser.print_help()
