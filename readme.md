# SumoBot Performance Analysis

A comprehensive data analysis toolkit for analyzing SumoBot gameplay performance across multiple configurations. This project processes simulation logs, generates statistical summaries, and provides rich visualizations through Jupyter notebooks and an interactive Streamlit dashboard.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Analysis Workflow](#analysis-workflow)
- [Optional: Interactive Dashboard](#optional-interactive-dashboard)
- [Project Structure](#project-structure)
- [Data Format](#data-format)

## Overview

This toolkit analyzes SumoBot match data to provide insights into:
- **Win rates** by configuration (Timer, Action Interval, Round, Skill)
- **Action patterns** and frequencies across different bots
- **Collision dynamics** (Hit, Struck, Tie events)
- **Temporal analysis** showing action/collision intensity over time
- **Arena heatmaps** visualizing bot movement patterns
- **Correlation analysis** between actions, collisions, and win rates

## Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `plotly` - Interactive plots
- `seaborn` - Statistical visualizations
- `streamlit` - Web dashboard (optional)
- `streamlit_modal` - Modal dialogs for Streamlit (optional)

## Quick Start

1. **Configure paths** in `analysis.ipynb`:
   ```python
   simulation_root = "/path/to/Simulation"  # Unity simulation logs
   csv_folder = "/path/to/CSV_output"       # Converted CSV files
   ```

2. **Run the analysis notebook**:
   ```bash
   jupyter notebook analysis.ipynb
   ```

3. **Execute cells sequentially** to process data and generate visualizations

## Analysis Workflow

The main analysis is performed in **`analysis.ipynb`**, which is divided into the following sections:

### 1. Configuration

Set up directory paths for input simulation logs and output files:

```python
simulation_root = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"
csv_folder = "/Users/defdef/Documents/Simulation"
```

### 2. Data Compilation

#### Step 2.1: Convert Simulation Logs to CSV

Convert Unity simulation logs to structured CSV format:

```python
from compile.log_to_csv import convert_all_configs
convert_all_configs(simulation_root, csv_folder)
```

This processes all configuration folders and generates CSV files organized by matchup.

#### Step 2.2: Generate Summarization CSV

Process CSV files in batches to create summary statistics:

```python
from compile.generator_polars_gpu import batch_process_csvs, generate

# Process in batches
batch_process_csvs(
    csv_folder,
    batch_size=2,
    time_bin_size=5,
    checkpoint_dir="batched",
    compute_timebins=True
)

# Generate final summaries
generate("batched", "result")
generate_timebins_from_batches("batched", "result")
```

**Outputs:**
- `result/summary_bot.csv` - Per-bot aggregated statistics
- `result/summary_matchup.csv` - Per-matchup configuration data
- `result/summary_action_timebins.csv` - Action intensity over time
- `result/summary_collision_timebins.csv` - Collision events over time

#### Step 2.3: Generate Arena Heatmaps

Create spatial heatmaps showing bot movement patterns:

```python
from compile.generate_arena_heatmap import create_phased_heatmaps_all_bots

create_phased_heatmaps_all_bots(
    csv_folder,
    output_dir="result/arena_heatmaps",
    actor_position="both",
    use_time_windows=True,
    skip_initial=2.5
)
```

Generates heatmaps for different time windows: Early (2.5-15s), Mid (15-30s, 30-45s), Late (45-60s).

### 3. Plotting and Analysis

#### Load Data

```python
df_sum = pd.read_csv("result/summary_bot.csv")
df = pd.read_csv("result/summary_matchup.csv")
df_timebins = pd.read_csv("result/summary_action_timebins.csv")
df_collision_timebins = pd.read_csv("result/summary_collision_timebins.csv")
```

#### Overall Analysis

Analyzes performance across all bots and configurations:

- **Win Rate Matrix** - Head-to-head matchup results
- **Action Radar Charts** - Mean action counts per bot
- **Collision Radar Charts** - Hit/Struck/Tie distribution
- **Grouped Configurations** - Win rates by Timer, ActInterval, Round, Skill
- **Time-Related Trends** - Match duration vs configuration
- **Action Distribution** - Stacked bar charts of action types
- **Action/Collision Intensity Over Time** - Temporal patterns
- **Correlation Analysis** - Pearson correlations between metrics

Example:
```python
from plotting.overall_analyzer import plot_winrate_matrix

fig = plot_winrate_matrix(df, width=10, height=6)
plt.show()
```

#### Individual Bot Analysis

Detailed per-bot analysis with configuration breakdowns:

- **Correlation Analysis** - Win rate vs actions, collisions, duration
- **Configuration Impact** - How Timer/ActInterval/Round/Skill affect performance
- **Arena Heatmaps** - Movement patterns across game phases
- **Position & Distance Distributions** - Spatial behavior analysis

Example:
```python
from plotting.individual_analyzer import plot_individual_bot_correlations

correlation_figs = plot_individual_bot_correlations(df, "BotName", width=10, height=6)
```

### 4. Interpreting Results

Key insights to look for:

1. **Win Rate Patterns**: Which configurations favor which bots?
2. **Action Effectiveness**: Do more actions correlate with higher win rates?
3. **Temporal Dynamics**: How do bots behave in early vs late game?
4. **Spatial Patterns**: Do winning bots control specific arena zones?
5. **Collision Analysis**: Is aggressive collision behavior advantageous?

## Optional: Interactive Dashboard

For interactive exploration, run the Streamlit dashboard:

### Prerequisites

Ensure you've completed the **Data Compilation** steps in `analysis.ipynb` to generate:
- `result/summary_bot.csv`
- `result/summary_matchup.csv`
- `result/summary_action_timebins.csv`
- `result/summary_collision_timebins.csv`
- `result/arena_heatmaps/` (optional but recommended)

### Running the Dashboard

```bash
streamlit run streamlit_plot.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Dashboard Features

- **Interactive Controls**: Adjust plot width/height via sidebar sliders
- **Summary Tables**: View bot rankings and matchup statistics
- **Dynamic Filtering**: Explore data by configuration parameters
- **Modal Views**: Click to view complete matchup details
- **Responsive Visualizations**: All plots from the notebook in an interactive format

### Usage Tips

1. Use the sidebar to adjust visualization dimensions
2. Click "View complete matchup" to see full matchup data
3. Scroll through sections: Overall Analysis → Individual Bot Reports → Arena Heatmaps
4. Hover over plots for detailed tooltips
5. Export plots using the built-in Streamlit controls

## Project Structure

```
sumobot-analytic/
├── analysis.ipynb              # Main analysis notebook (START HERE)
├── streamlit_plot.py           # Interactive dashboard (optional)
├── requirements.txt            # Python dependencies
├── compile/                    # Data processing modules
│   ├── log_to_csv.py          # Convert simulation logs to CSV
│   ├── generator_polars_gpu.py # Generate summary statistics
│   └── generate_arena_heatmap.py # Create spatial heatmaps
├── plotting/                   # Visualization modules
│   ├── overall_analyzer.py    # Cross-bot analysis plots
│   ├── individual_analyzer.py # Per-bot analysis plots
│   └── stoc.py                # Streamlit table of contents
└── result/                     # Generated output (auto-created)
    ├── summary_bot.csv
    ├── summary_matchup.csv
    ├── summary_action_timebins.csv
    ├── summary_collision_timebins.csv
    └── arena_heatmaps/
```

## Data Format

### Input: Simulation Logs
Located in Unity's data directory:
```
Simulation/
└── BotA_vs_BotB/
    └── Timer60_ActInterval0.1_Round1_SkillNone_SkillNone/
        ├── game_0.log
        ├── game_1.log
        └── ...
```

### Output: Summary CSV

**summary_bot.csv** - Aggregated bot statistics
- Columns: `Bot`, `WinRate`, `ActionCounts`, `Collisions`, `Duration`, `Games`, `Rank`

**summary_matchup.csv** - Configuration-specific matchup data
- Columns: `Bot_L`, `Bot_R`, `Timer`, `ActInterval`, `Round`, `SkillLeft`, `SkillRight`, `WinRate_L`, `WinRate_R`, `ActionCounts_L`, `ActionCounts_R`, `Collisions_L`, `Collisions_R`, `Duration_L`, `Duration_R`, `MatchDur`, `Games`

**summary_action_timebins.csv** - Temporal action data
- Time-binned action counts (default: 5-second bins)

**summary_collision_timebins.csv** - Temporal collision data
- Time-binned collision events (Hit, Struck, Tie)

---
