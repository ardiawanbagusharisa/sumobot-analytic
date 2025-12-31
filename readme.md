# SumoBot Performance Analysis Dashboard

Live dashboard on https://sumo.orpheus.my.id

## Overview
This project provides a Streamlit-based dashboard for analyzing SumoBot gameplay performance across multiple configurations. It processes structured match data to generate visual reports on win rates, action frequencies, collision counts, and match duration by configuration variables.

## Key Features

### 1. Individual Bot Analysis
- Win rate by configuration (Timer, Action Interval, Round, Skill)
- Action counts per bot across configurations
- Collision frequency per bot
- Duration per bot by configuration

Each report aggregates data from both left and right bot positions to ensure fair comparison.

### 2. Overall Matchup Analysis
- Win rate matrix showing how each bot performs against others
- Time-related trends: match duration vs. timer and action interval
- Action vs. win correlation: examines whether aggressive behavior correlates with higher win rates
- Top actions per bot: identifies most frequently used actions

## Data Requirements
- Input files:
  - `summary_bot.csv`: contains bot-level summary data
  - `summary_matchup.csv`: contains matchup-level data with columns:
    - `Bot_L`, `Bot_R`: left and right bot names
    - `WinRate_L`, `WinRate_R`: win rate of each bot
    - `Timer`, `ActInterval`, `Round`, `SkillLeft`, `SkillRight`
    - `ActionCounts_L`, `ActionCounts_R`, `Duration_L`, `Duration_R`, `Games`

## Data Processing
- Data is aggregated by configuration (Timer, ActInterval, etc.) to compute average performance per bot.
- Win rates are averaged across all matchups for each configuration.
- Action counts are normalized by number of games per matchup.

## Visualization
- Bar plots and heatmaps for win rate and action distribution
- Line plots for win rate trends over action interval
- Correlation plots to assess relationship between action frequency and win rate

## Usage
1. Place the CSV files in the project directory.
2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
3. Explore the dashboard to view performance across configurations.