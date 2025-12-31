import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats
from .analyzer_config import (
    get_bot_marker,
    get_bot_color,
    get_bot_linestyle,
    get_theme_color,
    get_metric_name
)


def calculate_legend_padding(ax, x_labels=None, rotation=0, base_padding=-0.15):
    """
    Calculate dynamic padding for legend based on x-axis label length and rotation.

    Args:
        ax: Matplotlib axes object
        x_labels: List of x-axis labels (optional, will try to get from ax if not provided)
        rotation: Rotation angle of x-axis labels in degrees
        base_padding: Base padding value (default: -0.15)

    Returns:
        Adjusted padding value for bbox_to_anchor
    """
    if x_labels is None:
        # Try to get labels from the axis
        x_labels = [label.get_text() for label in ax.get_xticklabels()]

    if not x_labels or all(not label for label in x_labels):
        return base_padding

    # Calculate maximum label length
    max_label_len = max(len(str(label)) for label in x_labels)

    # Calculate padding based on rotation and length
    if rotation >= 30:
        # For rotated labels, length affects vertical space more
        # Longer labels need more space
        extra_padding = (max_label_len - 10) * 0.005  # Adjust factor as needed
        extra_padding = max(0, min(extra_padding, 0.15))  # Cap between 0 and 0.15
    else:
        # For horizontal labels, less impact
        extra_padding = (max_label_len - 15) * 0.003
        extra_padding = max(0, min(extra_padding, 0.1))

    return base_padding - extra_padding

def plot_grouped(summary, key="WinRate", group_by="ActInterval", width=10, height=7, chart_type="line", error_type="std"):
    """
    Plot average win rate per bot, grouped by a specific configuration variable.

    Parameters:
        group_by: one of ["ActInterval", "Timer", "Round", "SkillType"]
        chart_type: "line" for line chart with error bands, "bar" for bar chart
        error_type: "se" for standard error (recommended), "std" for standard deviation,
                    "ci" for 95% confidence interval
    """

    # --- Handle SkillType special case ---
    # SkillType combines SkillLeft and SkillRight into a unified grouping
    if group_by == "SkillType":
        left_group_col = "SkillLeft"
        right_group_col = "SkillRight"
    else:
        left_group_col = group_by
        right_group_col = group_by

    # --- Merge both sides ---
    left_cols = ["Bot_L", f"{key}_L", left_group_col]
    right_cols = ["Bot_R", f"{key}_R", right_group_col]

    if "Rank_L" in summary.columns:
        left_cols.append("Rank_L")
    if "Rank_R" in summary.columns:
        right_cols.append("Rank_R")

    left = summary[left_cols].rename(
        columns={"Bot_L": "Bot", f"{key}_L": key, "Rank_L": "Rank", left_group_col: group_by}
    )
    right = summary[right_cols].rename(
        columns={"Bot_R": "Bot", f"{key}_R": key, "Rank_R": "Rank", right_group_col: group_by}
    )

    combined = pd.concat([left, right], ignore_index=True)

    # Fill missing Rank with large number so unranked bots go last
    if "Rank" not in combined.columns:
        combined["Rank"] = np.nan
    combined["Rank"] = combined["Rank"].fillna(9999)

    # --- Aggregate (with std and count) ---
    grouped = (
        combined.groupby(["Bot", group_by], dropna=False)
        .agg({key: ["mean", "std", "count"], "Rank": "first"})
        .reset_index()
    )

    # Flatten column names
    grouped.columns = ["Bot", group_by, f"{key}_mean", f"{key}_std", f"{key}_count", "Rank"]
    grouped[f"{key}_std"] = grouped[f"{key}_std"].fillna(0)  # Handle cases with no std
    grouped[f"{key}_count"] = grouped[f"{key}_count"].fillna(1)  # Avoid division by zero

    # --- Sort bots by Rank ---
    bot_order = grouped.groupby("Bot")["Rank"].first().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(width, height))

    if chart_type == "line":
        # --- Line chart with error bands ---
        x_values = sorted(grouped[group_by].unique())

        for bot in bot_order:
            bot_data = grouped[grouped["Bot"] == bot].sort_values(group_by)
            rank = int(bot_data["Rank"].iloc[0])

            means = []
            errors = []
            for x_val in x_values:
                row = bot_data[bot_data[group_by] == x_val]
                if not row.empty:
                    mean_val = row[f"{key}_mean"].values[0]
                    std_val = row[f"{key}_std"].values[0]
                    count_val = row[f"{key}_count"].values[0]

                    means.append(mean_val)

                    # Calculate error based on error_type
                    if error_type == "se":
                        # Standard Error
                        error = std_val / np.sqrt(count_val) if count_val > 0 else 0
                    elif error_type == "ci":
                        # 95% Confidence Interval (approximation using 1.96 * SE)
                        error = 1.96 * (std_val / np.sqrt(count_val)) if count_val > 0 else 0
                    else:  # "std"
                        error = std_val

                    errors.append(error)
                else:
                    means.append(np.nan)
                    errors.append(0)

            means = np.array(means)
            errors = np.array(errors)

            # Plot line with bot-specific marker, color, and rank-based linestyle
            marker = get_bot_marker(bot)
            color = get_bot_color(bot)
            linestyle = get_bot_linestyle(rank)
            ax.plot(x_values, means, marker=marker, linestyle=linestyle, linewidth=2.5, markersize=7,
                   label=f"{bot} (#{rank})", color=color)

            # Plot error band with lighter transparency
            # ax.fill_between(x_values, means - errors, means + errors,
            #               alpha=0.15, color=colors[i])

        ax.set_xlabel(get_metric_name(group_by), fontsize=12, fontweight='bold')
        ax.set_ylabel(get_metric_name(key), fontsize=12, fontweight='bold')
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(x) for x in x_values])

        # Set Y-axis limits for WinRate
        if key == "WinRate":
            ax.set_ylim(-0.05, 1.05)

    else:
        # --- Bar chart (original) ---
        grouped_bar = grouped.rename(columns={f"{key}_mean": key})
        grouped_bar["Bot"] = pd.Categorical(grouped_bar["Bot"], categories=bot_order, ordered=True)
        grouped_bar = grouped_bar.sort_values(["Bot", group_by])

        labels = [
            f"{b} (#{int(grouped_bar[grouped_bar['Bot'] == b]['Rank'].iloc[0])})"
            for b in bot_order
        ]

        groups = sorted(grouped_bar[group_by].unique())
        x = np.arange(len(bot_order))
        bar_width = 0.8 / len(groups)

        for i, g in enumerate(groups):
            subset = grouped_bar[grouped_bar[group_by] == g]
            avg_by_bot = subset.set_index("Bot").reindex(bot_order)[key].fillna(0)
            ax.bar(x + i * bar_width, avg_by_bot, width=bar_width, label=str(g))

        ax.set_xticks(x + bar_width * (len(groups) - 1) / 2)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("Bots")

    # --- Common styling ---
    ax.set_title(f"{get_metric_name(key)} grouped by {get_metric_name(group_by)}", fontsize=14, fontweight='bold', pad=15)

    # Calculate dynamic padding for legend
    legend_padding = calculate_legend_padding(ax, rotation=0)
    ax.legend(title="Bot (Rank)" if chart_type == "line" else group_by,
             loc='upper center', bbox_to_anchor=(0.5, legend_padding), fontsize=10, framealpha=0.9, ncol=3, markerscale=1.2)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.8)
    fig.tight_layout()

    return fig

def prepare_individual_bot_data(df, bot_name):
    """
    Prepare data for a specific bot combining left and right perspectives.

    Args:
        df: Summary matchup dataframe
        bot_name: Name of the bot to analyze

    Returns:
        DataFrame with bot's data from all configurations
    """
    # Bot_L perspective
    df_left = df[df['Bot_L'] == bot_name].copy()
    df_left['WinRate'] = df_left['WinRate_L']
    df_left['Actions'] = df_left['ActionCounts_L']
    df_left['Collisions'] = df_left['Collisions_L']
    df_left['Collisions_Hit'] = df_left['Collisions_L']  # Hit (Actor_L)
    df_left['Collisions_Struck'] = df_left['Collisions_R']  # Struck (Actor_R)
    df_left['Collisions_Tie'] = df_left['Collisions_Tie']  # Tie
    df_left['Duration'] = df_left['Duration_L']
    df_left['Accelerate_Act'] = df_left['Accelerate_Act_L']
    df_left['TurnLeft_Act'] = df_left['TurnLeft_Act_L']
    df_left['TurnRight_Act'] = df_left['TurnRight_Act_L']
    df_left['Dash_Act'] = df_left['Dash_Act_L']
    df_left['SkillBoost_Act'] = df_left['SkillBoost_Act_L']
    df_left['SkillStone_Act'] = df_left['SkillStone_Act_L']

    df_left['Accelerate_Dur'] = df_left['Accelerate_Dur_L']
    df_left['TurnLeft_Dur'] = df_left['TurnLeft_Dur_L']
    df_left['TurnRight_Dur'] = df_left['TurnRight_Dur_L']
    df_left['Dash_Dur'] = df_left['Dash_Dur_L']
    # df_left['SkillBoost_Dur'] = df_left['SkillBoost_Dur_L']
    # df_left['SkillStone_Dur'] = df_left['SkillStone_Dur_L']

    df_left['SkillType'] = df_left['SkillLeft']

    # Bot_R perspective
    df_right = df[df['Bot_R'] == bot_name].copy()
    df_right['WinRate'] = df_right['WinRate_R']
    df_right['Actions'] = df_right['ActionCounts_R']
    df_right['Collisions'] = df_right['Collisions_R']
    df_right['Collisions_Hit'] = df_right['Collisions_R']  # Hit (Actor_R when on right)
    df_right['Collisions_Struck'] = df_right['Collisions_L']  # Struck (Actor_L when on right)
    df_right['Collisions_Tie'] = df_right['Collisions_Tie']  # Tie
    df_right['Duration'] = df_right['Duration_R']
    df_right['Accelerate_Act'] = df_right['Accelerate_Act_R']
    df_right['TurnLeft_Act'] = df_right['TurnLeft_Act_R']
    df_right['TurnRight_Act'] = df_right['TurnRight_Act_R']
    df_right['Dash_Act'] = df_right['Dash_Act_R']
    df_right['SkillBoost_Act'] = df_right['SkillBoost_Act_R']
    df_right['SkillStone_Act'] = df_right['SkillStone_Act_R']

    df_right['Accelerate_Dur'] = df_right['Accelerate_Dur_R']
    df_right['TurnLeft_Dur'] = df_right['TurnLeft_Dur_R']
    df_right['TurnRight_Dur'] = df_right['TurnRight_Dur_R']
    df_right['Dash_Dur'] = df_right['Dash_Dur_R']
    # df_right['SkillBoost_Dur'] = df_right['SkillBoost_Dur_R']
    # df_right['SkillStone_Dur'] = df_right['SkillStone_Dur_R']

    df_right['SkillType'] = df_right['SkillRight']

    # Combine both perspectives
    bot_data = pd.concat([df_left, df_right], ignore_index=True)

    # Add derived columns
    bot_data['RoundNumeric'] = bot_data['Round'].str.extract(r'(\d+)$').astype(int)
    bot_data['TotalSkillAct'] = bot_data['SkillBoost_Act'] + bot_data['SkillStone_Act']

    # Encode SkillType as numeric for correlation
    skill_map = {'Stone': 1, 'Boost': 2}
    bot_data['SkillTypeNumeric'] = bot_data['SkillType'].map(skill_map)

    return bot_data


def plot_individual_correlation_scatter(data, x_col, y_col, title, bot_name,
                                       alpha=0.95, figsize=(10, 8)):
    """
    Create scatter plot with regression line and Pearson correlation for individual bot.

    Args:
        data: DataFrame with bot's data
        x_col: Column name for x-axis
        y_col: Column name for y-axis (should be WinRate)
        title: Plot title
        bot_name: Name of the bot
        alpha: Transparency of scatter points
        figsize: Figure size tuple

    Returns:
        matplotlib figure
    """
    # Remove NaN values
    plot_data = data[[x_col, y_col]].dropna().copy()

    if len(plot_data) < 2:
        return None

    # Calculate Pearson correlation (on original data)
    pearson_r, pearson_p = stats.pearsonr(plot_data[x_col], plot_data[y_col])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Use original x values
    x_values = plot_data[x_col].values.copy()

    # Scatter plot - use bot color
    bot_color = get_bot_color(bot_name)
    ax.scatter(x_values, plot_data[y_col],
              alpha=alpha, s=60, color=bot_color, edgecolors='black', linewidth=0.5)

    # Add regression line (using original non-jittered data) - use theme color
    slope, intercept = np.polyfit(x_values, plot_data[y_col], 1)
    x_line = np.linspace(x_values.min(), x_values.max(), 100)
    y_line = slope * x_line + intercept
    # Clip to valid WinRate range [0, 1]
    if y_col == 'WinRate' or 'WinRate' in y_col:
        y_line = np.clip(y_line, 0, 1)
    ax.plot(x_line, y_line, '-', color=get_theme_color('regression_line'), linewidth=2.5, label=f'Regression Line')

    # Add correlation info to plot
    corr_text = f'Pearson r = {pearson_r:.3f}\np-value = {pearson_p:.3e}\nn = {len(plot_data)}'
    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.8), fontsize=11, family='monospace')

    # Labels and title
    ax.set_xlabel(get_metric_name(x_col), fontsize=12, fontweight='bold')
    ax.set_ylabel(get_metric_name(y_col), fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\n{bot_name}', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9, markerscale=1.2)

    plt.tight_layout()
    return fig


def plot_individual_bot_correlations(df, bot_name, width=10, height=8,alpha=0.2):
    """
    Create all correlation plots for a specific bot.
    For config variables, plots win rate directly against the config values.
    For actions/collisions, creates separate plots for each config value.

    Args:
        df: Summary matchup dataframe
        bot_name: Name of the bot to analyze
        width: Figure width
        height: Figure height

    Returns:
        Dictionary of figures with nested structure for config-separated plots
    """
    # Prepare data for this bot
    data = prepare_individual_bot_data(df, bot_name)

    if data.empty:
        return {}

    figs = {}

    # a. Winrate vs ActInterval (direct correlation)
    fig = plot_individual_correlation_scatter(
        data,
        x_col='ActInterval',
        y_col='WinRate',
        title='Win Rate vs Action Interval',
        bot_name=bot_name,
        figsize=(width, height),
    )
    if fig:
        figs['actinterval'] = fig

    # b. Winrate vs Round type (direct correlation)
    # Build dynamic round type mapping for title
    round_mapping = data[['Round', 'RoundNumeric']].drop_duplicates().dropna()
    round_labels = ', '.join([f"{int(row['RoundNumeric'])}={row['Round']}"
                              for _, row in round_mapping.sort_values('RoundNumeric').iterrows()])
    round_title = f'Win Rate vs Round Type ({round_labels})' if round_labels else 'Win Rate vs Round Type'

    fig = plot_individual_correlation_scatter(
        data,
        x_col='RoundNumeric',
        y_col='WinRate',
        title=round_title,
        bot_name=bot_name,
        figsize=(width, height),
    )
    if fig:
        figs['roundtype'] = fig

    # c. Winrate vs Timer (direct correlation)
    fig = plot_individual_correlation_scatter(
        data,
        x_col='Timer',
        y_col='WinRate',
        title='Win Rate vs Timer Duration',
        bot_name=bot_name,
        figsize=(width, height),
    )
    if fig:
        figs['timer'] = fig

    # d. Winrate vs Skill Type (direct correlation)
    fig = plot_individual_correlation_scatter(
        data,
        x_col='SkillTypeNumeric',
        y_col='WinRate',
        title='Win Rate vs Skill Type (1=Stone, 2=Boost)',
        bot_name=bot_name,
        figsize=(width, height),
    )
    if fig:
        figs['skilltype'] = fig

    # e. Winrate vs Individual Actions (combined across all configs)
    action_types = ['Accelerate_Act', 'TurnLeft_Act', 'TurnRight_Act',
                   'Dash_Act', 'SkillBoost_Act', 'SkillStone_Act']

    fig, axes = plt.subplots(2, 3, figsize=(width*1.8, height*1.2))
    axes = axes.flatten()

    for idx, action in enumerate(action_types):
        if action not in data.columns:
            continue

        plot_data = data[[action, 'WinRate']].dropna()

        if len(plot_data) < 2:
            axes[idx].text(0.5, 0.5, f'Insufficient data',
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue

        # Calculate Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(plot_data[action], plot_data['WinRate'])

        # Scatter plot - use bot color
        bot_color = get_bot_color(bot_name)
        axes[idx].scatter(plot_data[action], plot_data['WinRate'],
                        alpha=alpha, s=50, color=bot_color, edgecolors='black', linewidth=0.5)

        # Regression line - use theme color
        if len(plot_data) >= 2 and plot_data[action].std() > 0:
            slope, intercept = np.polyfit(plot_data[action], plot_data['WinRate'], 1)
            x_line = np.linspace(plot_data[action].min(), plot_data[action].max(), 100)
            y_line = slope * x_line + intercept
            # Clip to valid WinRate range [0, 1]
            y_line = np.clip(y_line, 0, 1)
            axes[idx].plot(x_line, y_line, '-', color=get_theme_color('regression_line'), linewidth=2)

        # Correlation info
        corr_text = f'r={pearson_r:.3f}\np={pearson_p:.2e}'
        axes[idx].text(0.05, 0.95, corr_text, transform=axes[idx].transAxes,
                      verticalalignment='top', bbox=dict(boxstyle='round',
                      facecolor='wheat', alpha=0.8), fontsize=9, family='monospace')

        axes[idx].set_xlabel(get_metric_name(action), fontsize=10, fontweight='bold')
        axes[idx].set_ylabel(get_metric_name('WinRate'), fontsize=10, fontweight='bold')
        axes[idx].set_title(f'{get_metric_name("WinRate")} vs {get_metric_name(action)}', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'Win Rate vs Individual Action Types\n{bot_name}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    figs['actions'] = fig

    # f. Winrate vs Individual Actions Duration (combined across all configs)
    action_dur_types = ['Accelerate_Dur', 'TurnLeft_Dur', 'TurnRight_Dur', 'Dash_Dur']

    fig, axes = plt.subplots(2, 2, figsize=(width*1.2, height*1.2))
    axes = axes.flatten()

    for idx, action in enumerate(action_dur_types):
        if action not in data.columns:
            continue

        plot_data = data[[action, 'WinRate']].dropna()

        if len(plot_data) < 2:
            axes[idx].text(0.5, 0.5, f'Insufficient data',
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue

        # Calculate Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(plot_data[action], plot_data['WinRate'])

        # Scatter plot - use bot color
        bot_color = get_bot_color(bot_name)
        axes[idx].scatter(plot_data[action], plot_data['WinRate'],
                        alpha=alpha, s=50, color=bot_color, edgecolors='black', linewidth=0.5)

        # Regression line - use theme color
        if len(plot_data) >= 2 and plot_data[action].std() > 0:
            slope, intercept = np.polyfit(plot_data[action], plot_data['WinRate'], 1)
            x_line = np.linspace(plot_data[action].min(), plot_data[action].max(), 100)
            y_line = slope * x_line + intercept
            # Clip to valid WinRate range [0, 1]
            y_line = np.clip(y_line, 0, 1)
            axes[idx].plot(x_line, y_line, '-', color=get_theme_color('regression_line'), linewidth=2)

        # Correlation info
        corr_text = f'r={pearson_r:.3f}\np={pearson_p:.2e}'
        axes[idx].text(0.05, 0.95, corr_text, transform=axes[idx].transAxes,
                      verticalalignment='top', bbox=dict(boxstyle='round',
                      facecolor='wheat', alpha=0.8), fontsize=9, family='monospace')

        axes[idx].set_xlabel(get_metric_name(action), fontsize=10, fontweight='bold')
        axes[idx].set_ylabel(get_metric_name('WinRate'), fontsize=10, fontweight='bold')
        axes[idx].set_title(f'{get_metric_name("WinRate")} vs {get_metric_name(action)}', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'Win Rate vs Individual Action Duration\n{bot_name}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    figs['actions_dur'] = fig

    # g. Winrate vs Collision Types (Hit, Struck, Tie) - Combined across all configs
    collision_types = ['Collisions_Hit', 'Collisions_Struck', 'Collisions_Tie']
    collision_labels = {'Collisions_Hit': 'Hit', 'Collisions_Struck': 'Struck', 'Collisions_Tie': 'Tie'}

    fig, axes = plt.subplots(1, 3, figsize=(width*1.8, height))

    for idx, col_type in enumerate(collision_types):
        if col_type not in data.columns:
            continue

        plot_data = data[[col_type, 'WinRate']].dropna()

        if len(plot_data) < 2:
            axes[idx].text(0.5, 0.5, f'Insufficient data',
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue

        # Calculate Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(plot_data[col_type], plot_data['WinRate'])

        # Scatter plot - use bot color
        bot_color = get_bot_color(bot_name)
        axes[idx].scatter(plot_data[col_type], plot_data['WinRate'],
                        alpha=alpha, s=60, color=bot_color, edgecolors='black', linewidth=0.5)

        # Regression line - use theme color
        if len(plot_data) >= 2 and plot_data[col_type].std() > 0:
            slope, intercept = np.polyfit(plot_data[col_type], plot_data['WinRate'], 1)
            x_line = np.linspace(plot_data[col_type].min(), plot_data[col_type].max(), 100)
            y_line = slope * x_line + intercept
            # Clip to valid WinRate range [0, 1]
            y_line = np.clip(y_line, 0, 1)
            axes[idx].plot(x_line, y_line, '-', color=get_theme_color('regression_line'), linewidth=2.5)

        # Correlation info
        corr_text = f'r={pearson_r:.3f}\np={pearson_p:.2e}'
        axes[idx].text(0.05, 0.95, corr_text, transform=axes[idx].transAxes,
                      verticalalignment='top', bbox=dict(boxstyle='round',
                      facecolor='wheat', alpha=0.8), fontsize=10, family='monospace')

        axes[idx].set_xlabel(collision_labels[col_type], fontsize=11, fontweight='bold')
        axes[idx].set_ylabel(get_metric_name('WinRate'), fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{get_metric_name("WinRate")} vs {collision_labels[col_type]}',
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'Win Rate vs Collision Types\n{bot_name}',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    figs['collisions'] = fig

    return figs


def show_individual_report(df, toc, width, height):
    toc.h2("Individual Reports")
    st.markdown("Analyze bot agent against its different configurations")
    st.markdown("Each of report: Win Rate; Collision; Action-Taken; Duration; is calculated with averaging data from matchup (left and right position)")
    
    # # Individual Win Rates
    # toc.h3("Win Rates")
    # st.markdown("Reports of win rates each of bot")
    
    # toc.h4("Win Rate by Timer")
    # st.pyplot(plot_grouped(df,group_by="Timer",width=width, height=height))

    # toc.h4("Win Rate by ActInterval")
    # st.pyplot(plot_grouped(df,group_by="ActInterval",width=width, height=height))

    # toc.h4("Win Rate by Round")
    # st.pyplot(plot_grouped(df,group_by="Round",width=width, height=height))

    # toc.h4("Win Rate by SkillType")
    # st.pyplot(plot_grouped(df,group_by="SkillType",width=width, height=height))


    # # Individual Action Taken
    # toc.h3("Action Taken")
    # st.markdown("Reports of action taken from each of bot")

    # toc.h4("Action Counts by Timer")
    # st.pyplot(plot_grouped(df,key="ActionCounts", group_by="Timer",width=width, height=height))

    # toc.h4("Action Counts by ActInterval")
    # st.pyplot(plot_grouped(df,key="ActionCounts", group_by="ActInterval",width=width, height=height))

    # toc.h4("Action Counts by Round")
    # st.pyplot(plot_grouped(df,key="ActionCounts", group_by="Round",width=width, height=height))

    # toc.h4("Action Counts by SkillType")
    # st.pyplot(plot_grouped(df,key="ActionCounts", group_by="SkillType",width=width, height=height))
    
    # # Individual Collision
    # toc.h3("Collisions")
    # st.markdown("Reports of collision made from each of bot")

    # toc.h4("Collisions by Timer")
    # st.pyplot(plot_grouped(df,key="Collisions", group_by="Timer",width=width, height=height))

    # toc.h4("Collisions by ActInterval")
    # st.pyplot(plot_grouped(df,key="Collisions", group_by="ActInterval",width=width, height=height))

    # toc.h4("Collisions by Round")
    # st.pyplot(plot_grouped(df,key="Collisions", group_by="Round",width=width, height=height))

    # toc.h4("Collisions by SkillType")
    # st.pyplot(plot_grouped(df,key="Collisions", group_by="SkillType",width=width, height=height))

    # # Individual Duration
    # toc.h3("Duration")
    # st.markdown("Reports of action-taken duration produced from each of bot")

    # toc.h4("Duration by Timer")
    # st.pyplot(plot_grouped(df,key="Duration", group_by="Timer",width=width, height=height))

    # toc.h4("Duration by ActInterval")
    # st.pyplot(plot_grouped(df,key="Duration", group_by="ActInterval",width=width, height=height))

    # toc.h4("Duration by Round")
    # st.pyplot(plot_grouped(df,key="Duration", group_by="Round",width=width, height=height))

    # toc.h4("Duration by SkillType")
    # st.pyplot(plot_grouped(df,key="Duration", group_by="SkillType",width=width, height=height))

    # Pearson Correlation Analysis for Individual Bots
    toc.h2("Pearson Correlation Analysis (Per Bot)")
    st.markdown("**Correlation analysis using Pearson coefficient with scatter plots and regression lines**")
    st.markdown("Detailed plots for individual bots, separated by configuration")

    # Get unique bots
    bots = sorted(df['Bot_L'].unique())

    for bot in bots:
        toc.h3(f"{bot} - Correlation Analysis")
        st.markdown(f"**Analyzing correlations for {bot}**")

        correlation_figs = plot_individual_bot_correlations(df, bot, width, height)

        if not correlation_figs:
            st.warning(f"No data available for {bot}")
            continue

        # Create tabs for different correlation plots
        tab_labels = []
        if 'actinterval' in correlation_figs:
            tab_labels.append("By ActInterval")
        if 'roundtype' in correlation_figs:
            tab_labels.append("By Round")
        if 'timer' in correlation_figs:
            tab_labels.append("By Timer")
        if 'skilltype' in correlation_figs:
            tab_labels.append("By Skill Type")
        if 'actions' in correlation_figs:
            tab_labels.append("Action Types")
        if 'actions_dur' in correlation_figs:
            tab_labels.append("Action Duration")
        if 'collisions' in correlation_figs:
            tab_labels.append("Collisions")

        if tab_labels:
            tabs = st.tabs(tab_labels)

            tab_idx = 0
            if 'actinterval' in correlation_figs:
                with tabs[tab_idx]:
                    st.markdown("**Win Rate vs Action Interval Configuration**")
                    st.pyplot(correlation_figs['actinterval'])
                tab_idx += 1

            if 'roundtype' in correlation_figs:
                with tabs[tab_idx]:
                    st.markdown("**Win Rate vs Round Type Configuration**")
                    st.pyplot(correlation_figs['roundtype'])
                tab_idx += 1

            if 'timer' in correlation_figs:
                with tabs[tab_idx]:
                    st.markdown("**Win Rate vs Timer Configuration**")
                    st.pyplot(correlation_figs['timer'])
                tab_idx += 1

            if 'skilltype' in correlation_figs:
                with tabs[tab_idx]:
                    st.markdown("**Win Rate vs Skill Type Configuration**")
                    st.pyplot(correlation_figs['skilltype'])
                tab_idx += 1

            if 'actions' in correlation_figs:
                with tabs[tab_idx]:
                    st.markdown("**Win Rate vs Individual Action Types**")
                    st.pyplot(correlation_figs['actions'])
                tab_idx += 1

            if 'actions_dur' in correlation_figs:
                with tabs[tab_idx]:
                    st.markdown("**Win Rate vs Individual Action Duration**")
                    st.pyplot(correlation_figs['actions_dur'])
                tab_idx += 1

            if 'collisions' in correlation_figs:
                with tabs[tab_idx]:
                    st.markdown("**Win Rate vs Collision Types (Hit, Struck, Tie)**")
                    st.pyplot(correlation_figs['collisions'])
                tab_idx += 1
