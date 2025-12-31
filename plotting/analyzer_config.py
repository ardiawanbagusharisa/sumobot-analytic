"""
Configuration and constants for sumobot analyzer
"""
import numpy as np

# =====================
# Arena Configuration
# =====================
arena_center = np.array([0.24, 1.97])
arena_radius = 4.73485

# =====================
# Visualization Parameters
# =====================
tile_size = 0.7   # Larger = bigger heatmap tiles (lower resolution)

# =====================
# Bot Marker Configuration
# =====================
# Map bot names to matplotlib marker shapes for easy visual differentiation
BOT_MARKER_MAP = {
    "Bot_NN": "o",           # #1: NN - Circle
    "Bot_ML_Classification": "s",          # #2: MLP - Square
    "Bot_MCTS": "8",         # #3: MCTS - Octagon
    "Bot_FuzzyLogic": "^",        # #4: Fuzzy - Triangle up
    "Bot_Primitive": "p",    # #5: Primitive - Pentagon
    "Bot_GA": "h",           # #6: GA - Hexagon
    "Bot_SLM_ActionGPT": "*",          # #7: SLM - Star
    "Bot_PPO": "8",          # #8: PPO - Octagon
    "Bot_BT": "X",           # #9: BT - X filled
    "Bot_UtilityAI": "P",      # #10: Utility - Plus
    "Bot_LLM_ActionGPT": "D",          # #11: LLM - Diamond
    "Bot_FSM": "v",          # #12: FSM - Triangle down
    "Bot_DQN": "d",          # #13: DQN - Thin diamond
}

# Default marker if bot not in map
DEFAULT_MARKER = "o"

# =====================
# Bot Linestyle Configuration (by Rank)
# =====================
# Map bot rank to matplotlib linestyle for performance visualization
# Top 5 performers get solid lines, others get different styles
BOT_LINESTYLE_BY_RANK = {
    1: "-",      # Solid - Best performer
    2: "-",      # Solid - 2nd best
    3: "-",      # Solid - 3rd best
    4: "-",      # Solid - 4th best
    5: "-",      # Solid - 5th best
    6: "--",     # Dashed
    7: "--",     # Dashed
    8: "-.",     # Dash-dot
    9: "-.",     # Dash-dot
    10: ":",     # Dotted
    11: ":",     # Dotted
    12: ":",     # Dotted
    13: ":",     # Dotted
}

# Default linestyle if rank not in map
DEFAULT_LINESTYLE = "-"

# =====================
# Bot Color Configuration
# =====================
# Map bot names to consistent colors for visualization
# Using distinct colors for easy differentiation
BOT_COLOR_MAP = {
    "Bot_NN": "#1f77b4",                    # Blue - #1
    "Bot_ML_Classification": "#ff7f0e",     # Orange - #2
    "Bot_MCTS": "#2ca02c",                  # Green - #3
    "Bot_FuzzyLogic": "#d62728",            # Red - #4
    "Bot_Primitive": "#9467bd",             # Purple - #5
    "Bot_GA": "#8c564b",                    # Brown - #6
    "Bot_SLM_ActionGPT": "#e377c2",         # Pink - #7
    "Bot_PPO": "#7f7f7f",                   # Gray - #8
    "Bot_BT": "#bcbd22",                    # Olive - #9
    "Bot_UtilityAI": "#17becf",             # Cyan - #10
    "Bot_LLM_ActionGPT": "#aec7e8",         # Light Blue - #11
    "Bot_FSM": "#ffbb78",                   # Light Orange - #12
    "Bot_DQN": "#98df8a",                   # Light Green - #13
}

# Default color if bot not in map
DEFAULT_COLOR = "#333333"

# =====================
# Theme Configuration (for non-bot visualizations)
# =====================
# Color palette for general charts (bars, matrices, etc.)
# Use these when bots are NOT the primary subject
THEME_COLORS = {
    # Primary theme colors
    'primary': '#2ca02c',      # Green
    'secondary': '#1f77b4',    # Blue
    'accent': '#ff7f0e',       # Orange
    'danger': '#d62728',       # Red
    'info': '#17becf',         # Cyan
    'warning': '#bcbd22',      # Olive

    # Chart-specific defaults
    'bar_default': '#d62728',           # Green for bar charts
    'heatmap_cmap': 'Blues',            # Colormap for heatmaps/matrices
    'regression_line': '#d62728',       # Red for regression lines
    'grid': '#cccccc',                  # Light gray for grids

    # Multi-category palettes (when you need multiple colors for non-bot categories)
    'categorical': ['#d62728', '#ff7f0e', '#2ca02c', '#17becf', '#9467bd', '#8c564b'],
}

# Default theme color
DEFAULT_THEME_COLOR = THEME_COLORS['primary']

# =====================
# Metric Name Mapping
# =====================
# Map metric/key names to proper display names
METRIC_NAME_MAP = {
    # Time-related metrics
    "MatchDur": "Match Duration",
    "ActInterval": "Action Interval",
    "Timer": "Timer",
    "Duration": "Duration",

    # Win/Performance metrics
    "WinRate": "Win Rate",
    "WinRate_L": "Win Rate (Left)",
    "WinRate_R": "Win Rate (Right)",
    "Rank": "Rank",

    # Action metrics
    "ActionCounts": "Action Counts",
    "ActionCounts_L": "Action Counts (Left)",
    "ActionCounts_R": "Action Counts (Right)",
    "Actions": "Actions",
    "AvgActions_L": "Avg Actions (Left)",
    "AvgActions_R": "Avg Actions (Right)",

    # Collision metrics
    "Collisions": "Collisions",
    "Collisions_L": "Collisions (Left)",
    "Collisions_R": "Collisions (Right)",
    "TotalCollisions": "Total Collisions",
    "Actor_L": "Actor (Left)",
    "Actor_R": "Actor (Right)",
    "Tie": "Tie",

    # Specific action types
    "Accelerate_Act": "Accelerate",
    "Accelerate_Dur": "Accelerate",
    "Accelerate_Act_L": "Accelerate (Left)",
    "Accelerate_Act_R": "Accelerate (Right)",
    "TurnLeft_Act": "Turn Left",
    "TurnLeft_Dur": "Turn Left",
    "TurnLeft_Act_L": "Turn Left (Left)",
    "TurnLeft_Act_R": "Turn Left (Right)",
    "TurnRight_Act": "Turn Right",
    "TurnRight_Dur": "Turn Right",
    "TurnRight_Act_L": "Turn Right (Left)",
    "TurnRight_Act_R": "Turn Right (Right)",
    "Dash_Act": "Dash",
    "Dash_Dur": "Dash",

    # Skill actions
    "SkillBoost_Act": "Skill Boost",
    "SkillBoost_Dur": "Skill Boost",
    "SkillBoost_Act_L": "Skill Boost (Left)",
    "SkillBoost_Act_R": "Skill Boost (Right)",
    "SkillStone_Act": "Skill Stone",
    "SkillStone_Dur": "Skill Stone",
    "SkillStone_Act_L": "Skill Stone (Left)",
    "SkillStone_Act_R": "Skill Stone (Right)",
    "TotalSkillAct": "Total Skill Actions",

    # Round/Game metrics
    "Round": "Round",
    "RoundNumeric": "Round",
    "SkillTypeNumeric": "Skill Type",
    "Games": "Games",

    # Bot identifiers
    "Bot": "Bot",
    "Bot_L": "Bot (Left)",
    "Bot_R": "Bot (Right)",
    "Enemy": "Enemy",
    "Left_Side": "Left Side",
    "Right_Side": "Right Side",

    # Skill types
    "Skill": "Skill",
    "SkillType": "Skill Type",
    "SkillLeft": "Skill (Left)",
    "SkillRight": "Skill (Right)",
    "SkillNumeric": "Skill (Numeric)",

    # Time bins
    "TimeBin": "Time Bin",

    # Other metrics
    "AvgDuration": "Avg Duration",
    "MeanCount": "Mean Count",
    "Count": "Count",
    "Action": "Action",
    "Side": "Side",
    "BotWithRank": "Bot (with Rank)",
    "BotWithRankLeft": "Bot (Left, with Rank)",
    "BotWithRankRight": "Bot (Right, with Rank)",
}


def get_metric_name(metric_key):
    """
    Get proper display name for a metric key.

    Args:
        metric_key: Raw metric/column name

    Returns:
        Proper display name if found in map, otherwise returns the raw metric key
    """
    return METRIC_NAME_MAP.get(metric_key, metric_key)


def get_bot_marker(bot_name):
    """
    Get marker shape for a given bot name.

    Args:
        bot_name: Name of the bot

    Returns:
        Matplotlib marker string
    """
    return BOT_MARKER_MAP.get(bot_name, DEFAULT_MARKER)


def get_bot_linestyle(rank):
    """
    Get linestyle for a given bot rank.
    Top 5 performers get solid lines, others get varied styles.

    Args:
        rank: Bot rank (1-13)

    Returns:
        Matplotlib linestyle string
    """
    return BOT_LINESTYLE_BY_RANK.get(rank, DEFAULT_LINESTYLE)


def get_bot_color(bot_name):
    """
    Get color for a given bot name.

    Args:
        bot_name: Name of the bot

    Returns:
        Hex color string
    """
    return BOT_COLOR_MAP.get(bot_name, DEFAULT_COLOR)


def get_theme_color(theme_key):
    """
    Get theme color for non-bot visualizations.

    Args:
        theme_key: Key from THEME_COLORS (e.g., 'primary', 'bar_default', 'categorical')

    Returns:
        Color string or list of colors for 'categorical'
    """
    return THEME_COLORS.get(theme_key, DEFAULT_THEME_COLOR)
