"""
Configuration and constants for sumobot analyzer
Loads configuration from config.json
"""
import json
import numpy as np
from pathlib import Path

# Load configuration from JSON file
_config_path = Path(__file__).parent.parent / "config.json"
with open(_config_path, 'r') as f:
    _config = json.load(f)

# =====================
# Arena Configuration
# =====================
arena_center = np.array(_config['arena']['center'])
arena_radius = _config['arena']['radius']

# =====================
# Visualization Parameters
# =====================
tile_size = _config['visualization']['tile_size']

# =====================
# Bot Marker Configuration
# =====================
BOT_MARKER_MAP = _config['bot_markers'].copy()
DEFAULT_MARKER = BOT_MARKER_MAP.pop('default')

# =====================
# Bot Linestyle Configuration (by Rank)
# =====================
# Convert string keys to integers for rank mapping
_linestyle_config = _config['bot_linestyles_by_rank']
BOT_LINESTYLE_BY_RANK = {int(k): v for k, v in _linestyle_config.items() if k != 'default'}
DEFAULT_LINESTYLE = _linestyle_config['default']

# =====================
# Bot Color Configuration
# =====================
BOT_COLOR_MAP = _config['bot_colors'].copy()
DEFAULT_COLOR = BOT_COLOR_MAP.pop('default')

# =====================
# Theme Configuration (for non-bot visualizations)
# =====================
THEME_COLORS = _config['theme_colors'].copy()
DEFAULT_THEME_COLOR = THEME_COLORS.pop('default')

# =====================
# Metric Name Mapping
# =====================
METRIC_NAME_MAP = _config['metric_names']


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
