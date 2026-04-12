import os
import polars as pl


def count_games_in_folder(root_folder: str, target_games: int = None):
    """
    Count unique games in each parquet file within the folder structure.

    Expected structure:
    root_folder/
        Bot_A_vs_Bot_B/
            Timer_X__ActInterval_Y__Round_Z__SkillLeft_W__SkillRight_V/
                Timer_X__ActInterval_Y__Round_Z__SkillLeft_W__SkillRight_V.parquet

    Args:
        root_folder: Path to the root folder containing bot matchup folders
        target_games: Target number of games to reach. If provided, will show
                      green checkmark (✅) if reached, red cross (❌) if not.
    """
    # Find all bot vs bot folders
    bot_folders = sorted([
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d)) and "_vs_" in d
    ])

    for bot_folder in bot_folders:
        print(f"\n{bot_folder}:")
        bot_path = os.path.join(root_folder, bot_folder)

        # Find all Timer_* configuration folders
        config_folders = sorted([
            d for d in os.listdir(bot_path)
            if os.path.isdir(os.path.join(bot_path, d)) and d.startswith("Timer_")
        ])

        for config_folder in config_folders:
            config_path = os.path.join(bot_path, config_folder)
            parquet_file = os.path.join(config_path, f"{config_folder}.parquet")

            if os.path.isfile(parquet_file):
                # Read parquet and count unique GameIndex values
                df = pl.read_parquet(parquet_file)
                n_games = df.select("GameIndex").n_unique()

                # Add emoji indicator if target is provided
                if target_games is not None:
                    emoji = "✅" if n_games >= target_games else "❌"
                    print(f"  {emoji} {config_folder}: {n_games} games")
                else:
                    print(f"  - {config_folder}: {n_games} games")
            else:
                emoji = "❌" if target_games is not None else "-"
                print(f"  {emoji} {config_folder}: MISSING PARQUET FILE")


if __name__ == "__main__":
    # Default path - adjust as needed
    root_folder = "converted"
    target_games = 4  # Default target, adjust as needed

    # Check if the folder exists
    if not os.path.isdir(root_folder):
        print(f"Error: Folder not found: {root_folder}")
        print("Please provide the correct path to the data folder.")
    else:
        count_games_in_folder(root_folder, target_games=target_games)
