import os
import re
from itertools import product


def count_games_in_folder(root_folder: str, target_configs: int = None, target_games: int = None):
    """
    Count unique games in each parquet file within the folder structure.

    Expected structure:
    root_folder/
        Bot_A_vs_Bot_B/
            Timer_X__ActInterval_Y__Round_Z__SkillLeft_W__SkillRight_V/
                Timer_X__ActInterval_Y__Round_Z__SkillLeft_W__SkillRight_V.parquet

    Args:
        root_folder: Path to the root folder containing bot matchup folders
        target_configs: Target number of configs expected per bot matchup. If provided,
                        will show REACH/NOT REACH for config count.
        target_games: Target number of games to reach per config. If provided, will show
                      REACH/NOT REACH for each config's game count.
    """
    import polars as pl

    # Find all bot vs bot folders
    bot_folders = sorted([
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d)) and "_vs_" in d
    ])

    for bot_folder in bot_folders:
        bot_path = os.path.join(root_folder, bot_folder)

        # Find all Timer_* configuration folders
        config_folders = sorted([
            d for d in os.listdir(bot_path)
            if os.path.isdir(os.path.join(bot_path, d)) and d.startswith("Timer_")
        ])

        # Count total configs
        total_configs = len(config_folders)

        # Print bot folder header with config count status
        if target_configs is not None:
            config_status = "REACH" if total_configs >= target_configs else "CONFIG NOT REACH"
            print(f"\n{bot_folder} ({total_configs}/{target_configs} configs - {config_status}):")
        else:
            print(f"\n{bot_folder} ({total_configs} configs):")

        # Second pass: print details for each config
        for config_folder in config_folders:
            config_path = os.path.join(bot_path, config_folder)
            parquet_file = os.path.join(config_path, f"{config_folder}.parquet")

            if os.path.isfile(parquet_file):
                # Read parquet and count unique GameIndex values
                df = pl.read_parquet(parquet_file)
                n_games = df.select("GameIndex").n_unique()

                # Add emoji indicator if target is provided
                if target_games is not None:
                    emoji = "(REACH)" if n_games >= target_games else f"(GAME NOT REACH)"
                    print(f"  {config_folder}: {n_games} games - {emoji}")
                else:
                    print(f"  - {config_folder}: {n_games} games")
            else:
                emoji = "❌" if target_games is not None else "-"
                print(f"  {emoji} {config_folder}: MISSING PARQUET FILE")


def find_missing_config_indices(res_file: str, mapping_file: str, output_file: str = None):
    """
    Parse res.txt to find configs with GAME NOT REACH, then look up their indices
    from config_index_mapping.txt to determine startConfig and endConfig ranges.

    Args:
        res_file: Path to res.txt (UTF-16LE encoded file with game counts)
        mapping_file: Path to config_index_mapping.txt (ASCII file with config indices)
        output_file: Optional output file path. If None, prints to console.

    Returns:
        List of tuples: [(bot_matchup, config_name, index), ...]
    """
    import codecs

    # Step 1: Parse res.txt to extract configs with GAME NOT REACH
    missing_configs = []
    current_bot = None

    with codecs.open(res_file, 'r', encoding='utf-16-le') as f:
        for line in f:
            line = line.strip()

            # Check for bot matchup header (e.g., "Bot_BT_vs_Bot_DQN (144/144 configs - REACH):")
            bot_match = re.match(r'^(Bot_\w+_vs_Bot_\w+)\s+\(.*\):$', line)
            if bot_match:
                current_bot = bot_match.group(1)
                continue

            # Check for config with GAME NOT REACH
            if "GAME NOT REACH" in line and current_bot:
                # Extract config name (e.g., "Timer_45__ActInterval_0.5__Round_BestOf5__SkillLeft_Boost__SkillRight_Stone")
                config_match = re.match(r'^\s*(Timer_[\w\.]+(?:__[\w\.]+)*):.*GAME NOT REACH', line)
                if config_match:
                    config_name = config_match.group(1)
                    missing_configs.append((current_bot, config_name))

    # Step 2: Parse config_index_mapping.txt to build a lookup dictionary
    config_index_map = {}  # {(bot_matchup, config_name): index}

    with open(mapping_file, 'r', encoding='ascii') as f:
        for line in f:
            # Match lines like: "  [00123] Bot_BT_vs_Bot_DQN | Timer_15__ActInterval_0.1__Round_BestOf1__SkillLeft_Boost__SkillRight_Boost"
            match = re.match(r'^\s*\[(\d+)\]\s+(Bot_[\w\.]+_vs_Bot_[\w\.]+)\s+\|\s+(Timer_[\w\.]+(?:__[\w\.]+)*)', line)
            if match:
                index = int(match.group(1))
                bot_matchup = match.group(2)
                config_name = match.group(3)
                config_index_map[(bot_matchup, config_name)] = index

    # Step 3: Look up indices for missing configs
    missing_with_indices = []
    for bot_matchup, config_name in missing_configs:
        key = (bot_matchup, config_name)
        if key in config_index_map:
            index = config_index_map[key]
            missing_with_indices.append((bot_matchup, config_name, index))
        else:
            print(f"WARNING: Config not found in mapping: {bot_matchup} | {config_name}")

    # Step 4: Sort by index and output results
    missing_with_indices.sort(key=lambda x: x[2])

    # Prepare output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"Missing Configs Requiring Re-run")
    output_lines.append(f"Total Missing Configs: {len(missing_with_indices)}")
    output_lines.append("=" * 80)
    output_lines.append("")

    if missing_with_indices:
        output_lines.append("Individual Config Indices:")
        output_lines.append("-" * 80)
        for bot_matchup, config_name, index in missing_with_indices:
            output_lines.append(f"[{index:05d}] {bot_matchup} | {config_name}")

        # Calculate range (startConfig, endConfig)
        min_index = missing_with_indices[0][2]
        max_index = missing_with_indices[-1][2]
        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append(f"Batch Run Command:")
        output_lines.append(f"Sumobot.exe startConfig={min_index} endConfig={max_index}")
        output_lines.append("=" * 80)

    # Output to file or console
    output_text = "\n".join(output_lines)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Results written to: {output_file}")
    else:
        print(output_text)

    return missing_with_indices


def generate_config_mapping(config: dict):
    """
    Generate config index mapping based on the configuration dictionary.
    Replicates the Unity C# logic for config generation.

    Args:
        config: Dictionary with keys: Timer, ActInterval, Round, SkillLeft, SkillRight, Bots

    Returns:
        Dictionary mapping (bot_matchup, config_name) -> index
    """
    config_index_map = {}
    index = 0

    bots = config["Bots"]

    # Replicate Unity C# nested loop order
    for i in range(len(bots)):
        for j in range(len(bots)):
            if i == j:  # Skip self-matchups
                continue

            for round_system in config["Round"]:
                for timer in config["Timer"]:
                    for interval in config["ActInterval"]:
                        # If skills are defined
                        if config.get("SkillLeft") and config.get("SkillRight"):
                            for left_skill in config["SkillLeft"]:
                                for right_skill in config["SkillRight"]:
                                    bot_matchup = f"{bots[i]}_vs_{bots[j]}"
                                    config_name = f"Timer_{timer}__ActInterval_{interval}__Round_{round_system}__SkillLeft_{left_skill}__SkillRight_{right_skill}"
                                    config_index_map[(bot_matchup, config_name)] = index
                                    index += 1
                        else:
                            # No skills
                            bot_matchup = f"{bots[i]}_vs_{bots[j]}"
                            config_name = f"Timer_{timer}__ActInterval_{interval}__Round_{round_system}"
                            config_index_map[(bot_matchup, config_name)] = index
                            index += 1

    return config_index_map


def find_all_missing_configs(res_file: str, mapping_file: str = None, config: dict = None, output_file: str = None):
    """
    Find ALL missing configs including:
    1. Configs with GAME NOT REACH (insufficient games)
    2. Configs that are completely missing (CONFIG NOT REACH - not in res.txt at all)

    Args:
        res_file: Path to res.txt (UTF-16LE encoded file with game counts)
        mapping_file: Path to config_index_mapping.txt (ASCII file with config indices).
                      Either mapping_file or config must be provided.
        config: Dictionary with configuration (Timer, ActInterval, Round, SkillLeft, SkillRight, Bots).
                Either mapping_file or config must be provided.
        output_file: Optional output file path. If None, prints to console.

    Returns:
        Tuple: (configs_with_insufficient_games, completely_missing_configs)
    """
    import codecs

    # Step 1: Get ALL expected configs (either from file or generate from config)
    all_expected_configs = {}  # {(bot_matchup, config_name): index}

    if config is not None:
        # Generate mapping from config
        all_expected_configs = generate_config_mapping(config)
    elif mapping_file is not None:
        # Parse mapping file
        with open(mapping_file, 'r', encoding='ascii') as f:
            for line in f:
                match = re.match(r'^\s*\[(\d+)\]\s+(Bot_[\w\.]+_vs_Bot_[\w\.]+)\s+\|\s+(Timer_[\w\.]+(?:__[\w\.]+)*)', line)
                if match:
                    index = int(match.group(1))
                    bot_matchup = match.group(2)
                    config_name = match.group(3)
                    all_expected_configs[(bot_matchup, config_name)] = index
    else:
        raise ValueError("Either mapping_file or config must be provided")

    # Step 2: Parse res.txt to get configs that exist (with any game count)
    existing_configs = set()  # Set of (bot_matchup, config_name)
    configs_insufficient_games = []  # List of (bot_matchup, config_name, index)
    current_bot = None

    with codecs.open(res_file, 'r', encoding='utf-16-le') as f:
        for line in f:
            line = line.strip()

            # Check for bot matchup header
            bot_match = re.match(r'^(Bot_\w+_vs_Bot_\w+)\s+\(.*\):$', line)
            if bot_match:
                current_bot = bot_match.group(1)
                continue

            # Check for config line (any config, not just GAME NOT REACH)
            if current_bot and line.startswith("Timer_"):
                config_match = re.match(r'^\s*(Timer_[\w\.]+(?:__[\w\.]+)*):.*', line)
                if config_match:
                    config_name = config_match.group(1)
                    existing_configs.add((current_bot, config_name))

                    # Check if it has insufficient games
                    if "GAME NOT REACH" in line:
                        key = (current_bot, config_name)
                        if key in all_expected_configs:
                            index = all_expected_configs[key]
                            configs_insufficient_games.append((current_bot, config_name, index))

    # Step 3: Find completely missing configs (in mapping but not in res.txt)
    completely_missing = []
    for (bot_matchup, config_name), index in all_expected_configs.items():
        if (bot_matchup, config_name) not in existing_configs:
            completely_missing.append((bot_matchup, config_name, index))

    # Step 4: Combine and sort all missing configs
    all_missing = configs_insufficient_games + completely_missing
    all_missing.sort(key=lambda x: x[2])

    # Step 5: Prepare output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"All Missing/Incomplete Configs")
    output_lines.append(f"Total: {len(all_missing)}")
    output_lines.append(f"  - Configs with insufficient games: {len(configs_insufficient_games)}")
    output_lines.append(f"  - Completely missing configs: {len(completely_missing)}")
    output_lines.append("=" * 80)
    output_lines.append("")

    if all_missing:
        output_lines.append("All Missing Config Indices (sorted by index):")
        output_lines.append("-" * 80)

        # Add blank lines between gaps in index sequence
        prev_index = None
        for bot_matchup, config_name, index in all_missing:
            # If there's a gap (difference > 1) from previous index, add blank line
            if prev_index is not None and index - prev_index > 1:
                output_lines.append("")

            status = "MISSING" if (bot_matchup, config_name) not in existing_configs else "INCOMPLETE"
            output_lines.append(f"[{index:05d}] {status:10s} | {bot_matchup} | {config_name}")
            prev_index = index

        # Calculate range
        min_index = all_missing[0][2]
        max_index = all_missing[-1][2]
        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append(f"Batch Run Command (covers all missing configs):")
        output_lines.append(f"Sumobot.exe startConfig={min_index} endConfig={max_index}")
        output_lines.append("=" * 80)
        output_lines.append("")
        output_lines.append("Note: This range may include configs that are complete.")
        output_lines.append("For precise re-run, use individual indices or run in smaller batches.")

    # Output to file or console
    output_text = "\n".join(output_lines)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Results written to: {output_file}")
    else:
        print(output_text)

    return (configs_insufficient_games, completely_missing)


if __name__ == "__main__":
    import sys

    # Choose which function to run
    if len(sys.argv) > 1 and sys.argv[1] == "analyze-missing":
        # Full analysis: scan converted folder + find missing configs
        from dotenv import load_dotenv

        print("="*80)
        print("FULL MISSING CONFIG ANALYSIS")
        print("="*80)
        print()

        # Load .env configuration
        load_dotenv()
        converted_dir = os.getenv("CONVERT_TARGET_DIR", "converted")
        target_games = 50  # Expected games per config (from res.txt)

        print(f"Scanning directory: {converted_dir}")
        print()

        # Configuration - Use full simulation config
        config = {
            "Timer": [15.0, 30, 45.0, 60.0],
            "ActInterval": [0.1, 0.2, 0.5],
            "Round": ["BestOf1", "BestOf3", "BestOf5"],
            "SkillLeft": ["Boost", "Stone"],
            "SkillRight": ["Boost", "Stone"],
            "Bots": ["Bot_BT", "Bot_DQN","Bot_FSM","Bot_FuzzyLogic", "Bot_GA","Bot_LLM_ActionGPT","Bot_MCTS", "Bot_ML_Classification", "Bot_NN", "Bot_PPO","Bot_Primitive","Bot_SLM_ActionGPT","Bot_UtilityAI"],
        }

        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

        # Calculate expected configs
        num_bots = len(config["Bots"])
        num_matchups = num_bots * (num_bots - 1)  # No self-matchups
        num_configs_per_matchup = (
            len(config["Timer"]) *
            len(config["ActInterval"]) *
            len(config["Round"]) *
            len(config["SkillLeft"]) *
            len(config["SkillRight"])
        )
        total_expected_configs = num_matchups * num_configs_per_matchup

        print(f"Expected configurations:")
        print(f"  Bots: {num_bots}")
        print(f"  Matchups (no self-play): {num_matchups}")
        print(f"  Configs per matchup: {num_configs_per_matchup}")
        print(f"  Total expected configs: {total_expected_configs}")
        print()

        # Step 1: Scan converted folder and build existing configs
        print("="*80)
        print("STEP 1: Scanning converted folder for existing configs")
        print("="*80)
        print()

        if not os.path.isdir(converted_dir):
            print(f"Error: {converted_dir} not found")
            print("Please check CONVERT_TARGET_DIR in .env file")
            sys.exit(1)

        # Build existing configs by scanning directory
        existing_configs = {}  # {(bot_matchup, config_name): game_count}

        try:
            import polars as pl

            # Scan for all bot matchup folders
            bot_folders = [
                d for d in os.listdir(converted_dir)
                if os.path.isdir(os.path.join(converted_dir, d)) and "_vs_" in d
            ]

            print(f"Found {len(bot_folders)} bot matchup folders")

            for bot_folder in bot_folders:
                bot_path = os.path.join(converted_dir, bot_folder)

                # Find all Timer_* configuration folders
                config_folders = [
                    d for d in os.listdir(bot_path)
                    if os.path.isdir(os.path.join(bot_path, d)) and d.startswith("Timer_")
                ]

                for config_folder in config_folders:
                    config_path = os.path.join(bot_path, config_folder)
                    parquet_file = os.path.join(config_path, f"{config_folder}.parquet")
                    csv_file = os.path.join(config_path, f"{config_folder}.csv")

                    # Try parquet first, then csv
                    data_file = parquet_file if os.path.isfile(parquet_file) else csv_file if os.path.isfile(csv_file) else None

                    if data_file:
                        try:
                            if data_file.endswith('.parquet'):
                                df = pl.read_parquet(data_file)
                            else:
                                df = pl.read_csv(data_file)

                            n_games = df.select("GameIndex").n_unique()
                            existing_configs[(bot_folder, config_folder)] = n_games
                        except Exception as e:
                            print(f"  Warning: Could not read {data_file}: {e}")

            print(f"Found {len(existing_configs)} existing configs with data")
            print()

        except ModuleNotFoundError:
            print(f"⚠️  polars not installed, cannot count games")
            print(f"    Will only identify completely missing configs (not insufficient games)")
            print(f"    Install with: pip install polars")
            print()

        # Step 2: Find missing configs using generated mapping
        output_dir = "locals"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/missing_configs.txt"

        print("="*80)
        print("STEP 2: Comparing with expected configs")
        print("="*80)
        print()

        # Generate expected config mapping
        all_expected_configs = generate_config_mapping(config)

        # Find missing and incomplete configs
        configs_insufficient_games = []
        completely_missing = []

        for (bot_matchup, config_name), index in all_expected_configs.items():
            key = (bot_matchup, config_name)

            if key in existing_configs:
                # Config exists, check if it has enough games
                n_games = existing_configs[key]
                if n_games < target_games:
                    configs_insufficient_games.append((bot_matchup, config_name, index, n_games))
            else:
                # Config completely missing
                completely_missing.append((bot_matchup, config_name, index))

        # Combine and output
        all_missing = [(bm, cn, idx, 0) for bm, cn, idx in completely_missing] + \
                      [(bm, cn, idx, ng) for bm, cn, idx, ng in configs_insufficient_games]
        all_missing.sort(key=lambda x: x[2])  # Sort by index

        # Prepare output
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append(f"All Missing/Incomplete Configs")
        output_lines.append(f"Total: {len(all_missing)}")
        output_lines.append(f"  - Configs with insufficient games: {len(configs_insufficient_games)}")
        output_lines.append(f"  - Completely missing configs: {len(completely_missing)}")
        output_lines.append("=" * 80)
        output_lines.append("")

        if all_missing:
            output_lines.append("All Missing Config Indices (sorted by index):")
            output_lines.append("-" * 80)

            prev_index = None
            for bot_matchup, config_name, index, n_games in all_missing:
                # Add blank line if there's a gap
                if prev_index is not None and index - prev_index > 1:
                    output_lines.append("")

                status = "MISSING" if n_games == 0 else f"INCOMPLETE({n_games}/{target_games})"
                output_lines.append(f"[{index:05d}] {status:20s} | {bot_matchup} | {config_name}")
                prev_index = index

            # Calculate range
            min_index = all_missing[0][2]
            max_index = all_missing[-1][2]
            output_lines.append("")
            output_lines.append("=" * 80)
            output_lines.append(f"Batch Run Command (covers all missing configs):")
            output_lines.append(f"Sumobot.exe startConfig={min_index} endConfig={max_index}")
            output_lines.append("=" * 80)
            output_lines.append("")
            output_lines.append("Note: This range may include configs that are complete.")
            output_lines.append("For precise re-run, use individual indices or run in smaller batches.")

        # Output to file
        output_text = "\n".join(output_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"Results written to: {output_file}")
        print()
        print(f"Summary:")
        print(f"  Total missing/incomplete: {len(all_missing)}")
        print(f"  Insufficient games: {len(configs_insufficient_games)}")
        print(f"  Completely missing: {len(completely_missing)}")
        print(f"  Batch command: Sumobot.exe startConfig={min_index if all_missing else 0} endConfig={max_index if all_missing else 0}")

    elif len(sys.argv) > 1 and sys.argv[1] == "find-missing":
        # Find ALL missing config indices (both incomplete and completely missing)
        res_file = "locals/res.txt"
        mapping_file = "locals/config_index_mapping.txt"
        output_file = "locals/missing_configs.txt"

        if not os.path.isfile(res_file):
            print(f"Error: {res_file} not found")
            sys.exit(1)
        if not os.path.isfile(mapping_file):
            print(f"Error: {mapping_file} not found")
            sys.exit(1)

        find_all_missing_configs(res_file, mapping_file=mapping_file, output_file=output_file)
    else:
        # Default: count games in folder
        root_folder = "converted"
        target_configs = 5  # Default target configs per bot matchup
        target_games = 3  # Default target games per config

        # Check if the folder exists
        if not os.path.isdir(root_folder):
            print(f"Error: Folder not found: {root_folder}")
            print("Please provide the correct path to the data folder.")
        else:
            count_games_in_folder(root_folder, target_configs=target_configs, target_games=target_games)
