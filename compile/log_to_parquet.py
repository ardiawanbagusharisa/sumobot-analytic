import os
import re
import json
import polars as pl
from glob import glob
import shutil
from tqdm import tqdm


def extract_game_index(filename: str) -> int:
    """Extract numeric index from filename like 'game_001.json'."""
    match = re.search(r"game_(\d+)", filename)
    return int(match.group(1)) if match else -1


def safe_int(value, default=None):
    """Convert to int, return default if None/empty."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value, default=None):
    """Convert to float, return default if None/empty."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value) -> int:
    """Convert boolean to 1 or 0 (as int to match CSV format)."""
    return 1 if value else 0


def safe_str(value, default=None):
    """Convert to string, handling None."""
    if value is None or value == "":
        return default
    return str(value)


def convert_logs_to_parquet(folder_path: str, output_path: str):
    """Convert all game_*.json files in folder to one Parquet file."""
    rows = []

    files = sorted(
        glob(os.path.join(folder_path, "game_*.json")),
        key=lambda f: extract_game_index(os.path.basename(f))
    )

    for file in tqdm(files, desc=f"Processing {folder_path}", ncols=100):
        with open(file, "r", encoding="utf-8") as f:
            root = json.load(f)

        game_index = root.get("Index", -1)
        game_timestamp = root.get("Timestamp", "")
        game_winner = root.get("Winner", "")

        rounds = root.get("Rounds", [])
        for round_data in rounds:
            round_index = round_data.get("Index", -1)
            round_timestamp = round_data.get("Timestamp", "")
            round_winner = round_data.get("Winner", "")

            player_events = round_data.get("PlayerEvents", [])
            for event_log in player_events:
                if event_log.get("Category") == "LastPosition":
                    continue

                row = {
                    "GameIndex": safe_int(game_index + 1),
                    "GameWinner": 2 if game_winner == "Draw" else 0 if game_winner == "Left" else 1,
                    "GameTimestamp": safe_str(game_timestamp),
                    "RoundIndex": safe_int(round_index),
                    "RoundWinner": 2 if round_winner == "Draw" else 0 if round_winner == "Left" else 1,
                    "RoundTimestamp": safe_str(round_timestamp),
                    "StartedAt": safe_float(event_log.get("StartedAt")),
                    "UpdatedAt": safe_float(event_log.get("UpdatedAt")),
                    "Actor": 0 if event_log.get("Actor") == "Left" else 1,
                }

                target = event_log.get("Target", "")
                row["Target"] = None if target == "" else 0 if target == "Left" else 1
                row["Category"] = safe_str(event_log.get("Category"))
                row["State"] = safe_str(event_log.get("State"))

                act = event_log.get("Data")
                if act:
                    row["Name"] = safe_str(act.get("Name"))
                    row["Duration"] = safe_float(act.get("Duration"))
                    reason = act.get("Reason")
                    row["Reason"] = None if reason is None or str(reason) == "None" else safe_str(reason)

                    robot = act.get("Robot")
                    if robot:
                        pos = robot.get("Position", {})
                        row.update({
                            "BotPosX": safe_float(pos.get("X")),
                            "BotPosY": safe_float(pos.get("Y")),
                            "BotLinv": safe_float(robot.get("LinearVelocity")),
                            "BotAngv": safe_float(robot.get("AngularVelocity")),
                            "BotRot": safe_float(robot.get("Rotation")),
                            "BotIsDashActive": safe_bool(robot.get("IsDashActive")),
                            "BotIsSkillActive": safe_bool(robot.get("IsSkillActive")),
                            "BotIsOutFromArena": safe_bool(robot.get("IsOutFromArena")),
                        })

                    enemy = act.get("EnemyRobot")
                    if enemy:
                        pos = enemy.get("Position", {})
                        row.update({
                            "EnemyBotPosX": safe_float(pos.get("X")),
                            "EnemyBotPosY": safe_float(pos.get("Y")),
                            "EnemyBotLinv": safe_float(enemy.get("LinearVelocity")),
                            "EnemyBotAngv": safe_float(enemy.get("AngularVelocity")),
                            "EnemyBotRot": safe_float(enemy.get("Rotation")),
                            "EnemyBotIsDashActive": safe_bool(enemy.get("IsDashActive")),
                            "EnemyBotIsSkillActive": safe_bool(enemy.get("IsSkillActive")),
                            "EnemyBotIsOutFromArena": safe_bool(enemy.get("IsOutFromArena")),
                        })

                if event_log.get("Category") == "Collision":
                    col_data = event_log.get("Data", {})
                    row["ColActor"] = safe_bool(col_data.get("IsActor"))
                    row["ColImpact"] = safe_float(col_data.get("Impact"))
                    row["ColTieBreaker"] = safe_bool(col_data.get("IsTieBreaker"))
                    row["ColLockDuration"] = safe_float(col_data.get("LockDuration"))

                    col_robot = col_data.get("Robot")
                    if col_robot:
                        pos = col_robot.get("Position", {})
                        row.update({
                            "ColBotPosX": safe_float(pos.get("X")),
                            "ColBotPosY": safe_float(pos.get("Y")),
                            "ColBotLinv": safe_float(col_robot.get("LinearVelocity")),
                            "ColBotAngv": safe_float(col_robot.get("AngularVelocity")),
                            "ColBotRot": safe_float(col_robot.get("Rotation")),
                            "ColBotIsDashActive": safe_bool(col_robot.get("IsDashActive")),
                            "ColBotIsSkillActive": safe_bool(col_robot.get("IsSkillActive")),
                            "ColBotIsOutFromArena": safe_bool(col_robot.get("IsOutFromArena")),
                        })

                    col_enemy = col_data.get("EnemyRobot")
                    if col_enemy:
                        pos = col_enemy.get("Position", {})
                        row.update({
                            "ColEnemyBotPosX": safe_float(pos.get("X")),
                            "ColEnemyBotPosY": safe_float(pos.get("Y")),
                            "ColEnemyBotLinv": safe_float(col_enemy.get("LinearVelocity")),
                            "ColEnemyBotAngv": safe_float(col_enemy.get("AngularVelocity")),
                            "ColEnemyBotRot": safe_float(col_enemy.get("Rotation")),
                            "ColEnemyBotIsDashActive": safe_bool(col_enemy.get("IsDashActive")),
                            "ColEnemyBotIsSkillActive": safe_bool(col_enemy.get("IsSkillActive")),
                            "ColEnemyBotIsOutFromArena": safe_bool(col_enemy.get("IsOutFromArena")),
                        })

                rows.append(row)

    # Define preferred column order and schema
    preferred_order = [
        "GameIndex", "GameWinner", "GameTimestamp", "RoundIndex", "RoundWinner", "RoundTimestamp",
        "StartedAt", "UpdatedAt", "Actor", "Target", "Category", "State", "Name", "Duration", "Reason",
        "BotPosX", "BotPosY", "BotLinv", "BotAngv", "BotRot", "BotIsDashActive", "BotIsSkillActive", "BotIsOutFromArena",
        "EnemyBotPosX", "EnemyBotPosY", "EnemyBotLinv", "EnemyBotAngv", "EnemyBotRot",
        "EnemyBotIsDashActive", "EnemyBotIsSkillActive", "EnemyBotIsOutFromArena",
        "ColActor", "ColImpact", "ColTieBreaker", "ColLockDuration",
        "ColBotPosX", "ColBotPosY", "ColBotLinv", "ColBotAngv", "ColBotRot",
        "ColBotIsDashActive", "ColBotIsSkillActive", "ColBotIsOutFromArena",
        "ColEnemyBotPosX", "ColEnemyBotPosY", "ColEnemyBotLinv", "ColEnemyBotAngv", "ColEnemyBotRot",
        "ColEnemyBotIsDashActive", "ColEnemyBotIsSkillActive", "ColEnemyBotIsOutFromArena"
    ]

    # Define explicit schema to avoid type conflicts
    # Note: Using Int64 for boolean-like fields to match CSV behavior and ensure compatibility
    schema = {
        "GameIndex": pl.Int64,
        "GameWinner": pl.Int64,
        "GameTimestamp": pl.Utf8,
        "RoundIndex": pl.Int64,
        "RoundWinner": pl.Int64,
        "RoundTimestamp": pl.Utf8,
        "StartedAt": pl.Float64,
        "UpdatedAt": pl.Float64,
        "Actor": pl.Int64,
        "Target": pl.Int64,
        "Category": pl.Utf8,
        "State": pl.Utf8,  # Keep as string to match CSV format
        "Name": pl.Utf8,
        "Duration": pl.Float64,
        "Reason": pl.Utf8,
        "BotPosX": pl.Float64,
        "BotPosY": pl.Float64,
        "BotLinv": pl.Float64,
        "BotAngv": pl.Float64,
        "BotRot": pl.Float64,
        "BotIsDashActive": pl.Int64,  # Store as 0/1 to match CSV format
        "BotIsSkillActive": pl.Int64,
        "BotIsOutFromArena": pl.Int64,
        "EnemyBotPosX": pl.Float64,
        "EnemyBotPosY": pl.Float64,
        "EnemyBotLinv": pl.Float64,
        "EnemyBotAngv": pl.Float64,
        "EnemyBotRot": pl.Float64,
        "EnemyBotIsDashActive": pl.Int64,
        "EnemyBotIsSkillActive": pl.Int64,
        "EnemyBotIsOutFromArena": pl.Int64,
        "ColActor": pl.Int64,
        "ColImpact": pl.Float64,
        "ColTieBreaker": pl.Int64,
        "ColLockDuration": pl.Float64,
        "ColBotPosX": pl.Float64,
        "ColBotPosY": pl.Float64,
        "ColBotLinv": pl.Float64,
        "ColBotAngv": pl.Float64,
        "ColBotRot": pl.Float64,
        "ColBotIsDashActive": pl.Int64,
        "ColBotIsSkillActive": pl.Int64,
        "ColBotIsOutFromArena": pl.Int64,
        "ColEnemyBotPosX": pl.Float64,
        "ColEnemyBotPosY": pl.Float64,
        "ColEnemyBotLinv": pl.Float64,
        "ColEnemyBotAngv": pl.Float64,
        "ColEnemyBotRot": pl.Float64,
        "ColEnemyBotIsDashActive": pl.Int64,
        "ColEnemyBotIsSkillActive": pl.Int64,
        "ColEnemyBotIsOutFromArena": pl.Int64,
    }

    # Normalize rows to ensure all have the same columns
    for row in rows:
        for col in preferred_order:
            if col not in row:
                # Set default None for all missing columns
                row[col] = None

    # Create polars DataFrame with explicit schema
    df = pl.DataFrame(rows, schema=schema, infer_schema_length=0)

    # Select columns in preferred order
    df = df.select(preferred_order)

    # Write to parquet with zstd compression (better compression than snappy, similar speed)
    # ZSTD provides ~2-3x better compression than Snappy with minimal speed penalty
    df.write_parquet(output_path, compression="zstd", compression_level=3)
    print(f"✅ Saved Parquet: {output_path}")


def convert_all_configs(simulation_root: str, output_root: str):
    """Convert all config folders recursively (Timer_*) to Parquet files."""
    config_folders = []
    for root, dirs, _ in os.walk(simulation_root):
        for d in dirs:
            if d.startswith("Timer_"):
                config_folders.append(os.path.join(root, d))

    for i, config_folder in enumerate(config_folders, 1):
        config_name = os.path.basename(config_folder)
        parent_name = os.path.basename(os.path.dirname(config_folder))

        print(f"DEBUG: config_folder = {config_folder}")
        print(f"DEBUG: parent_name = {parent_name}")
        print(f"DEBUG: config_name = {config_name}")

        # Create output folder with parent structure if it doesn't exist
        output_folder = os.path.join(output_root, parent_name, config_name)
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, f"{config_name}.parquet")

        if os.path.isfile(output_path):
            print(f"[{i}/{len(config_folders)}] Skipped {config_name} already exists")
            continue

        # Check if Parquet exists in original location, move it instead of regenerating
        old_parquet_path = os.path.join(config_folder, f"{config_name}.parquet")
        if os.path.isfile(old_parquet_path):
            shutil.move(old_parquet_path, output_path)
            print(f"[{i}/{len(config_folders)}] Moved {config_name} to output folder")
            continue

        print(f"[{i}/{len(config_folders)}] Processing {config_name}")
        convert_logs_to_parquet(config_folder, output_path)


if __name__ == "__main__":
    # Example usage:
    simulation_root = "/Users/user_name/Library/Application Support/DefaultCompany/Sumobot/Simulation"

    # Convert all configs to parquet
    convert_all_configs(simulation_root, simulation_root)
