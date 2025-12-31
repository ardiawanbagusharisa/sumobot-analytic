import os
import re
import json
import csv
from glob import glob
import shutil
from tqdm import tqdm  # optional: pip install tqdm

def extract_game_index(filename: str) -> int:
    """Extract numeric index from filename like 'game_001.json'."""
    match = re.search(r"game_(\d+)", filename)
    return int(match.group(1)) if match else -1


def escape_csv(value: str) -> str:
    """Escape CSV fields like C# version."""
    if any(c in value for c in [',', '"', '\n']):
        return '"' + value.replace('"', '""') + '"'
    return value


def safe_int(value, default="") -> str:
    """Convert to int string, return default if None/empty."""
    if value is None or value == "":
        return default
    try:
        return str(int(value))
    except (ValueError, TypeError):
        return default


def safe_float(value, default="") -> str:
    """Convert to float string with consistent precision, return default if None/empty."""
    if value is None or value == "":
        return default
    try:
        return f"{float(value):.10g}"  # Use general format, up to 10 significant digits
    except (ValueError, TypeError):
        return default


def safe_bool(value) -> str:
    """Convert boolean to '1' or '0'."""
    return "1" if value else "0"


def safe_str(value, default="") -> str:
    """Convert to string, handling None."""
    if value is None or value == "":
        return default
    return str(value)


def convert_logs_to_csv(folder_path: str, output_path: str):
    """Convert all game_*.json files in folder to one CSV."""
    csv_rows = []

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
                    "GameWinner": "2" if game_winner == "Draw" else "0" if game_winner == "Left" else "1",
                    "GameTimestamp": safe_str(game_timestamp),
                    "RoundIndex": safe_int(round_index),
                    "RoundWinner": "2" if round_winner == "Draw" else "0" if round_winner == "Left" else "1",
                    "RoundTimestamp": safe_str(round_timestamp),
                    "StartedAt": safe_float(event_log.get("StartedAt")),
                    "UpdatedAt": safe_float(event_log.get("UpdatedAt")),
                    "Actor": "0" if event_log.get("Actor") == "Left" else "1",
                }

                target = event_log.get("Target", "")
                row["Target"] = "" if target == "" else "0" if target == "Left" else "1"
                row["Category"] = safe_str(event_log.get("Category"))
                row["State"] = safe_str(event_log.get("State"))

                act = event_log.get("Data")
                if act:
                    row["Name"] = safe_str(act.get("Name"))
                    row["Duration"] = safe_float(act.get("Duration"))
                    reason = act.get("Reason")
                    row["Reason"] = "" if reason is None or str(reason) == "None" else safe_str(reason)

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

                csv_rows.append(row)

    # Collect all CSV columns
    preferred_order = [
        "GameIndex","GameWinner","GameTimestamp","RoundIndex","RoundWinner","RoundTimestamp","StartedAt","UpdatedAt","Actor","Target","Category","State","Name","Duration","Reason","BotPosX","BotPosY","BotLinv","BotAngv","BotRot","BotIsDashActive","BotIsSkillActive","BotIsOutFromArena","EnemyBotPosX","EnemyBotPosY","EnemyBotLinv","EnemyBotAngv","EnemyBotRot","EnemyBotIsDashActive","EnemyBotIsSkillActive","EnemyBotIsOutFromArena","ColActor","ColImpact","ColTieBreaker","ColLockDuration","ColBotPosX","ColBotPosY","ColBotLinv","ColBotAngv","ColBotRot","ColBotIsDashActive","ColBotIsSkillActive","ColBotIsOutFromArena","ColEnemyBotPosX","ColEnemyBotPosY","ColEnemyBotLinv","ColEnemyBotAngv","ColEnemyBotRot","ColEnemyBotIsDashActive","ColEnemyBotIsSkillActive","ColEnemyBotIsOutFromArena"
    ]
    # Merge preferred order with dynamically discovered keys
    all_keys = preferred_order + [k for k in {kk for d in csv_rows for kk in d.keys()} if k not in preferred_order]


    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(all_keys)
        for row in csv_rows:
            writer.writerow([row.get(k, "") for k in all_keys])

    print(f"✅ Saved CSV: {output_path}")


def convert_all_configs(simulation_root: str, output_root: str):
    """Convert all config folders recursively (Timer_*)."""
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
        
        output_path = os.path.join(output_folder, f"{config_name}.csv")
        
        if os.path.isfile(output_path):
            print(f"[{i}/{len(config_folders)}] Skipped {config_name} already exists")
            continue
        
        # Check if CSV exists in original location, move it instead of regenerating
        old_csv_path = os.path.join(config_folder, f"{config_name}.csv")
        if os.path.isfile(old_csv_path):
            shutil.move(old_csv_path, output_path)
            print(f"[{i}/{len(config_folders)}] Moved {config_name} to output folder")
            continue
        
        print(f"[{i}/{len(config_folders)}] Processing {config_name}")
        convert_logs_to_csv(config_folder, output_path)
        config_name = os.path.basename(config_folder)
        
        # Create output folder if it doesn't exist
        output_folder = os.path.join(output_root, parent_name,config_name)
        os.makedirs(output_folder, exist_ok=True)
        
        output_path = os.path.join(output_folder, f"{config_name}.csv")
        
        if os.path.isfile(output_path):
            print(f"[{i}/{len(config_folders)}] Skipped {config_name} already exists")
            continue
        
        # Check if CSV exists in original location, move it instead of regenerating
        old_csv_path = os.path.join(config_folder, f"{config_name}.csv")
        if os.path.isfile(old_csv_path):
            shutil.move(old_csv_path, output_path)
            print(f"[{i}/{len(config_folders)}] Moved {config_name} to output folder")
            continue
        
        print(f"[{i}/{len(config_folders)}] Processing {config_name}")
        convert_logs_to_csv(config_folder, output_path)


if __name__ == "__main__":
    # Example usage:
    # incompletes = [
    #     ["Bot_GA_vs_Bot_Primitive","Timer_60__ActInterval_0.1__Round_BestOf5__SkillLeft_Boost__SkillRight_Boost"],
    #     ["Bot_FSM_vs_Bot_Primitive","Timer_15__ActInterval_0.1__Round_BestOf3__SkillLeft_Boost__SkillRight_Boost"],
    #     ["Bot_BT_vs_Bot_UtilityAI","Timer_45__ActInterval_0.1__Round_BestOf5__SkillLeft_Stone__SkillRight_Stone"],
    #     ["Bot_UtilityAI_vs_Bot_FSM","Timer_15__ActInterval_0.1__Round_BestOf5__SkillLeft_Boost__SkillRight_Stone"]
    # ]
    simulation_root = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"
    # output_root = "C:/Simulation_CSV"
    # for inc in incompletes:
    #     name = inc[1]
    #     specific_folder = os.path.join(
    #         simulation_root,
    #         inc[0],
    #         name,
    #     )
    #     convert_logs_to_csv(specific_folder, os.path.join(specific_folder, f"{name}.csv"))


    # OR convert all configs:
    convert_all_configs(simulation_root,simulation_root)
