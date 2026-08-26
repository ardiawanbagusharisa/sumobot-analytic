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


def _extract_event_rows(root: dict) -> list:
    """Flatten one game_*.json root's PlayerEvents into one row per event."""
    rows = []

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

    return rows


def _extract_pacing_segment_rows(root: dict) -> list:
    """
    Flatten one game_*.json root's LeftPacingSegment/RightPacingSegment into one row
    per game/round/side/segment, tagged Category="PacingSegment" so it can live
    alongside event rows (Category in {"Action", "Collision", ...}) in the same table.
    """
    rows = []

    game_index = root.get("Index", -1)
    game_winner = root.get("Winner", "")

    rounds = root.get("Rounds", [])
    for round_data in rounds:
        round_index = round_data.get("Index", -1)
        round_winner = round_data.get("Winner", "")
        round_duration = safe_float(round_data.get("Duration"))

        for side, key in (("Left", "LeftPacingSegment"), ("Right", "RightPacingSegment")):
            segments = round_data.get(key) or {}
            # Segment keys are string indices; sort numerically for correct ordering.
            sorted_segment_keys = sorted(segments.keys(), key=lambda k: int(k))
            num_segments = len(sorted_segment_keys)

            for local_idx, seg_key in enumerate(sorted_segment_keys):
                segment = segments[seg_key]
                row = {
                    "GameIndex": safe_int(game_index + 1),
                    "GameWinner": 2 if game_winner == "Draw" else 0 if game_winner == "Left" else 1,
                    "RoundIndex": safe_int(round_index),
                    "RoundWinner": 2 if round_winner == "Draw" else 0 if round_winner == "Left" else 1,
                    "RoundDuration": round_duration,
                    "Category": "PacingSegment",
                    "Side": side,
                    "SegmentIndex": safe_int(seg_key),
                    "LocalSegmentIndex": local_idx,
                    "NumSegmentsInRound": num_segments,
                    "SegmentProgress": (local_idx / (num_segments - 1)) if num_segments > 1 else 0.0,
                }
                row.update(_flatten_pacing_segment(segment))
                rows.append(row)

    return rows


# Column order: event columns first, then pacing-segment-only columns appended.
# GameIndex/GameWinner/RoundIndex/RoundWinner/Category are shared by both row types;
# every other column is null on the row type it doesn't apply to.
_PREFERRED_ORDER = [
    "GameIndex", "GameWinner", "GameTimestamp", "RoundIndex", "RoundWinner", "RoundTimestamp",
    "StartedAt", "UpdatedAt", "Actor", "Target", "Category", "State", "Name", "Duration", "Reason",
    "BotPosX", "BotPosY", "BotLinv", "BotAngv", "BotRot", "BotIsDashActive", "BotIsSkillActive", "BotIsOutFromArena",
    "EnemyBotPosX", "EnemyBotPosY", "EnemyBotLinv", "EnemyBotAngv", "EnemyBotRot",
    "EnemyBotIsDashActive", "EnemyBotIsSkillActive", "EnemyBotIsOutFromArena",
    "ColActor", "ColImpact", "ColTieBreaker", "ColLockDuration",
    "ColBotPosX", "ColBotPosY", "ColBotLinv", "ColBotAngv", "ColBotRot",
    "ColBotIsDashActive", "ColBotIsSkillActive", "ColBotIsOutFromArena",
    "ColEnemyBotPosX", "ColEnemyBotPosY", "ColEnemyBotLinv", "ColEnemyBotAngv", "ColEnemyBotRot",
    "ColEnemyBotIsDashActive", "ColEnemyBotIsSkillActive", "ColEnemyBotIsOutFromArena",
    "RoundDuration", "Side", "SegmentIndex", "LocalSegmentIndex", "NumSegmentsInRound", "SegmentProgress",
    "Tempo", "Threat", "OverallPacing", "OverallPacingNormalized", "OverallPacingPercentile",
    "TempoPercentile", "ThreatPercentile",
    "TargetTempo", "TargetThreat", "TargetOverallPacing",
    "TempoDelta", "ThreatDelta",
    "ThreatFactor_HitCollision", "ThreatFactor_Ability", "ThreatFactor_Angle", "ThreatFactor_SafeDistance",
    "TempoFactor_ActionIntensity", "TempoFactor_ActionDensity", "TempoFactor_BotsDistance", "TempoFactor_Velocity",
    "SegCollisionWindowSize", "SegCollisionCurrentCount", "SegCollisionWindowSum",
    "SegActionCount", "SegCollisionEventCount",
    "SegAvgVelocity", "SegAvgBotsDistance", "SegAvgAngle", "SegAvgSafeDistance",
]

# Explicit schema to avoid type conflicts (Int64 for boolean-like fields to match CSV
# behavior and ensure compatibility).
_SCHEMA = {
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
    "RoundDuration": pl.Float64,
    "Side": pl.Utf8,
    "SegmentIndex": pl.Int64,
    "LocalSegmentIndex": pl.Int64,
    "NumSegmentsInRound": pl.Int64,
    "SegmentProgress": pl.Float64,
    "Tempo": pl.Float64,
    "Threat": pl.Float64,
    "OverallPacing": pl.Float64,
    "OverallPacingNormalized": pl.Float64,
    "OverallPacingPercentile": pl.Float64,
    "TempoPercentile": pl.Float64,
    "ThreatPercentile": pl.Float64,
    "TargetTempo": pl.Float64,
    "TargetThreat": pl.Float64,
    "TargetOverallPacing": pl.Float64,
    "TempoDelta": pl.Float64,
    "ThreatDelta": pl.Float64,
    "ThreatFactor_HitCollision": pl.Float64,
    "ThreatFactor_Ability": pl.Float64,
    "ThreatFactor_Angle": pl.Float64,
    "ThreatFactor_SafeDistance": pl.Float64,
    "TempoFactor_ActionIntensity": pl.Float64,
    "TempoFactor_ActionDensity": pl.Float64,
    "TempoFactor_BotsDistance": pl.Float64,
    "TempoFactor_Velocity": pl.Float64,
    "SegCollisionWindowSize": pl.Int64,
    "SegCollisionCurrentCount": pl.Int64,
    "SegCollisionWindowSum": pl.Int64,
    "SegActionCount": pl.Int64,
    "SegCollisionEventCount": pl.Int64,
    "SegAvgVelocity": pl.Float64,
    "SegAvgBotsDistance": pl.Float64,
    "SegAvgAngle": pl.Float64,
    "SegAvgSafeDistance": pl.Float64,
}


def convert_logs_to_parquet(folder_path: str, output_path: str):
    """
    Convert all game_*.json files in a config folder into one flat Parquet file.

    Combines two row types in the same table, discriminated by "Category":
      - Per-event rows (Category in {"Action", "Collision", ...}) from PlayerEvents.
      - Per-segment pacing rows (Category == "PacingSegment") from
        LeftPacingSegment/RightPacingSegment, one row per game/round/side/segment.
    Both come from the same JSON parse pass, so downstream consumers that only care
    about one row type (batch_process_csvs filters Category == "Action"/"Collision";
    batch_process_pacing_segments filters Category == "PacingSegment") simply ignore
    the other's rows - the columns that don't apply to a given row type are null.
    """
    rows = []

    files = sorted(
        glob(os.path.join(folder_path, "game_*.json")),
        key=lambda f: extract_game_index(os.path.basename(f))
    )

    for file in tqdm(files, desc=f"Processing {folder_path}", ncols=100):
        with open(file, "r", encoding="utf-8") as f:
            root = json.load(f)

        rows.extend(_extract_event_rows(root))
        rows.extend(_extract_pacing_segment_rows(root))

    # Normalize rows to ensure all have the same columns
    for row in rows:
        for col in _PREFERRED_ORDER:
            if col not in row:
                row[col] = None

    # Create polars DataFrame with explicit schema
    df = pl.DataFrame(rows, schema=_SCHEMA, infer_schema_length=0)

    # Select columns in preferred order
    df = df.select(_PREFERRED_ORDER)

    # Write to parquet with zstd compression (better compression than snappy, similar speed)
    # ZSTD provides ~2-3x better compression than Snappy with minimal speed penalty
    df.write_parquet(output_path, compression="zstd", compression_level=3)
    print(f"✅ Saved Parquet: {output_path}")


def parse_pacing_folder_name(config_name: str):
    """
    Extract the dynamic pacing target name and the constraint/normalization mode
    from a config folder name.

    Folder names normally look like:
        Timer_60__ActInterval_0.1__...__Pacing_linear_decrease_0.6_to_0.4_60s|avg_bot

    The "Pacing_" segment carries the target curve identifier and, after a "|",
    the constraint set used to normalize it, e.g.:
        target = "linear_decrease_0.6_to_0.4_60s"
        constraint = "avg_bot"

    Some older/other simulation batches glue the two together with a literal
    "_constraint_" instead of "|" (no pipe at all), e.g.:
        Pacing_lin_down_06_04_constraint_avg_bot
    which without handling here would leave "avg_bot"/"nn" stuck onto the end of
    target ("lin_down_06_04_constraint_avg_bot") - a string that then can't
    exact-match any Sim_Targets/<subfolder>/<target>.json filename in
    plotting.pacing_filter_comparison.load_pacing_target_curves, silently
    breaking the Target curve lookup there (falls back to reconstructing Target
    from logged rows, which is bounded by however far any observed round reached
    rather than the full curve). Falling back to "_constraint_" as a second
    separator (only tried when "|" isn't present) fixes this at the source, so
    every downstream consumer of PacingTarget/PacingConstraint sees a clean split
    regardless of which convention the batch used.

    Returns (target, constraint) - constraint is None if neither separator is present.
    """
    pacing_segment = None
    for seg in config_name.split("__"):
        if seg.startswith("Pacing_"):
            pacing_segment = seg[len("Pacing_"):]
            break

    if pacing_segment is None:
        return None, None

    if "|" in pacing_segment:
        target, constraint = pacing_segment.split("|", 1)
    elif "_constraint_" in pacing_segment:
        target, constraint = pacing_segment.split("_constraint_", 1)
    else:
        target, constraint = pacing_segment, None

    return target, constraint


def _flatten_pacing_segment(segment: dict) -> dict:
    """Flatten a single PacingSegment entry's scalar fields (excludes raw SegmentData arrays)."""
    threat_factors = segment.get("ThreatFactors", {}) or {}
    tempo_factors = segment.get("TempoFactors", {}) or {}
    segment_data = segment.get("SegmentData", {}) or {}
    collision_data = segment_data.get("CollisionData", {}) or {}

    target_tempo = safe_float(segment.get("TargetTempo"))
    target_threat = safe_float(segment.get("TargetThreat"))
    target_overall = (
        (target_tempo + target_threat) / 2.0
        if target_tempo is not None and target_threat is not None
        else None
    )

    velocities = segment_data.get("Velocities") or []
    bots_distances = segment_data.get("BotsDistances") or []
    angles = segment_data.get("Angles") or []
    safe_distances = segment_data.get("SafeDistances") or []
    actions = segment_data.get("Actions") or []

    return {
        "Tempo": safe_float(segment.get("Tempo")),
        "Threat": safe_float(segment.get("Threat")),
        "OverallPacing": safe_float(segment.get("OverallPacing")),
        "OverallPacingNormalized": safe_float(segment.get("OverallPacingNormalized")),
        "OverallPacingPercentile": safe_float(segment.get("OverallPacingPercentile")),
        "TempoPercentile": safe_float(segment.get("TempoPercentile")),
        "ThreatPercentile": safe_float(segment.get("ThreatPercentile")),
        "TargetTempo": target_tempo,
        "TargetThreat": target_threat,
        "TargetOverallPacing": target_overall,
        "TempoDelta": safe_float(segment.get("TempoDelta")),
        "ThreatDelta": safe_float(segment.get("ThreatDelta")),
        "ThreatFactor_HitCollision": safe_float(threat_factors.get("HitCollision")),
        "ThreatFactor_Ability": safe_float(threat_factors.get("Ability")),
        "ThreatFactor_Angle": safe_float(threat_factors.get("Angle")),
        "ThreatFactor_SafeDistance": safe_float(threat_factors.get("SafeDistance")),
        "TempoFactor_ActionIntensity": safe_float(tempo_factors.get("ActionIntensity")),
        "TempoFactor_ActionDensity": safe_float(tempo_factors.get("ActionDensity")),
        "TempoFactor_BotsDistance": safe_float(tempo_factors.get("BotsDistance")),
        "TempoFactor_Velocity": safe_float(tempo_factors.get("Velocity")),
        "SegCollisionWindowSize": safe_int(collision_data.get("WindowSize")),
        "SegCollisionCurrentCount": len(collision_data.get("CurrentSegmentCollisions") or []),
        "SegCollisionWindowSum": sum(collision_data.get("WindowCollisions") or []),
        "SegActionCount": len(actions),
        "SegCollisionEventCount": len(segment_data.get("Collisions") or []),
        "SegAvgVelocity": safe_float(sum(velocities) / len(velocities)) if velocities else None,
        "SegAvgBotsDistance": safe_float(sum(bots_distances) / len(bots_distances)) if bots_distances else None,
        "SegAvgAngle": safe_float(sum(angles) / len(angles)) if angles else None,
        "SegAvgSafeDistance": safe_float(sum(safe_distances) / len(safe_distances)) if safe_distances else None,
    }


def convert_all_configs(simulation_root: str, output_root: str):
    """
    Convert all config folders recursively (Timer_*) to Parquet files.

    Each output "<config_name>.parquet" holds both event rows and, if the config's
    logs have dynamic-pacing data (LeftPacingSegment/RightPacingSegment), PacingSegment
    rows - see convert_logs_to_parquet.
    """
    config_folders = []
    for root, dirs, _ in os.walk(simulation_root):
        for d in dirs:
            if d.startswith("Timer_"):
                config_folders.append(os.path.join(root, d))

    for i, config_folder in enumerate(config_folders, 1):
        config_name = os.path.basename(config_folder)
        parent_name = os.path.basename(os.path.dirname(config_folder))

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
    simulation_root = "/Users/def/Library/Application Support/DefaultCompany/Sumobot/Simulation"
    target_root = "/Users/def/Simulation"

    # Convert all configs to parquet
    convert_all_configs(simulation_root, target_root)
