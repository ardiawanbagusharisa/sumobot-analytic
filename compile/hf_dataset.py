"""
Fetch existing (pre-converted) SumoBot simulation data from Hugging Face.

The dataset already has the same layout as the output of
`compile.log_to_parquet.convert_all_configs`:

    <subfolder>/
    └── BotA_vs_BotB/
        └── Timer_60__ActInterval_0.1__Round_BestOf1__SkillLeft_Boost__SkillRight_Boost/
            └── Timer_60__ActInterval_0.1__Round_BestOf1__SkillLeft_Boost__SkillRight_Boost.parquet

So the returned directory can be used directly as `converted_dir` for
`compile.generator` / `compile.arena_generator` (skip the log conversion step).

Usage (CLI):
    python3 -m compile.hf_dataset                         # full dataset (~20GB)
    python3 -m compile.hf_dataset --bots Bot_NN Bot_MCTS  # only matchups involving these bots
    python3 -m compile.hf_dataset --timers 60 --matchups Bot_NN_vs_Bot_MCTS

The dataset is public, so no token is required. Setting HF_TOKEN (or running
`huggingface-cli login`) is optional and only gives higher rate limits.
"""

import argparse
import os

from huggingface_hub import snapshot_download

REPO_ID = "ardiawanbagus/sumobot-simulation-data"
DEFAULT_SUBFOLDER = "simulated-13agents"
DEFAULT_OUTPUT_DIR = os.path.join(".analytic-cache", "converted", "hf")


def build_allow_patterns(subfolder=DEFAULT_SUBFOLDER, matchups=None, bots=None, timers=None):
    """
    Build `allow_patterns` for snapshot_download from optional filters.

    - matchups: exact matchup folder names, e.g. ["Bot_NN_vs_Bot_MCTS"]
    - bots:     any matchup where the bot plays left or right, e.g. ["Bot_NN"]
    - timers:   match timer values, e.g. [30, 60]

    matchups and bots are combined (union); timers narrows both.
    """
    matchup_globs = []
    for m in matchups or []:
        matchup_globs.append(m)
    for b in bots or []:
        matchup_globs.extend([f"{b}_vs_*", f"*_vs_{b}"])
    if not matchup_globs:
        matchup_globs = ["*"]

    config_globs = [f"Timer_{t}__*" for t in timers] if timers else ["*"]

    return [
        f"{subfolder}/{m}/{c}/*.parquet"
        for m in matchup_globs
        for c in config_globs
    ]


def pull_simulation_dataset(
    output_dir=DEFAULT_OUTPUT_DIR,
    subfolder=DEFAULT_SUBFOLDER,
    matchups=None,
    bots=None,
    timers=None,
    repo_id=REPO_ID,
    revision=None,
    max_workers=8,
):
    """
    Download (or resume/refresh) the simulation dataset from Hugging Face.

    Already-downloaded files are skipped, so calling this again is cheap.

    Returns:
        str: path to `<output_dir>/<subfolder>`, usable as `converted_dir`.
    """
    allow_patterns = build_allow_patterns(subfolder, matchups, bots, timers)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=output_dir,
        max_workers=max_workers,
    )

    converted_dir = os.path.join(output_dir, subfolder)
    if not os.path.isdir(converted_dir):
        raise FileNotFoundError(
            f"No files matched in {repo_id}/{subfolder} for patterns: {allow_patterns}"
        )
    return converted_dir


def main():
    parser = argparse.ArgumentParser(description="Pull SumoBot simulation data from Hugging Face.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subfolder", default=DEFAULT_SUBFOLDER)
    parser.add_argument("--matchups", nargs="*", help="e.g. Bot_NN_vs_Bot_MCTS")
    parser.add_argument("--bots", nargs="*", help="e.g. Bot_NN Bot_MCTS")
    parser.add_argument("--timers", nargs="*", help="e.g. 30 60")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    path = pull_simulation_dataset(
        output_dir=args.output_dir,
        subfolder=args.subfolder,
        matchups=args.matchups,
        bots=args.bots,
        timers=args.timers,
        revision=args.revision,
        max_workers=args.max_workers,
    )
    print("DONE:", path)


if __name__ == "__main__":
    main()
