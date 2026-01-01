"""
Upload SumoBot Simulation Parquet files to Hugging Face Hub.

This script uploads the entire Simulation_parquet folder structure to Hugging Face,
preserving the nested directory structure:
  Simulation_parquet/
  └── Bot_A_vs_Bot_B/
      └── Timer_X__ActInterval_Y__Round_Z__SkillLeft_W__SkillRight_V/
          └── *.parquet

Usage:
    python upload_to_huggingface.py

Requirements:
    pip install huggingface_hub
"""

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path
from tqdm import tqdm


def upload_simulation_to_huggingface(
    local_parquet_dir: str,
    repo_id: str,
    repo_type: str = "dataset",
    private: bool = False,
    token: str = None
):
    """
    Upload entire Simulation_parquet folder to Hugging Face Hub.

    Args:
        local_parquet_dir: Path to your local Simulation_parquet folder
        repo_id: Hugging Face repo ID (e.g., "username/sumobot-simulation-data")
        repo_type: "dataset" or "model" (use "dataset" for data repositories)
        private: Whether to create a private repository
        token: Hugging Face token (if not set, will use cached token from `huggingface-cli login`)
    """

    # Initialize Hugging Face API
    api = HfApi(token=token)

    print("="*80)
    print("UPLOADING SUMOBOT SIMULATION DATA TO HUGGING FACE")
    print("="*80)
    print(f"Local directory: {local_parquet_dir}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print()

    # Check if local directory exists
    if not os.path.exists(local_parquet_dir):
        raise FileNotFoundError(f"Directory not found: {local_parquet_dir}")

    # Create repository if it doesn't exist
    try:
        print(f"Creating repository '{repo_id}'...")
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
            token=token
        )
        print(f"✅ Repository created/verified: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        print("Continuing with upload...")

    print()

    # Count total files to upload
    parquet_files = list(Path(local_parquet_dir).rglob("*.parquet"))
    total_files = len(parquet_files)

    print(f"Found {total_files} parquet files to upload")
    print()

    if total_files == 0:
        print("❌ No parquet files found. Exiting.")
        return

    # Upload entire folder
    print("Starting upload...")
    print("Note: This may take a while depending on your dataset size and internet speed.")
    print()

    try:
        # Upload the entire folder at once
        api.upload_folder(
            folder_path=local_parquet_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=".",  # Upload directly to root (no 'data/' prefix)
            token=token,
            ignore_patterns=["*.csv", "*.log", "*.json", "__pycache__", ".git"],  # Skip non-parquet files
        )

        print()
        print("="*80)
        print("✅ UPLOAD COMPLETE!")
        print("="*80)
        print(f"View your dataset: https://huggingface.co/datasets/{repo_id}")
        print()
        print("Next steps:")
        print("1. Add a README.md (dataset card) to describe your dataset")
        print("2. Test loading the dataset using the code below:")
        print()
        print("   from datasets import load_dataset")
        print(f"   ds = load_dataset('{repo_id}')")
        print()

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure you're logged in: `huggingface-cli login`")
        print("2. Check your internet connection")
        print("3. Verify you have write access to the repository")


def create_dataset_card(repo_id: str, output_path: str = "README.md"):
    """
    Generate a README.md dataset card for Hugging Face.

    Args:
        repo_id: Your Hugging Face repo ID
        output_path: Where to save the README.md file
    """

    readme_content = f"""---
license: mit
task_categories:
- reinforcement-learning
- robotics
tags:
- sumobot
- simulation
- game-ai
- agent-evaluation
size_categories:
- 10K<n<100K
---

# SumoBot Simulation Dataset

This dataset contains simulation data from SumoBot agent matches across various configurations.

## Dataset Description

- **Repository**: [{repo_id}](https://huggingface.co/datasets/{repo_id})
- **Source**: Unity SumoBot Simulation Engine
- **Format**: Parquet files (columnar, compressed)
- **Structure**: Nested by matchup and configuration

## Dataset Structure

```
Bot_A_vs_Bot_B/
└── Timer_X__ActInterval_Y__Round_Z__SkillLeft_W__SkillRight_V/
    └── *.parquet
```

### File Naming Convention

Each parquet file corresponds to a specific matchup and configuration:
- **Matchup**: `Bot_A_vs_Bot_B` - The two agents competing
- **Configuration**: Simulation parameters
  - `Timer`: Match duration
  - `ActInterval`: Action interval (decision frequency)
  - `Round`: Round type (e.g., BestOf1, BestOf3)
  - `SkillLeft`: Left bot's special skill
  - `SkillRight`: Right bot's special skill

## Loading the Dataset

### Load All Data

```python
from datasets import load_dataset

# Load entire dataset
ds = load_dataset("{repo_id}")
```

### Load Specific Matchup

```python
import glob
from datasets import Dataset
import polars as pl

# Load specific matchup parquet files
files = glob.glob("Bot_BT_vs_Bot_DQN/**/*.parquet", recursive=True)
df = pl.read_parquet(files)
```

### Using with Polars/Pandas

```python
import polars as pl
from huggingface_hub import hf_hub_download

# Download specific file
file_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="Bot_BT_vs_Bot_DQN/Timer_15__ActInterval_0.1__Round_BestOf1__SkillLeft_Boost__SkillRight_Boost/Timer_15__ActInterval_0.1__Round_BestOf1__SkillLeft_Boost__SkillRight_Boost.parquet",
    repo_type="dataset"
)

# Read with Polars
df = pl.read_parquet(file_path)
```

## Data Fields

Typical columns in the parquet files include:
- `timestamp`: Game timestamp
- `bot_position`: Left/Right
- `action`: Action taken by bot
- `x_position`, `y_position`: Bot coordinates
- `velocity_x`, `velocity_y`: Bot velocity
- `collision_type`: Type of collision event
- `winner`: Match outcome
- Additional game state variables

## Use Cases

This dataset can be used for:
- **Agent Evaluation**: Compare performance across different bot architectures
- **Configuration Analysis**: Determine optimal game parameters
- **Behavioral Analysis**: Study agent strategies and movement patterns
- **Reinforcement Learning**: Train or fine-tune SumoBot agents
- **Game Balance**: Analyze competitive balance across configurations
```
"""

    with open(output_path, 'w') as f:
        f.write(readme_content)

    print(f"✅ Dataset card created: {output_path}")
    print(f"Upload this to your Hugging Face repo to complete the dataset documentation")


if __name__ == "__main__":
    # ==========================================================================
    # CONFIGURATION - UPDATE THESE VALUES
    # ==========================================================================

    # Path to your local Simulation_parquet folder
    LOCAL_PARQUET_DIR = "D:/Simulation_parquet"

    # Your Hugging Face repository ID (username/repo-name)
    REPO_ID = "ardiawanbagus/sumobot-simulation-data"

    # Whether to make the repository private
    PRIVATE = False

    # Hugging Face token (optional - leave as None to use cached token)
    # Get token from: https://huggingface.co/settings/tokens
    TOKEN = None

    # ==========================================================================
    # UPLOAD
    # ==========================================================================

    print("Before running this script, make sure you're logged in:")
    print("  huggingface-cli login")
    print()

    response = input(f"Upload '{LOCAL_PARQUET_DIR}' to '{REPO_ID}'? (y/n): ")

    if response.lower() == 'y':
        # Upload dataset
        upload_simulation_to_huggingface(
            local_parquet_dir=LOCAL_PARQUET_DIR,
            repo_id=REPO_ID,
            private=PRIVATE,
            token=TOKEN
        )

        # Generate README
        print()
        print("Generating dataset card (README.md)...")
        create_dataset_card(REPO_ID, output_path="README_HUGGINGFACE.md")
        print()
        print("To upload the README:")
        print(f"1. Go to https://huggingface.co/datasets/{REPO_ID}")
        print("2. Click 'Files and versions' → 'Add file' → 'Create a new file'")
        print("3. Name it 'README.md' and paste the content from README_HUGGINGFACE.md")
        print("   Or use: huggingface-cli upload {REPO_ID} README_HUGGINGFACE.md README.md")
    else:
        print("Upload cancelled.")
