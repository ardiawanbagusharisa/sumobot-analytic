import os
import polars as pl
from glob import glob
from tqdm import tqdm


def convert_csv_to_parquet(csv_path: str, parquet_path: str, compression="zstd", compression_level=3):
    """Convert a single CSV file to Parquet using polars.

    Args:
        csv_path: Path to input CSV file
        parquet_path: Path to output Parquet file
        compression: Compression algorithm - "zstd" (recommended), "snappy", "gzip", "lz4", "brotli"
        compression_level: Compression level (1-22 for zstd, default 3 for balanced speed/size)
    """
    # Read CSV with polars (automatically infers types)
    df = pl.read_csv(csv_path)

    # Write to parquet with compression
    # ZSTD level 3 provides good compression (~2-3x better than Snappy) with minimal speed penalty
    df.write_parquet(parquet_path, compression=compression, compression_level=compression_level)
    print(f"✅ Converted: {os.path.basename(csv_path)} -> {os.path.basename(parquet_path)}")


def convert_all_csv_to_parquet(input_root: str, output_root: str = None, inplace: bool = False):
    """
    Convert all CSV files in the input_root to Parquet.

    Args:
        input_root: Root directory containing CSV files
        output_root: Output directory for parquet files. If None and inplace=True,
                     saves parquet files alongside CSV files.
        inplace: If True, saves .parquet files in same location as .csv files
    """
    if inplace:
        output_root = input_root
    elif output_root is None:
        raise ValueError("Either provide output_root or set inplace=True")

    # Find all CSV files recursively
    csv_files = glob(os.path.join(input_root, "**", "*.csv"), recursive=True)

    if not csv_files:
        print(f"No CSV files found in {input_root}")
        return

    print(f"Found {len(csv_files)} CSV files to convert")

    for csv_path in tqdm(csv_files, desc="Converting to Parquet", ncols=100):
        # Determine output path
        if inplace:
            parquet_path = csv_path.replace(".csv", ".parquet")
        else:
            # Preserve folder structure in output_root
            rel_path = os.path.relpath(csv_path, input_root)
            parquet_path = os.path.join(output_root, rel_path.replace(".csv", ".parquet"))

            # Create output directory if needed
            os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

        # Skip if parquet already exists
        if os.path.exists(parquet_path):
            print(f"⏭️  Skipped: {os.path.basename(parquet_path)} (already exists)")
            continue

        try:
            convert_csv_to_parquet(csv_path, parquet_path)
        except Exception as e:
            print(f"❌ Error converting {csv_path}: {e}")


def compare_file_sizes(directory: str):
    """Compare total sizes of CSV vs Parquet files in a directory."""
    csv_files = glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    parquet_files = glob(os.path.join(directory, "**", "*.parquet"), recursive=True)

    csv_size = sum(os.path.getsize(f) for f in csv_files)
    parquet_size = sum(os.path.getsize(f) for f in parquet_files)

    print(f"\n📊 File Size Comparison:")
    print(f"CSV files: {csv_size / 1024 / 1024:.2f} MB ({len(csv_files)} files)")
    print(f"Parquet files: {parquet_size / 1024 / 1024:.2f} MB ({len(parquet_files)} files)")

    if csv_size > 0:
        reduction = (1 - parquet_size / csv_size) * 100
        print(f"Space saved: {reduction:.1f}%")


if __name__ == "__main__":
    # Example usage:
    simulation_root = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"

    # Option 1: Convert in-place (parquet files alongside csv files)
    convert_all_csv_to_parquet(simulation_root, inplace=True)

    # Option 2: Convert to separate output directory
    # output_root = "/Users/defdef/Development/research/sumobot/sumobot_data/parquet_output"
    # convert_all_csv_to_parquet(simulation_root, output_root=output_root)

    # Compare file sizes
    compare_file_sizes(simulation_root)
