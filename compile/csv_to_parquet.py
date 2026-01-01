import os
import polars as pl
from glob import glob
from tqdm import tqdm
from datetime import datetime
import json


def convert_csv_to_parquet(csv_path: str, parquet_path: str, compression="zstd", compression_level=3):
    """Convert a single CSV file to Parquet using polars.

    Args:
        csv_path: Path to input CSV file
        parquet_path: Path to output Parquet file
        compression: Compression algorithm - "zstd" (recommended), "snappy", "gzip", "lz4", "brotli"
        compression_level: Compression level (1-22 for zstd, default 3 for balanced speed/size)
    """
    # Read CSV with polars (automatically infers types)
    # Use None for infer_schema_length to scan entire file for accurate type inference
    df = pl.read_csv(csv_path, infer_schema_length=None)

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

    # Track errors for summary at the end
    errors = []
    success_count = 0
    skipped_count = 0

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
            skipped_count += 1
            continue

        try:
            convert_csv_to_parquet(csv_path, parquet_path)
            success_count += 1
        except Exception as e:
            # Extract folder structure for better error reporting
            rel_path = os.path.relpath(csv_path, input_root)
            path_parts = rel_path.split(os.sep)

            # Try to identify matchup and config folders
            matchup_folder = path_parts[0] if len(path_parts) > 0 else "Unknown"
            config_folder = path_parts[1] if len(path_parts) > 1 else "Unknown"
            filename = os.path.basename(csv_path)

            error_info = {
                'file': csv_path,
                'matchup': matchup_folder,
                'config': config_folder,
                'filename': filename,
                'error': str(e)
            }
            errors.append(error_info)

            print(f"\n❌ ERROR:")
            print(f"   Matchup: {matchup_folder}")
            print(f"   Config:  {config_folder}")
            print(f"   File:    {filename}")
            print(f"   Error:   {e}")
            print()

    # Print summary
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    print(f"✅ Successfully converted: {success_count} files")
    print(f"⏭️  Skipped (already exists): {skipped_count} files")
    print(f"❌ Failed: {len(errors)} files")

    if errors:
        print("\n" + "="*80)
        print("ERROR DETAILS")
        print("="*80)
        for i, err in enumerate(errors, 1):
            print(f"\n{i}. Matchup: {err['matchup']}")
            print(f"   Config:  {err['config']}")
            print(f"   File:    {err['filename']}")
            print(f"   Error:   {err['error']}")
        print("="*80)

        # Save error log to compile folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"csv_to_parquet_errors_{timestamp}.log"
        log_path = os.path.join(script_dir, log_filename)

        # Also save as JSON for easier parsing
        json_filename = f"csv_to_parquet_errors_{timestamp}.json"
        json_path = os.path.join(script_dir, json_filename)

        # Write text log
        with open(log_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"CSV TO PARQUET CONVERSION ERROR LOG\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Input Root: {input_root}\n")
            f.write(f"Output Root: {output_root}\n")
            f.write(f"Inplace Mode: {inplace}\n\n")
            f.write(f"Total Files Found: {len(csv_files)}\n")
            f.write(f"Successfully Converted: {success_count}\n")
            f.write(f"Skipped (Already Exists): {skipped_count}\n")
            f.write(f"Failed: {len(errors)}\n\n")
            f.write("="*80 + "\n")
            f.write("ERROR DETAILS\n")
            f.write("="*80 + "\n\n")

            for i, err in enumerate(errors, 1):
                f.write(f"{i}. ERROR\n")
                f.write(f"   Matchup:  {err['matchup']}\n")
                f.write(f"   Config:   {err['config']}\n")
                f.write(f"   File:     {err['filename']}\n")
                f.write(f"   Full Path: {err['file']}\n")
                f.write(f"   Error:    {err['error']}\n")
                f.write("\n")

        # Write JSON log
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'input_root': input_root,
            'output_root': output_root,
            'inplace': inplace,
            'summary': {
                'total_files': len(csv_files),
                'success': success_count,
                'skipped': skipped_count,
                'failed': len(errors)
            },
            'errors': errors
        }

        with open(json_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n📝 Error logs saved:")
        print(f"   Text: {log_path}")
        print(f"   JSON: {json_path}")


if __name__ == "__main__":
    # Example usage:
    simulation_root = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"
    output_root = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"

    # Option 1: Convert in-place (parquet files alongside csv files)
    convert_all_csv_to_parquet(simulation_root, inplace=False, output_root=output_root)

    # Option 2: Convert to separate output directory
    # output_root = "/Users/defdef/Development/research/sumobot/sumobot_data/parquet_output"
    # convert_all_csv_to_parquet(simulation_root, output_root=output_root)
