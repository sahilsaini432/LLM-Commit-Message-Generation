#!/usr/bin/env python3
"""
Script to match entries by SHA between two JSONL files and split into train/test/val sets
"""

import json
import os
import argparse
import random
from collections import defaultdict


def load_jsonl_file(file_path):
    """
    Load JSONL file and return list of entries

    Args:
        file_path (str): Path to JSONL file

    Returns:
        list: List of JSON entries
    """
    entries = []

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return entries

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping line {line_num} in {file_path}: Invalid JSON - {e}")

    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")

    return entries


def extract_shas_from_file(entries):
    """
    Extract all SHA values from entries

    Args:
        entries (list): List of JSON entries

    Returns:
        set: Set of SHA values
    """
    shas = set()
    for entry in entries:
        sha = entry.get("sha")
        if sha:
            shas.add(sha)
    return shas


def create_sha_lookup(entries):
    """
    Create a lookup dictionary for entries by SHA

    Args:
        entries (list): List of JSON entries

    Returns:
        dict: Dictionary mapping SHA to entry
    """
    lookup = {}
    for entry in entries:
        sha = entry.get("sha")
        if sha:
            lookup[sha] = entry
    return lookup


def split_data(matched_entries, train_ratio=0.7, test_ratio=0.1, val_ratio=0.2):
    """
    Split matched entries into train/test/val sets

    Args:
        matched_entries (list): List of matched entries
        train_ratio (float): Percentage for training set
        test_ratio (float): Percentage for test set
        val_ratio (float): Percentage for validation set

    Returns:
        tuple: (train_data, test_data, val_data)
    """
    # Verify ratios sum to 1.0
    total_ratio = train_ratio + test_ratio + val_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"⚠️  Warning: Ratios sum to {total_ratio}, not 1.0. Normalizing...")
        train_ratio /= total_ratio
        test_ratio /= total_ratio
        val_ratio /= total_ratio

    # Shuffle the data for random split
    random.shuffle(matched_entries)

    total_count = len(matched_entries)
    train_count = int(total_count * train_ratio)
    test_count = int(total_count * test_ratio)
    val_count = total_count - train_count - test_count  # Remaining goes to val

    train_data = matched_entries[:train_count]
    test_data = matched_entries[train_count : train_count + test_count]
    val_data = matched_entries[train_count + test_count :]

    return train_data, test_data, val_data


def save_jsonl_file(data, output_path):
    """
    Save data to JSONL file

    Args:
        data (list): List of JSON entries
        output_path (str): Output file path
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as file:
            for entry in data:
                json.dump(entry, file, ensure_ascii=False)
                file.write("\n")
        print(f"✅ Saved {len(data)} entries to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving file {output_path}: {e}")


def match_and_split_files(
    pred_file_path, og_data_path, output_dir, train_ratio=0.7, test_ratio=0.1, val_ratio=0.2, seed=42
):
    """
    Main function to match files by SHA and split into train/test/val

    Args:
        file1_path (str): Path to first JSONL file (source of SHAs)
        file2_path (str): Path to second JSONL file (to find matching entries)
        output_dir (str): Directory to save output files
        train_ratio (float): Training set ratio
        test_ratio (float): Test set ratio
        val_ratio (float): Validation set ratio
        seed (int): Random seed for reproducible splits
    """

    # Set random seed for reproducible results
    random.seed(seed)

    print(f"🔍 Loading files...")
    print(f"   File 1 (SHA source): {pred_file_path}")
    print(f"   File 2 (data source): {og_data_path}")

    # Load both files
    file1_entries = load_jsonl_file(pred_file_path)
    file2_entries = load_jsonl_file(og_data_path)

    if not file1_entries:
        print(f"❌ No entries loaded from file 1: {pred_file_path}")
        return

    if not file2_entries:
        print(f"❌ No entries loaded from file 2: {og_data_path}")
        return

    print(f"📊 Loaded {len(file1_entries)} entries from pred file")
    print(f"📊 Loaded {len(file2_entries)} entries from og data set")

    # Extract SHAs from first file
    target_shas = extract_shas_from_file(file1_entries)
    print(f"🎯 Found {len(target_shas)} unique SHAs in pred file")

    # Create lookup for second file
    file2_lookup = create_sha_lookup(file2_entries)
    print(f"🔍 Created lookup for {len(file2_lookup)} entries in file 2")

    # Find matching entries
    matched_entries = []

    for sha in target_shas:
        if sha in file2_lookup:
            matched_entries.append(file2_lookup[sha])

    print(f"✅ Found {len(matched_entries)} matching entries")

    if not matched_entries:
        print("❌ No matching entries found. Cannot proceed with splitting.")
        return

    # Split the matched data
    print(f"📊 Splitting {len(matched_entries)} entries...")
    print(f"   Train: {train_ratio*100:.1f}% | Test: {test_ratio*100:.1f}% | Val: {val_ratio*100:.1f}%")

    train_data, test_data, val_data = split_data(matched_entries, train_ratio, test_ratio, val_ratio)

    print(f"📈 Split results:")
    print(f"   Train: {len(train_data)} entries ({len(train_data)/len(matched_entries)*100:.1f}%)")
    print(f"   Test:  {len(test_data)} entries ({len(test_data)/len(matched_entries)*100:.1f}%)")
    print(f"   Val:   {len(val_data)} entries ({len(val_data)/len(matched_entries)*100:.1f}%)")

    # Save split files
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    save_jsonl_file(train_data, train_path)
    save_jsonl_file(test_data, test_path)
    save_jsonl_file(val_data, val_path)

    print(f"🎉 Successfully created train/test/val splits in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Match JSONL files by SHA and split into train/test/val sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python match_and_split.py -f1 file1.jsonl -f2 file2.jsonl -o output/
  python match_and_split.py -f1 file1.jsonl -f2 file2.jsonl -o output/ --train 0.8 --test 0.1 --val 0.1
  python match_and_split.py -f1 file1.jsonl -f2 file2.jsonl -o output/ --seed 123
        """,
    )

    parser.add_argument("-p", "--pred_file", required=True, help="Path to first JSONL file (source of SHAs)")

    parser.add_argument(
        "-d", "--og_data", required=True, help="Path to second JSONL file (data source for matching)"
    )

    parser.add_argument("-o", "--output", required=True, help="Output directory for train/test/val files")

    parser.add_argument("--train", type=float, default=0.7, help="Training set ratio (default: 0.7)")

    parser.add_argument("--test", type=float, default=0.1, help="Test set ratio (default: 0.1)")

    parser.add_argument("--val", type=float, default=0.2, help="Validation set ratio (default: 0.2)")

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible splits (default: 42)"
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train + args.test + args.val
    if abs(total_ratio - 1.0) > 0.001:
        print(f"⚠️  Warning: Train + Test + Val ratios = {total_ratio}, not 1.0")
        print(f"   Will normalize automatically")

    match_and_split_files(
        args.pred_file, args.og_data, args.output, args.train, args.test, args.val, args.seed
    )


if __name__ == "__main__":
    main()
