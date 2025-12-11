#!/usr/bin/env python3
"""
Script to filter JSONL file and extract entries with predicted_result = 3
"""

import json
import os
import argparse


def filter_predictions(input_file):
    """
    Filter JSONL file to extract entries with specific predicted_result value

    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
        target_result (int): Value to filter by (default: 3)
    """

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    filtered_entries = []
    total_entries = 0
    labels = [0, 1, 2, 3]

    print(f"🔍 Processing file: {input_file}")

    for target_result in labels:
        filtered_entries = []
        total_entries = 0

        try:
            with open(input_file, "r", encoding="utf-8") as infile:
                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                        total_entries += 1

                        # Check if predicted_result matches target value
                        if entry.get("predicted_result") == target_result:
                            filtered_entries.append(entry)

                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping line {line_num}: Invalid JSON - {e}")
                        continue

            # Show some statistics
            if filtered_entries:
                print(f"\n📈 Statistics for predicted_result = {target_result}:")

                # Count labels in filtered results
                label_counts = {}
                for entry in filtered_entries:
                    label = entry.get("label", "unknown")
                    label_counts[label] = label_counts.get(label, 0) + 1

                print(f"   Label distribution: {dict(sorted(label_counts.items()))}")

                # Show percentage of total
                percentage = (len(filtered_entries) / total_entries) * 100 if total_entries > 0 else 0
                print(f"   Percentage of total: {percentage:.2f}%")

        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return


def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL file to extract entries with predicted_result = 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-i", "--input", required=True, help="Input JSONL file path")

    args = parser.parse_args()

    filter_predictions(args.input)


if __name__ == "__main__":
    main()
