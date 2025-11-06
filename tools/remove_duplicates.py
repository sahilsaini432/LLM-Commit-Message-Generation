import json
import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <ground_truth.jsonl> <generated.jsonl>")
        sys.exit(1)

    ground_truth_file = sys.argv[1]
    generated_file = sys.argv[2]

    # Step 1: Collect all unique "sha" values from the ground truth file
    sha_set = set()
    try:
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if "sha" in obj:
                            sha_set.add(obj["sha"])
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON in {ground_truth_file}: {e}")
                        sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {ground_truth_file}")
        sys.exit(1)

    # Step 2: Filter the generated file, removing entries where "sha" is in sha_set
    # Also deduplicate within the generated file by "sha"
    filtered_lines = []
    seen_sha = set()
    try:
        with open(generated_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if "sha" in obj and obj["sha"] not in seen_sha:
                            filtered_lines.append(line)
                            seen_sha.add(obj["sha"])
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON in {generated_file}: {e}")
                        sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {generated_file}")
        sys.exit(1)

    # Step 3: Write the filtered lines back to the generated file
    try:
        with open(generated_file, "w", encoding="utf-8") as f:
            for line in filtered_lines:
                f.write(line + "\n")
        print(
            f"Filtered {generated_file} successfully. Removed duplicates based on 'sha' from {ground_truth_file} and deduplicated within {generated_file}."
        )
    except Exception as e:
        print(f"Error writing to {generated_file}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
