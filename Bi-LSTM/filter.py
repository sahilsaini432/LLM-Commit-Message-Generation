import argparse
from ast import arg
import json
import os
from anyio import Path
import jsonlines
from tqdm import tqdm
import re
import nltk
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Args to run the filter script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-i", "--input_file", help="JSON file(s) to analyze (supports wildcards)")
    parser.add_argument("-r", "--repo", help="repo name")

    args = parser.parse_args()

    path = f"{Path(__file__).parent}/data/{args.repo}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    output_file = f"{path}/filtered_data.jsonl"

    last_file_path = run_processing(args, args.input_file, output_file)

    messages = []
    samples = select_message_from_jsonl(last_file_path)
    for sample in samples:
        # sample = sample.replace(' ','')
        message = find_url(sample)
        message = find_version(message)
        message = find_rawCode(message)
        message = find_enter(message)
        message = find_table(message)
        messages.append(message)

    update_jsonl_file(messages, output_file)


def run_processing(args, input_file, output_file):
    path = f"{Path(__file__).parent}/data/{args.repo}"

    # Stage 1 - Get All Commits with "mod_diff" Field
    print("Stage 1 - Get All Commits with mod_diff Field \n")
    with open(input_file, "r") as infile, open(output_file, "w") as lan_outfile:
        for line in tqdm(infile):
            data = json.loads(line)
            if "mod_diff" in data:
                lan_outfile.write(json.dumps(data) + "\n")

    input_file1 = output_file
    output_file1 = f"{path}/1.jsonl"

    # Stage 2 - Extract File Names from "mod_diff" and save to new field "file_name"
    print("Stage 2 - Extract File Names from mod_diff and save to new field file_name \n")
    with open(input_file1, "r") as infile, open(output_file1, "w") as outfile:
        for line in tqdm(infile):
            data = json.loads(line)
            if "mod_diff" in data:
                diff_text = data["mod_diff"]
                file_names = []

                # Find all occurrences of "ppp b "
                search_start = 0
                while True:
                    start_index = diff_text.find("ppp b ", search_start)
                    if start_index == -1:
                        break

                    end_index = diff_text.find("<nl>", start_index)
                    if end_index != -1:
                        extracted_data = diff_text[start_index + len("ppp b ") : end_index]
                        # Keep only content between the last "/" and "."
                        last_slash_index = extracted_data.rfind("/")
                        last_dot_index = extracted_data.rfind(".")
                        if last_slash_index != -1 and last_dot_index != -1:
                            file_name = extracted_data[last_slash_index + 1 : last_dot_index]
                            if file_name and file_name not in file_names:
                                file_names.append(file_name)

                    # Move search position forward
                    search_start = start_index + len("ppp b ")

                # Store all file names as a list
                data["file_name"] = file_names
                print(f"[SAHIL] {file_names}")

            # Write the original dataset containing processed data to output file
            outfile.write(json.dumps(data) + "\n")

    input_file2 = output_file1
    output_file2 = f"{path}/2.jsonl"

    # Stage 3 - Replace occurrences of any file name in "msg" with "<file_name>"
    print("Stage 3 - Replace occurrences of any file name in msg with <file_name> \n")
    with open(input_file2, "r") as infile, open(output_file2, "w") as outfile:
        for line in tqdm(infile):
            data = json.loads(line)
            if "msg" in data and "file_name" in data:
                msg = data["msg"]
                file_names = data["file_name"]

                # Check if "msg" contains any of the file names
                for file_name in file_names:
                    if file_name in msg:
                        msg = msg.replace(file_name, f" <{file_name}> ")

                data["msg"] = msg
            outfile.write(json.dumps(data) + "\n")

    input_file3 = output_file2
    output_file3 = f"{path}/3.jsonl"

    # Stage 4 - Extract function names from diff
    print("Stage 4 - Extract function names from diff \n")
    process_jsonl_extract_functions(input_file3, output_file3)

    input_file4 = output_file3
    output_file4 = f"{path}/4.jsonl"

    # Stage 5 - Replace occurrences of any function name in "msg" with "<method_name>"
    print("Stage 5 - Replace occurrences of any function name in msg with <method_name> \n")
    with open(input_file4, "r", encoding="UTF-8") as infile, open(
        output_file4, "w", encoding="UTF-8"
    ) as outfile:
        for line in tqdm(infile):
            data = json.loads(line)
            if "msg" in data and "function_names" in data:
                msg = data["msg"]
                function_names = data["function_names"]
                # Replace content in msg that is contained in function_names list
                msg = replace_function_names(msg, function_names)
                # Update "msg" field in data
                data["msg"] = msg
            # Write updated data to output file
            outfile.write(json.dumps(data) + "\n")

    input_file5 = output_file4
    output_file5 = f"{path}/5.jsonl"

    # Stage 6 - Replace common identifiers in both "msg" and "diff" with "<iden>"
    print("Stage 6 - Replace common identifiers in both msg and diff with <iden> \n")
    process_jsonl_replace_tokens(input_file5, output_file5)

    input_file6 = output_file5
    output_file6 = f"{path}/6.jsonl"

    # Stage 7 - Clean up "msg" field
    print("Stage 7 - Clean up msg field \n")
    with open(input_file6, "r", encoding="UTF-8") as infile, open(
        output_file6, "w", encoding="UTF-8"
    ) as outfile:
        for line in tqdm(infile):
            data = json.loads(line)
            if "msg" in data:
                msg = data["msg"]
                # Replace "< method_name >" with "<method_name>"
                msg = replace_method_name(msg)
                # Update "msg" field in data
                data["msg"] = msg
            # Write updated data to output file
            outfile.write(json.dumps(data) + "\n")

    output_jsonl_file_path = f"{path}/6.jsonl"

    # Read first and second JSONL files
    input_file_1 = output_file6
    input_file_2 = output_jsonl_file_path

    data_1 = load_jsonl(input_file_1)
    data_2 = load_jsonl(input_file_2)

    # Replace "msg" content in first file
    if len(data_1) == len(data_2):
        for i in range(len(data_1)):
            data_1[i]["msg"] = data_2[i]["msg"]

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for item in tqdm(data_1):
            outfile.write(json.dumps(item) + "\n")

    return output_file6


def load_jsonl(file):
    data = []
    with open(file, "r", encoding="utf-8") as infile:
        for line in infile:
            data.append(json.loads(line))
    return data


# Define a function that takes diff as parameter and returns a list containing function names
def extract_function_names(diff):
    """Extract function/method names from diff - supports multiple languages."""
    function_names = []

    # Multi-language patterns for function/method detection
    patterns = [
        # Java, C++, C#, TypeScript: modifier returnType functionName(
        r"\b(?:public|private|protected|static|async|virtual|override|abstract|final)?\s*\w+\s+(\w+)\s*\(",
        # Python: def function_name(
        r"def\s+(\w+)\s*\(",
        # JavaScript/TypeScript: function functionName(
        r"function\s+(\w+)\s*\(",
        # JavaScript/TypeScript: const/let/var functionName = function(
        r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\s*function\s*\(",
        # Arrow functions: const/let/var name = () =>
        r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\s*\([^)]*\)\s*=>",
        # Ruby: def function_name
        r"def\s+(\w+)",
        # Go: func (receiver) functionName( or func functionName(
        r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
        # Rust: fn function_name( or pub fn function_name(
        r"(?:pub\s+)?fn\s+(\w+)\s*(?:<[^>]+>)?\s*\(",
        # Swift: func functionName(
        r"func\s+(\w+)\s*(?:<[^>]+>)?\s*\(",
        # PHP: function functionName(
        r"function\s+(\w+)\s*\(",
        # Kotlin: fun functionName(
        r"fun\s+(\w+)\s*(?:<[^>]+>)?\s*\(",
        # Scala: def functionName(
        r"def\s+(\w+)\s*(?:\[[^\]]+\])?\s*\(",
        # Objective-C: - (returnType)functionName:
        r"[-+]\s*\([^)]+\)\s*(\w+)",
    ]

    # Apply all patterns to the diff
    for pattern in patterns:
        matches = re.findall(pattern, diff)
        function_names.extend(matches)

    # Remove duplicates and filter out common keywords
    keywords = {
        "if",
        "for",
        "while",
        "class",
        "return",
        "import",
        "from",
        "new",
        "this",
        "self",
        "super",
        "switch",
        "case",
        "break",
        "continue",
        "try",
        "catch",
        "finally",
        "throw",
        "async",
        "await",
        "yield",
        "lambda",
        "def",
        "var",
        "let",
        "const",
        "function",
        "void",
        "int",
        "string",
        "bool",
        "boolean",
        "float",
        "double",
        "char",
        "long",
        "short",
        "byte",
        "public",
        "private",
        "protected",
        "static",
        "final",
        "abstract",
        "interface",
        "extends",
        "implements",
        "package",
        "namespace",
        "using",
        "include",
        "require",
        "export",
        "default",
        "get",
        "set",
        "constructor",
        "destructor",
    }

    # Filter: remove keywords, duplicates, and names shorter than 2 characters
    function_names = list(
        set([name for name in function_names if name.lower() not in keywords and len(name) > 1])
    )

    return function_names


# Define a function that takes input filename and output filename as parameters for processing and saving
def process_jsonl_extract_functions(input_file, output_file):
    # Use jsonlines module to open input file, get a reader object
    with jsonlines.open(input_file) as reader:
        # Use jsonlines module to open output file, get a writer object
        with jsonlines.open(output_file, mode="w") as writer:
            # Iterate through each json in reader object
            for obj in tqdm(reader):
                # Extract the value of diff attribute
                diff = obj["mod_diff"]

                # Call extract_function_names function, get a list containing function names
                function_names = extract_function_names(diff)

                # Add a function_names attribute to obj with value as function_names list
                obj["function_names"] = function_names

                # print(function_names)
                # 用writer对象把obj写入输出文件中
                writer.write(obj)


def replace_function_names(msg, function_names):
    for function_name in function_names:
        if function_name in msg:
            msg = msg.replace(function_name, "<method_name>")
    return msg


# This function anonymizes common identifiers that appear in both the commit message and
# the diff by replacing them with a generic <iden> placeholder.
def replace_token(msg, diff):
    # Use nltk's word_tokenize method to tokenize msg and diff, get two lists
    msg_tokens = nltk.word_tokenize(msg)
    diff_tokens = nltk.word_tokenize(diff)
    # Define an empty list to store tokens of replaced msg
    msgnew_tokens = []
    # Iterate through tokens in msg
    for token in msg_tokens:
        # If this token appears in diff tokens, replace it with <iden>
        if (
            (token in diff_tokens)
            and len(token) > 5
            and (token != "<file_name>")
            and (token != "<method_name>")
        ):

            token = "<iden>"
        # Add replaced token to list
        msgnew_tokens.append(token)
    # Join tokens in list with spaces to get msgnew
    msgnew = " ".join(msgnew_tokens)
    # Return msgnew
    return msgnew


# Define a function that takes input filename and output filename as parameters for processing and saving
def process_jsonl_replace_tokens(input_file, output_file):
    # Use jsonlines module to open input file, get a reader object
    with jsonlines.open(input_file) as reader:
        # Use jsonlines module to open output file, get a writer object
        with jsonlines.open(output_file, mode="w") as writer:
            # Iterate through each json in reader object
            for obj in reader:
                # Extract values of msg and diff attributes
                msg = obj["msg"]
                diff = obj["mod_diff"]
                # Call replace_token function, get msgnew
                msgnew = replace_token(msg, diff)
                # print(msgnew)
                obj["msg"] = msgnew

                # Use writer object to write obj to output file
                writer.write(obj)


def replace_method_name(msg):
    # Replace "< method_name >" with "<method_name>"
    msg = msg.replace("< method_name >", "<method_name>")
    msg = msg.replace("< file_name >", "<file_name>")
    return msg


def select_message_from_jsonl(last_file_path):
    # Load data from JSONL file and return
    with open(last_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    samples = []
    for line in lines:
        data = json.loads(line)
        # id = data['diff_id']
        message = data["msg"]
        #        file_names = data['file_names']
        samples.append(message)
    return samples


def update_jsonl_file(samples, output_file):
    # Write processed data back to JSONL file
    with open(output_file, "w", encoding="utf-8") as file:
        for sample in samples:
            data = {
                #'diff_id': sample[0],
                "msg": sample,
                #                'file_names': sample[2]
            }
            file.write(json.dumps(data, ensure_ascii=False) + "\n")


def find_url(message):
    if "git-svn-id: " in message:
        # For git-svn-id links, handle them separately
        pattern = re.compile(
            r"git-svn-id:\s+(?:http[s]?\s:\s/\s/\s(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\s+(?:[a-z]|[0-9])+(?:-(?:[a-z]|[0-9])+){4}\s+)"
        )

    else:
        pattern = re.compile(
            r"(http[s]?\s:\s/\s/\s(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\s+)"
        )
    urls = re.findall(pattern, message)
    urls = sorted(list(set(urls)), reverse=True)
    for url in urls:
        message = message.replace(url, "<link>")
    return message


def find_version(message):
    pattern = re.compile(r"[vVr]?\d+(?:\.\w+)+(?:-(?:\w)*){1,2}")
    versions = pattern.findall(message)
    versions = sorted(list(set(versions)), reverse=True)
    for version in versions:
        message = message.replace(version, "<version>")

    pattern2 = re.compile(r"[vVr]?\d+(?:\s\.\s\w+)+")
    versions = pattern2.findall(message)
    # Remove duplicate pattern
    versions = sorted(list(set(versions)), reverse=True)
    for version in versions:
        message = message.replace(version, "<version>")
    return message


def find_enter(message):
    pattern = re.compile(r"<nl>")
    enters = pattern.findall(message)
    enters = sorted(list(set(enters)), reverse=True)
    for enter in enters:
        message = message.replace(enter, "<enter>")
    return message


def find_table(message):
    pattern = re.compile(r"\t")
    tables = pattern.findall(message)
    tables = sorted(list(set(tables)), reverse=True)
    for table in tables:
        message = message.replace(table, "<tab>")
    return message


def find_rawCode(message):
    rawCodeSta = message.find("```")
    replaceIden = []
    res = ""
    while rawCodeSta > 0:
        rawCodeEnd = message.find("```", rawCodeSta + 3, len(message))
        if rawCodeEnd != -1:
            replaceIden.append([rawCodeSta, rawCodeEnd + 3])
        else:
            break
        rawCodeSta = message.find("```", rawCodeEnd + 3, len(message))
    if len(replaceIden) > 0:
        end = 0
        for iden in replaceIden:
            res += message[end : iden[0]]
            end = iden[1]
        res += message[end : len(message)]
        return res
    else:
        return message


if __name__ == "__main__":
    main()
