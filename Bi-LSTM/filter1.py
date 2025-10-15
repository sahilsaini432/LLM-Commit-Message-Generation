import argparse
import json
import jsonlines
import re
import nltk

# Declare global variables at the top before any assignment or usage
global input_file, output_file, output_jsonl_file_path, jsonl_file_path


def main():
    global input_file, output_file
    parser = argparse.ArgumentParser(
        description="Preprocess commit messages for any programming language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-i", "--input_file", required=True, help="Input JSONL file with commits")
    parser.add_argument("-o", "--output_file", required=True, help="Output JSONL file path")

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    run_processing()

    messages = []
    samples = select_message_from_jsonl()
    for sample in samples:
        message = find_url(sample)
        message = find_version(message)
        message = find_rawCode(message)
        message = find_enter(message)
        message = find_table(message)
        messages.append(message)

    update_jsonl_file(messages)
    print(f"✅ Processing complete!")
    print(f"📁 Output saved to: {output_file}")


def run_processing():
    global input_file, output_file

    # Stage 1: Extract ALL file names from diffs (handles multi-file diffs)
    input_file1 = input_file
    output_file1 = "1.jsonl"

    with open(input_file1, "r") as infile, open(output_file1, "w") as outfile:
        for line in infile:
            data = json.loads(line)

            # Handle both 'diff' and 'mod_diff' fields
            diff_field = "mod_diff" if "mod_diff" in data else "diff"

            if diff_field in data:
                diff_text = data[diff_field]
                # Extract ALL file names from the diff
                file_names = extract_all_file_names(diff_text)
                data["file_names"] = file_names  # Store as list

                # For backward compatibility, also store first file name
                data["file_name"] = file_names[0] if file_names else ""

            outfile.write(json.dumps(data) + "\n")

    # Stage 2: Replace ALL file names in messages with <file_name>
    input_file2 = output_file1
    output_file2 = "2.jsonl"

    with open(input_file2, "r") as infile, open(output_file2, "w") as outfile:
        for line in infile:
            data = json.loads(line)

            if "msg" in data and "file_names" in data:
                msg = data["msg"]
                file_names = data["file_names"]

                # Replace each file name in the message
                # Sort by length (longest first) to avoid partial replacements
                file_names_sorted = sorted(file_names, key=len, reverse=True)

                for file_name in file_names_sorted:
                    if file_name and file_name in msg:
                        msg = msg.replace(file_name, " <file_name> ")

                data["msg"] = msg

            outfile.write(json.dumps(data) + "\n")

    # Stage 3: Extract function names from diffs
    input_file3 = output_file2
    output_file3 = "3.jsonl"

    process_jsonl_extract_functions(input_file3, output_file3)

    # Stage 4: Replace function names in messages
    input_file4 = output_file3
    output_file4 = "4.jsonl"

    with open(input_file4, "r", encoding="UTF-8") as infile, open(
        output_file4, "w", encoding="UTF-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line)
            if "msg" in data and "function_names" in data:
                msg = data["msg"]
                function_names = data["function_names"]
                msg = replace_function_names(msg, function_names)
                data["msg"] = msg
            outfile.write(json.dumps(data) + "\n")

    # Stage 5: Replace common tokens with <iden>
    input_file5 = output_file4
    output_file5 = "5.jsonl"

    process_jsonl_replace_tokens(input_file5, output_file5)

    # Stage 6: Clean up method name formatting
    input_file6 = output_file5
    output_file6 = "6.jsonl"

    with open(input_file6, "r", encoding="UTF-8") as infile, open(
        output_file6, "w", encoding="UTF-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line)
            if "msg" in data:
                msg = data["msg"]
                msg = replace_method_name(msg)
                data["msg"] = msg
            outfile.write(json.dumps(data) + "\n")

    global jsonl_file_path
    jsonl_file_path = output_file6
    global output_jsonl_file_path
    output_jsonl_file_path = "6.jsonl"

    # Merge results
    def load_jsonl(file):
        data = []
        with open(file, "r", encoding="utf-8") as infile:
            for line in infile:
                data.append(json.loads(line))
        return data

    data_1 = load_jsonl(output_file6)
    data_2 = load_jsonl(output_jsonl_file_path)

    if len(data_1) == len(data_2):
        for i in range(len(data_1)):
            data_1[i]["msg"] = data_2[i]["msg"]

    # Write final output
    with open(output_file, "w", encoding="utf-8") as outfile:
        for item in data_1:
            outfile.write(json.dumps(item) + "\n")


def extract_all_file_names(diff_text):
    """Extract ALL file names from a diff that may contain multiple files."""
    file_names = []

    # Find all occurrences of file paths
    # Patterns: "ppp b/path/to/file.ext" or "+++ b/path/to/file.ext"
    patterns = [
        r"ppp b[/ ]([^\n<]+?)(?:<nl>|\n|@@)",
        r"\+\+\+ b/([^\n<]+?)(?:<nl>|\n|@@)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, diff_text)
        for match in matches:
            file_path = match.strip()
            # Extract just the filename without extension
            file_path = file_path.replace("\\", "/")

            if "/" in file_path:
                file_name = file_path.split("/")[-1]
            else:
                file_name = file_path

            # Remove extension
            if "." in file_name:
                file_name = file_name.rsplit(".", 1)[0]

            if file_name and file_name not in file_names:
                file_names.append(file_name)

    return file_names


def extract_function_names(diff):
    """Extract function/method names from diff - supports multiple languages."""
    function_names = []

    # Multi-language patterns for function/method detection
    patterns = [
        # Java, C++, C#: modifier returnType functionName(
        r"\b(?:public|private|protected|static|async|virtual|override)?\s*\w+\s+(\w+)\s*\(",
        # Python: def function_name(
        r"def\s+(\w+)\s*\(",
        # JavaScript/TypeScript: function functionName( or const/let/var functionName =
        r"function\s+(\w+)\s*\(",
        r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\s*(?:function\s*)?\(",
        # Arrow functions: const name = () =>
        r"(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>",
        # Ruby: def function_name
        r"def\s+(\w+)",
        # Go: func functionName(
        r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
        # Rust: fn function_name(
        r"fn\s+(\w+)\s*(?:<[^>]+>)?\s*\(",
        # Swift: func functionName(
        r"func\s+(\w+)\s*(?:<[^>]+>)?\s*\(",
        # PHP: function functionName(
        r"function\s+(\w+)\s*\(",
        # Kotlin: fun functionName(
        r"fun\s+(\w+)\s*\(",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, diff)
        function_names.extend(matches)

    # Remove duplicates and common keywords
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
        "float",
        "double",
        "char",
        "long",
        "short",
        "byte",
        "boolean",
    }
    function_names = list(
        set([name for name in function_names if name.lower() not in keywords and len(name) > 1])
    )

    return function_names


def process_jsonl_extract_functions(input_file, output_file):
    """Extract function names and add to JSON objects."""
    with jsonlines.open(input_file) as reader:
        with jsonlines.open(output_file, mode="w") as writer:
            for obj in reader:
                # Handle both 'diff' and 'mod_diff' fields
                diff_field = "mod_diff" if "mod_diff" in obj else "diff"
                diff = obj.get(diff_field, "")

                function_names = extract_function_names(diff)
                obj["function_names"] = function_names

                writer.write(obj)


def replace_function_names(msg, function_names):
    """Replace function names in message with <method_name>."""
    # Sort by length (longest first) to avoid partial replacements
    function_names_sorted = sorted(function_names, key=len, reverse=True)

    for function_name in function_names_sorted:
        if function_name in msg:
            msg = msg.replace(function_name, " <method_name> ")
    return msg


def replace_token(msg, diff):
    """Replace common tokens appearing in both msg and diff with <iden>."""
    try:
        msg_tokens = nltk.word_tokenize(msg)
        diff_tokens = nltk.word_tokenize(diff)
        msgnew_tokens = []

        for token in msg_tokens:
            if (
                (token in diff_tokens)
                and len(token) > 5
                and (token != "<file_name>")
                and (token != "<method_name>")
            ):
                token = "<iden>"
            msgnew_tokens.append(token)

        msgnew = " ".join(msgnew_tokens)
        return msgnew
    except:
        return msg


def process_jsonl_replace_tokens(input_file, output_file):
    """Replace common tokens with <iden>."""
    with jsonlines.open(input_file) as reader:
        with jsonlines.open(output_file, mode="w") as writer:
            for obj in reader:
                diff_field = "mod_diff" if "mod_diff" in obj else "diff"

                msg = obj.get("msg", "")
                diff = obj.get(diff_field, "")

                msgnew = replace_token(msg, diff)
                obj["msg"] = msgnew

                writer.write(obj)


def replace_method_name(msg):
    """Clean up spacing around placeholders."""
    msg = msg.replace("< method_name >", "<method_name>")
    msg = msg.replace("< file_name >", "<file_name>")
    msg = msg.replace("< iden >", "<iden>")
    msg = re.sub(r"\s+", " ", msg)  # Remove extra spaces
    return msg.strip()


def select_message_from_jsonl():
    """Load messages from JSONL file."""
    with open(jsonl_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    samples = []
    for line in lines:
        data = json.loads(line)
        message = data.get("msg", "")
        samples.append(message)
    return samples


def update_jsonl_file(samples):
    """Write processed messages back to JSONL file."""
    global output_jsonl_file_path
    with open(output_jsonl_file_path, "w", encoding="utf-8") as file:
        for sample in samples:
            data = {"msg": sample}
            file.write(json.dumps(data, ensure_ascii=False) + "\n")


def find_url(message):
    """Replace URLs with <link> placeholder."""
    if "git-svn-id: " in message:
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
        message = message.replace(url, " <link> ")

    # Handle normal URLs without spaces
    pattern2 = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    urls2 = re.findall(pattern2, message)
    urls2 = sorted(list(set(urls2)), reverse=True)
    for url in urls2:
        message = message.replace(url, " <link> ")

    return message


def find_version(message):
    """Replace version numbers with <version> placeholder."""
    pattern = re.compile(r"[vVr]?\d+(?:\.\w+)+(?:-(?:\w)*){1,2}")
    versions = pattern.findall(message)
    versions = sorted(list(set(versions)), reverse=True)
    for version in versions:
        message = message.replace(version, " <version> ")

    pattern2 = re.compile(r"[vVr]?\d+(?:\s\.\s\w+)+")
    versions = pattern2.findall(message)
    versions = sorted(list(set(versions)), reverse=True)
    for version in versions:
        message = message.replace(version, " <version> ")
    return message


def find_enter(message):
    """Replace newline tokens with <enter> placeholder."""
    message = message.replace("<nl>", " <enter> ")
    message = message.replace("\n", " <enter> ")
    message = message.replace("\r", "")
    return message


def find_table(message):
    """Replace tab characters with <tab> placeholder."""
    message = message.replace("\t", " <tab> ")
    return message


def find_rawCode(message):
    """Remove code blocks enclosed in backticks."""
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
    # Download required NLTK data
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading NLTK punkt_tab...")
        nltk.download("punkt_tab")

    main()
