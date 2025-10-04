import argparse
import json
import jsonlines
import re
import nltk

# Declare global variables at the top before any assignment or usage
global input_file, lan_output_file, other_output_file, lan, output_jsonl_file_path, jsonl_file_path


def main():
    global input_file, lan_output_file, other_output_file, lan
    parser = argparse.ArgumentParser(
        description="Args to run the filter script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-i", "--input_file", help="JSON file(s) to analyze (supports wildcards)")
    parser.add_argument("-l", "--lan_output_file", help="The output file path of the selected language")
    parser.add_argument("-o", "--other_output_file", help="The output file path of the other language")
    parser.add_argument(
        "-p", "--lan", help="The programming language to filter (e.g., java, py, js, cpp, cs)"
    )

    args = parser.parse_args()

    input_file = args.input_file
    lan_output_file = args.lan_output_file
    other_output_file = args.other_output_file
    lan = args.lan

    run_processing()

    messages = []
    samples = select_message_from_jsonl()
    for sample in samples:
        # sample = sample.replace(' ','')
        message = find_url(sample)
        message = find_version(message)
        message = find_rawCode(message)
        message = find_enter(message)
        message = find_table(message)
        messages.append(message)
        # sample = str(sample)

    # sample ="src/org/junit/experimental/theories/Theories.java- Moved InitializationError to ParentRunner, since it was only used by <enter>   subclasses of ParentRunner. <enter> - Broke up TestMethod into FrameworkMethod (which makes it more clear <enter>   that these methods can also be Before, After, etc.), and <enter>   TestAnnotation (for specific information only available on the @Test <enter>   annotation). <enter> - Created TestMethodElement to encapsulate the relationship between <enter>   @Test, @Before, and @After.  This class may go away again quickly <enter> - Updated version in docs to 4.5 <enter> - Included docs about junit-dep jar"
    # sample = ['1','- Moved InitializationError to ParentRunnerhThis class may go away again quickly <enter>','src/org/junit/experimental/theories/Theories.java']
    # sample = sample.split(" ")
    # sample = sample.replace(' ','')
    # b = replace_file_name(sample)
    # print(b)

    update_jsonl_file(messages)


def run_processing():
    global input_file, lan_output_file, other_output_file, lan
    with open(input_file, "r") as infile, open(lan_output_file, "w") as lan_outfile, open(
        other_output_file, "w"
    ) as other_outfile:
        for line in infile:
            data = json.loads(line)
            if "diff" in data:
                diff_text = data["diff"]
                start_index = diff_text.find("ppp b ")
                end_index = diff_text.find("<nl>", start_index)
                if start_index != -1 and end_index != -1:
                    extracted_data = diff_text[start_index + len("ppp b ") : end_index]
                    dot_index = extracted_data.rfind(".")
                    if dot_index != -1:
                        content_between_dot_nl = extracted_data[dot_index + 1 :]
                        if content_between_dot_nl == " java ":
                            lan_outfile.write(json.dumps(data) + "\n")
                        elif content_between_dot_nl == " cs ":
                            lan_outfile.write(json.dumps(data) + "\n")
                        elif content_between_dot_nl == " js ":
                            lan_outfile.write(json.dumps(data) + "\n")
                        elif content_between_dot_nl == " py ":
                            lan_outfile.write(json.dumps(data) + "\n")
                        elif content_between_dot_nl == " cpp ":
                            lan_outfile.write(json.dumps(data) + "\n")
                        else:
                            other_outfile.write(json.dumps(data) + "\n")

    input_file1 = lan_output_file
    output_file1 = "1.jsonl"

    with open(input_file1, "r") as infile, open(output_file1, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            if "diff" in data:
                diff_text = data["diff"]
                start_index = diff_text.find("ppp b ")
                end_index = diff_text.find("<nl>", start_index)
                if start_index != -1 and end_index != -1:
                    extracted_data = diff_text[start_index + len("ppp b ") : end_index]
                    # Keep only content between the last "/" and "."
                    last_slash_index = extracted_data.rfind("/")
                    last_dot_index = extracted_data.rfind(".")
                    if last_slash_index != -1 and last_dot_index != -1:
                        extracted_data = extracted_data[last_slash_index + 1 : last_dot_index]
                    # Add extracted data to original dataset, overwriting "file_name"
                    data["file_name"] = extracted_data
            # Write the original dataset containing processed data to output file
            outfile.write(json.dumps(data) + "\n")

    # Read input JSONL file
    input_file2 = output_file1
    output_file2 = "2.jsonl"

    with open(input_file2, "r") as infile, open(output_file2, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            if "msg" in data and "file_name" in data:
                msg = data["msg"]
                file_name = data["file_name"]
                # Check if "msg" contains content corresponding to "file_name"
                if file_name in msg:
                    # Replace the original corresponding character data in msg with "<file_name>"
                    msg = msg.replace(file_name, " <file_name> ")
                # 更新数据中的"msg"字段
                data["msg"] = msg
            # 将更新后的数据写入输出文件
            outfile.write(json.dumps(data) + "\n")

    input_file3 = output_file2
    output_file3 = "3.jsonl"

    # 测试函数
    process_jsonl(input_file3, output_file3)

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
                # Replace content in msg that is contained in function_names list
                msg = replace_function_names(msg, function_names)
                # Update "msg" field in data
                data["msg"] = msg
            # Write updated data to output file
            outfile.write(json.dumps(data) + "\n")

    input_file5 = output_file4
    output_file5 = "5.jsonl"

    # Test function
    process_jsonl(input_file5, output_file5)

    input_file6 = output_file5
    output_file6 = "6.jsonl"

    with open(input_file6, "r", encoding="UTF-8") as infile, open(
        output_file6, "w", encoding="UTF-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line)
            if "msg" in data:
                msg = data["msg"]
                # Replace "< method_name >" with "<method_name>"
                msg = replace_method_name(msg)
                # Update "msg" field in data
                data["msg"] = msg
            # Write updated data to output file
            outfile.write(json.dumps(data) + "\n")

    global jsonl_file_path
    jsonl_file_path = output_file6
    global output_jsonl_file_path
    output_jsonl_file_path = "6.jsonl"

    # Read first and second JSONL files
    input_file_1 = output_file6
    input_file_2 = output_jsonl_file_path

    def load_jsonl(file):
        data = []
        with open(file, "r", encoding="utf-8") as infile:
            for line in infile:
                data.append(json.loads(line))
        return data

    data_1 = load_jsonl(input_file_1)
    data_2 = load_jsonl(input_file_2)

    # Replace "msg" content in first file
    if len(data_1) == len(data_2):
        for i in range(len(data_1)):
            data_1[i]["msg"] = data_2[i]["msg"]

    # Write to output file
    with open(lan_output_file, "w", encoding="utf-8") as outfile:
        for item in data_1:
            outfile.write(json.dumps(item) + "\n")


# Define a function that takes diff as parameter and returns a list containing function names
def extract_function_names(diff):
    # Define an empty list to store function names
    global lan
    function_names = []
    # Define a regular expression to match return value type and function name
    if lan == "java" or "cpp" or "csharp":
        pattern = r"\w+\s+(\w+)\s*\("
    if lan == "py":
        pattern = r"def\s+(\w+)"
    if lan == "js":
        pattern = r"function\s+(\w+)"
    # Use re module's findall method to find all strings in diff that match pattern, get a list
    matches = re.findall(pattern, diff)
    # Iterate through each string in matches list
    for match in matches:
        # Add string to function_names list
        function_names.append(match)
    # Return function_names list
    return function_names


# Define a function that takes input filename and output filename as parameters for processing and saving
def process_jsonl(input_file, lan_output_file):
    # Use jsonlines module to open input file, get a reader object
    with jsonlines.open(input_file) as reader:
        # Use jsonlines module to open output file, get a writer object
        with jsonlines.open(lan_output_file, mode="w") as writer:
            # Iterate through each json in reader object
            for obj in reader:
                # Extract the value of diff attribute
                diff = obj["diff"]
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


# Define a function that takes msg and diff as parameters, returns replaced msgnew
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
def process_jsonl(input_file, lan_output_file):
    # Use jsonlines module to open input file, get a reader object
    with jsonlines.open(input_file) as reader:
        # Use jsonlines module to open output file, get a writer object
        with jsonlines.open(lan_output_file, mode="w") as writer:
            # Iterate through each json in reader object
            for obj in reader:
                # Extract values of msg and diff attributes
                msg = obj["msg"]
                diff = obj["diff"]
                # Call replace_token function, get msgnew
                msgnew = replace_token(msg, diff)
                # print(msgnew)
                # Replace msg attribute in obj with msgnew attribute
                obj.pop("msg")
                obj["msg"] = msgnew

                # Use writer object to write obj to output file
                writer.write(obj)


def replace_method_name(msg):
    # Replace "< method_name >" with "<method_name>"
    msg = msg.replace("< method_name >", "<method_name>")
    msg = msg.replace("< file_name >", "<file_name>")
    return msg


def select_message_from_jsonl():
    # Load data from JSONL file and return
    with open(jsonl_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    samples = []
    for line in lines:
        data = json.loads(line)
        # id = data['diff_id']
        message = data["msg"]
        #        file_names = data['file_names']
        samples.append(message)
    return samples


def update_jsonl_file(samples):
    global output_jsonl_file_path
    # Write processed data back to JSONL file
    with open(output_jsonl_file_path, "w", encoding="utf-8") as file:
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
