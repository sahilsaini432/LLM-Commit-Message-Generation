# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("JetBrains-Research/cmg-race-without-history")
model = AutoModelForSeq2SeqLM.from_pretrained("JetBrains-Research/cmg-race-without-history")

# Example usage for prediction
# Provide a code diff or context as input
input_diff = """
diff --git a/example.py b/example.py
index 1234567..abcdef0 100644
--- a/example.py
+++ b/example.py
@@ -1,3 +1,4 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
+    return True
"""

# Tokenize input
inputs = tokenizer(input_diff, return_tensors="pt", max_length=512, truncation=True)

# Generate commit message
outputs = model.generate(
    inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True, temperature=0.7, do_sample=True
)

# Decode and print the generated message
generated_message = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_message)
