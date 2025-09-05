
# Ensure 'transformers' is installed
import subprocess
import sys
try:
    import transformers
except ImportError:
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'transformers'])
    import transformers

from transformers import pipeline

# Load small GPT-2 model
generator = pipeline("text-generation", model="gpt2", truncation=True)

# Generate text
prompt = "Artificial intelligence will"
output = generator(prompt, max_length=30, num_return_sequences=1)

print(output[0]["generated_text"])
