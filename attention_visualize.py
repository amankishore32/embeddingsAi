from transformers import AutoModel, AutoTokenizer
import torch
import seaborn as sns
import matplotlib.pyplot as plt

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

sentence = "The bank will collapse if it is unstable"

inputs = tokenizer(sentence, return_tensors="pt")

outputs = model(**inputs)
attentions = outputs.attentions

attn = attentions[0][0][0].detach().numpy()

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

plt.figure(figsize=(8, 6))
sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
            cmap="Blues", annot=True, fmt=".2f")
plt.title("Attention Heatmap (Layer 1, Head 1)")
plt.show()
