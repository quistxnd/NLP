from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

name = 'DeepPavlov/rubert-base-cased-conversational'
tokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained(name, return_dict=True)

# Предложение с маской
left_context = "К завтраку один был кислый пирожок, а другой — "
right_context = "."
text = left_context + tokenizer.mask_token + right_context

inputs = tokenizer._encode_plus(text, return_tensors="pt")
mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)
output = model(**inputs)

logits = output.logits
softmax = F.softmax(logits, dim=-1)
mask_word = softmax[0, mask_index[0], :]
top = torch.topk(mask_word, 10)

target_words = ["кислый", "пряник"]

predictions = []
for token in top[-1][0].data:
    predictions.append(tokenizer.decode([token]))

print(f"Используемая модель: {name}")
print(f"Левый контекст: {left_context}")
print(f"Правый контекст: {right_context}")
print(f"Полное предложение: {text}\n")

print("Десять самых вероятных слов:")
for i, word in enumerate(predictions, 1):
    print(f"{i}. {word}")

print()
for word in target_words:
    if word in predictions:
        pos = predictions.index(word) + 1
        print(f"Слово {word} найдено на {pos} месте.")
    else:
        print(f"Слово {word} не найдено в первой десятке.")
