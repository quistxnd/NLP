import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList


def ends_with(row, prefix):
    return not prefix or (len(row) >= len(prefix) and row[-len(prefix):] == list(prefix))


class InquisitorLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, banned_words, boosted_words, boost=5.0):
        self.boost = boost
        self.banned = []
        for w in banned_words:
            for f in (w.lower(), w.capitalize(), " " + w.lower(), " " + w.capitalize()):
                ids = tokenizer.encode(f, add_special_tokens=False)
                if ids and ids not in self.banned:
                    self.banned.append(ids)

        self.boosted = set()
        for w in boosted_words:
            for f in (" " + w.lower(), " " + w.capitalize()):
                ids = tokenizer.encode(f, add_special_tokens=False)
                if ids and len(tokenizer.decode([ids[0]]).strip()) >= 3:
                    self.boosted.add(ids[0])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for b in range(input_ids.shape[0]):
            row = input_ids[b].tolist()
            for seq in self.banned:
                if ends_with(row, seq[:-1]):
                    scores[b, seq[-1]] = -float("inf")

            recent = set(row[-8:])
            for tid in self.boosted:
                if tid not in recent and scores[b, tid] != -float("inf"):
                    scores[b, tid] += self.boost
        return scores


banned_words = (
    "компьютер компьютера компьютере компьютерный компьютерное суперкомпьютер "
    "интернет интернета программа программы программе программный программное "
    "алгоритм алгоритма робот процессор микросхема чип наука науки научный научное "
    "ученый ученые ученых физика физики математика математически смартфон телефон "
    "технология технологии технологий цифровой цифровую сервер разрешение лицензия лицензию"
).split()

boosted_words = (
    "ересь ереси еретик костер пламя молитва молитвы молитвой молиться "
    "покаяние святой священник церковь церкви дьявол сатана анафема грех "
    "сжечь очистить покарать крест божий спасение"
).split()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ai-forever/rugpt3small_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

processor = InquisitorLogitsProcessor(tokenizer, banned_words, boosted_words, boost=5.0)


def generate(prompt, lp=None, seed=43):
    torch.manual_seed(seed)
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=42, do_sample=True, temperature=0.85,
            top_k=40, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=3,
            logits_processor=lp, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


prompt = "Вчера ученые создали новейший суперкомпьютер, который способен"

print("Лабораторная работа 5. Управление генерацией текста с помощью LogitsProcessor.\n")
print(f"Входной промпт: {prompt}\n")

print("1. Обычная генерация модели:")
print(generate(prompt, seed=43))

print("\n2. Генерация с процессором (подавление технологий и уклон в инквизицию):")
print(generate(prompt, LogitsProcessorList([processor]), seed=43))
