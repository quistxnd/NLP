import re
import gensim

pos = ["бант_NOUN", "муравейник_NOUN"]
neg = ["голова_NOUN"]

word2vec = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)

dist = word2vec.most_similar(positive=pos, negative=neg, topn=100)

pat = re.compile(r"(.*)_NOUN")

print("Результаты операций сложения и вычитания:")
print(f"Положительные слова: {pos}")
print(f"Отрицательные слова: {neg}")
print("\nПервая десятка имён существительных:")

count = 0
for i in dist:
    e = pat.match(i[0])
    if e is not None:
        count += 1
        word = e.group(1)
        sim = i[1]
        marker = " <--- ЗАДАННОЕ СЛОВО" if word in ("лента", "муравей") else ""
        print(f"{count:2d}. {word} (косинусное сходство: {sim:.4f}){marker}")
        if count == 10:
            break
