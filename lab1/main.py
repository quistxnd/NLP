import os
import sys
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pymorphy3

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def ensure_nltk_data():
    os.environ['NLTK_ALLOW_PROXIED_URLOPEN'] = '1'
    for package in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

ensure_nltk_data()

morph = pymorphy3.MorphAnalyzer()

ALLOWED_POS = {'NOUN', 'ADJF'}

def check_agreement(word1, word2):
    parses1 = morph.parse(word1)
    parses2 = morph.parse(word2)

    if parses1[0].tag.POS not in ALLOWED_POS or parses2[0].tag.POS not in ALLOWED_POS:
        return None

    pos1 = parses1[0].tag.POS
    pos2 = parses2[0].tag.POS

    for p1 in parses1:
        if p1.tag.POS != pos1:
            continue
        for p2 in parses2:
            if p2.tag.POS != pos2:
                continue

            if not p1.tag.number or p1.tag.number != p2.tag.number:
                continue

            if not p1.tag.case or p1.tag.case != p2.tag.case:
                continue

            # Во множественном числе род у прилагательных не выражен
            if p1.tag.number == 'plur':
                return (pos1, pos2), p1.normal_form, p2.normal_form
            elif p1.tag.gender and p1.tag.gender == p2.tag.gender:
                return (pos1, pos2), p1.normal_form, p2.normal_form

    return None

def process_text(text):
    groups = {
        ('ADJF', 'NOUN'): [],
        ('NOUN', 'ADJF'): [],
        ('ADJF', 'ADJF'): [],
        ('NOUN', 'NOUN'): []
    }

    sentences = sent_tokenize(text)

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        words = [t for t in tokens if any(c.isalpha() for c in t)]

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            match = check_agreement(w1, w2)

            if match:
                combo, lemma1, lemma2 = match
                entry = f"{w1.lower()} {w2.lower()} | {lemma1} {lemma2}"

                if entry not in groups[combo]:
                    groups[combo].append(entry)

    return groups

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")

    if not os.path.exists(file_path):
        print(f"Ошибка: входной файл не найден: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    results = process_text(text)

    titles = [
        (('ADJF', 'NOUN'), "1. Прилагательное + существительное (ADJF + NOUN):"),
        (('NOUN', 'ADJF'), "2. Существительное + прилагательное (NOUN + ADJF):"),
        (('ADJF', 'ADJF'), "3. Прилагательное + прилагательное (ADJF + ADJF):"),
        (('NOUN', 'NOUN'), "4. Существительное + существительное (NOUN + NOUN):")
    ]

    for combo, title in titles:
        print(title)
        pairs = results.get(combo, [])
        if pairs:
            for pair in pairs:
                print(pair)
        else:
            print("(пар не найдено)")
        print()

if __name__ == "__main__":
    main()
