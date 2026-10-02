# UzToken — O'zbek tili uchun morfologik tokenizatsiya kutubxonasi

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**UzToken** — o'zbek tili uchun maxsus ishlab chiqilgan gibrid tokenizator. U qoidalarga asoslangan morfologik tahlil bilan BPE subword tokenizatsiyasini birlashtiradi.

## Xususiyatlari

- **Morfologik tahlil**: O'zak + qo'shimchalar ajratish (maktablarimizda → maktab + lar + imiz + da)
- **Morfonologik tiklash**: qishlog'imizdan → qishloq + imiz + dan (g' → q avtomatik)
- **Apostrofni to'g'ri normalizatsiya**: o', g' va tutuq belgisini (' vs ʻ vs ʼ) aniq ajratish
- **Lossless tokenizatsiya**: decode(encode(matn)) == matn — asl matn yo'qotishsiz tiklanadi
- **OOV fallback**: Bazada bo'lmagan so'zlardan qo'shimchalarni qirqish (Pythonda → Python + da)
- **BPE zaxira**: Mutlaqo notanish so'zlar uchun Byte Pair Encoding
- **72,000+ toza o'zak bazasi**: Hunspell lug'atidan tozalangan
- **Tezkor Trie ma'lumot tuzilmasi**: O(n) murakkablikda suffiks qidirish

## O'rnatish

```bash
pip install uztoken
```

Yoki manba kodidan:

```bash
git clone https://github.com/ilyosbekorolov135-svg/uztoken.git
cd uztoken
pip install -e .
```

## Foydalanish

```python
from uztoken import create_tokenizer

tok = create_tokenizer(load_dict=True)

# Morfologik tahlil
for token in tok.tokenize("Maktablarimizda yangi darsliklar keldi"):
    print(token)

# Token('Maktablarimizda' → maktab +lar(ko'plik) -> +imiz(egalik) -> +da(o'rin-payt))
# Token('yangi' → yangi [bare])
# Token('darsliklar' → dars +lik(ot yasovchi) -> +lar(ko'plik))
# Token('keldi' → kel +di(o'tgan zamon))

# Subword ro'yxati (til modeli uchun)
subwords = tok.tokenize_to_strings("O'zbekistonning go'zal qishlog'imizdan", style="hybrid")
print(subwords)
# ['ĠOʻzbekiston', '##ning', 'Ġgoʻzal', 'Ġqishloq', '##imiz', '##dan']
```

## Interaktiv test

```bash
python -X utf8 interactive.py
```

## Arxitektura

```
Kiruvchi matn
    │
    ▼
Normalizatsiya (apostrof, Unicode NFC)
    │
    ▼
Morfologik Analizator
    ├─ O'zak bazasida bor? → O'zak + Qo'shimchalar
    ├─ Morfonologik qoida? → qishlog' → qishloq
    ├─ OOV + qo'shimcha?   → Python + da
    └─ Topilmadi?          → BPE Fallback
    │
    ▼
Token ro'yxati (lossless)
```

## Litsenziya

MIT

## Muallif

Ilyosbek Orolov ([@ilyosbekorolov135-svg](https://github.com/ilyosbekorolov135-svg))