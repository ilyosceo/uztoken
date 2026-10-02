# UzToken - O'zbek tili uchun morfologik tokenizatsiya kutubxonasi
# Arxitektura rejasi va tuzilishi

## Loyiha tuzilishi:
"""
uztoken/
├── pyproject.toml          # Package konfiguratsiyasi
├── README.md               # Hujjat
├── uztoken/
│   ├── __init__.py         # Public API
│   ├── normalizer.py       # O'zbek matni normalizatsiyasi
│   ├── affixes.py          # To'liq qo'shimchalar bazasi (60+)
│   ├── trie.py             # Trie ma'lumot tuzilmasi (Python + NumPy)
│   ├── _ctrie.c            # C kengaytmasi (tezlik uchun)
│   ├── dictionary.py       # Lug'at boshqaruvi
│   ├── morphology.py       # Morfologik tahlilchi
│   ├── bpe.py              # BPE tokenizator (fallback)
│   └── tokenizer.py        # Asosiy gibrid tokenizator
└── tests/
    └── test_tokenizer.py   # Testlar
"""
