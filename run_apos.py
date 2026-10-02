# -*- coding: utf-8 -*-
from uztoken import create_tokenizer
tok = create_tokenizer(load_dict=True)

words = [
 "o'zbekistonlik",
 "oʻzbekistonlik",
 "o’zbekistonlik",
 "g'alaba",
 "gʻalaba",
 "g’alaba",
 "a'lo",
 "ma'no",
 "mas'ul",
 "san'a",
 "e'tibor",
 "e'tiroz",
 "e'tiqod",
 "qis'm",
 "g'urur",
 "gʻurur",
 "o'rgatmoq",
 "oʻrgatmoq",
 "qo'shmoq",
 "qoʻshmoq",
]

for w in words:
    toks = tok.tokenize(w)
    t = toks[0]
    stem = getattr(t, 'stem', None)
    affs = getattr(t, 'affixes', None) or []
    parts = [stem] + [a if isinstance(a, str) else getattr(a,'value',str(a)) for a in affs]
    parts = [p for p in parts if p]
    conf = getattr(t, 'confidence', None)
    print(f"{w!r}\t| breakdown: {'+'.join(parts)}\t| conf={conf}")
