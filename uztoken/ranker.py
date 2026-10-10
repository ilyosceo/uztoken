"""O'rganilgan nomzod tanlagich (ranker).

Tahlilchi bir so'z uchun bir nechta nomzod beradi (o'zak + qo'shimchalar zanjiri). Qo'lda
sozlangan `confidence` ball ba'zan soxta qisqa o'zakni tanlaydi (ke+la+di, so+la+di+gan).
Bu modul nomzodlarni belgilar (o'zak uzunligi, qo'shimcha turlari, lug'atda uzunroq o'zak
borligi, ...) bo'yicha chiziqli model bilan qayta baholaydi. Og'irliklar UD (UzUDT train)
lemmalaridan `benchmarks/train_ranker.py` bilan o'rgatiladi va `data/ranker_weights.json`
da saqlanadi. Ish vaqtida faqat sof Python ishlatiladi.
"""
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence

_CATS = ["PLURAL", "POSSESSION", "CASE", "TENSE", "PERSON", "NEGATION", "VOICE", "MOOD",
         "PARTICIPLE", "GERUND", "WORD_FORMATION", "PARTICLE_CAT", "QUESTION", "DIMINUTIVE",
         "COMPARISON"]

FEATURE_NAMES = (
    ["confidence", "is_top_by_confidence", "rank_frac", "root_len", "root_le2", "root_eq3",
     "root_eq4", "n_suffixes", "no_suffix", "is_found", "suffix_chars_frac", "root_frac"]
    + [f"cat_{c.lower()}" for c in _CATS]
    + ["first_suffix_derivational", "root_is_verb_base", "longer_stem_prefix_exists",
       "root_is_longest_stem_prefix"]
)

DEFAULT_WEIGHTS = Path(__file__).parent / "data" / "ranker_weights.json"


def _cat_name(x) -> str:
    return str(getattr(x, "name", x)).upper()


def candidate_features(word: str, cand, rank: int, n: int, stems, prefix_hits=None) -> List[float]:
    root = cand.root
    L, wl, suf = len(root), len(word), cand.suffixes
    cats = [_cat_name(s.category) for s in suf]
    f = [
        cand.confidence, float(rank == 0), rank / max(1, n - 1), min(L, 10) / 10,
        float(L <= 2), float(L == 3), float(L == 4), float(len(suf)), float(len(suf) == 0),
        float(bool(cand.is_found)), sum(len(s.text) for s in suf) / wl, L / wl,
    ]
    f += [float(cats.count(c)) for c in _CATS]
    f.append(float(bool(suf) and "DERIVATIONAL" in _cat_name(suf[-1].affix_type)))
    f.append(float((root + "moq") in stems or (root + "imoq") in stems))
    if prefix_hits is None:
        prefix_hits = [k for k in range(1, wl) if word[:k] in stems]
    longer = any(k > L for k in prefix_hits)
    f.append(float(longer))
    f.append(float((not longer) and L >= 3))
    return f


class CandidateRanker:
    def __init__(self, weights: Sequence[float], mean: Sequence[float], std: Sequence[float]):
        self.w, self.mu, self.sd = list(weights), list(mean), list(std)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional["CandidateRanker"]:
        p = Path(path) if path else DEFAULT_WEIGHTS
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf8"))
        if d.get("features") != FEATURE_NAMES:
            return None  # belgilar ro'yxati mos emas: eski/begona og'irliklar
        return cls(d["w"], d["mu"], d["sd"])

    def score(self, feats: Sequence[float]) -> float:
        return sum(((x - m) / s) * w for x, m, s, w in zip(feats, self.mu, self.sd, self.w))

    def best(self, word: str, results: list, stems) -> object:
        """Nomzodlar (confidence bo'yicha saralangan) orasidan eng yaxshisini qaytaradi."""
        n = len(results)
        hits = [k for k in range(1, len(word)) if word[:k] in stems]  # so'z uchun bir marta
        scores = [self.score(candidate_features(word, c, i, n, stems, hits)) for i, c in enumerate(results)]
        return results[max(range(n), key=lambda i: (scores[i], -i))]
