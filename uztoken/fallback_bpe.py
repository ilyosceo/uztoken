"""Lug'atda yo'q (OOV) so'zlar uchun kichik BPE zaxirasi.

Lug'atga asoslangan morfologik tahlil bilmagan so'zlar (nomlar, yangi/texnik so'zlar) avval
harfma-harf kodlanardi (~10 token/so'z). Bu modul ularni o'zbekcha o'zaklardan o'rganilgan
bo'laklarga ajratadi. Merjlar `benchmarks/train_bpe_fallback.py` bilan paketdagi o'zaklar
ro'yxatidan o'rgatiladi (tashqi korpus kerak emas) va `data/bpe_merges.json` da saqlanadi.
So'zlar kichik harfda va kanonik apostrofda (ʻ, ʼ) beriladi.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_MERGES = Path(__file__).parent / "data" / "bpe_merges.json"


def train_merges(words: Iterable[str], num_merges: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Har bir noyob so'z teng vazn bilan hisoblanadi. (alifbo, merjlar) qaytaradi."""
    wc = Counter(words)
    seqs = [list(w) for w in wc]
    freqs = list(wc.values())
    alphabet = sorted({c for w in wc for c in w})
    stats: Dict[Tuple[str, str], int] = defaultdict(int)
    index: Dict[Tuple[str, str], set] = defaultdict(set)
    for i, w in enumerate(seqs):
        for p in zip(w, w[1:]):
            stats[p] += freqs[i]
            index[p].add(i)
    merges: List[Tuple[str, str]] = []
    for _ in range(num_merges):
        if not stats:
            break
        best = max(stats, key=lambda p: (stats[p], p))
        if stats[best] < 2:
            break
        merges.append(best)
        a, b = best
        new = a + b
        for i in list(index[best]):
            w, f = seqs[i], freqs[i]
            old = list(zip(w, w[1:]))
            for p in old:
                stats[p] -= f
            for p in set(old):
                if stats[p] <= 0:
                    del stats[p]
            nw, j = [], 0
            while j < len(w):
                if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                    nw.append(new); j += 2
                else:
                    nw.append(w[j]); j += 1
            seqs[i] = nw
            for p in zip(nw, nw[1:]):
                stats[p] += f
                index[p].add(i)
        index.pop(best, None)
        stats.pop(best, None)
    return alphabet, merges


class FallbackBPE:
    def __init__(self, alphabet: Iterable[str], merges: Iterable[Tuple[str, str]]):
        self.alphabet = set(alphabet)
        self.ranks = {tuple(m): i for i, m in enumerate(merges)}
        self.pieces = sorted(self.alphabet | {a + b for a, b in self.ranks})
        self._cache: Dict[str, Optional[List[str]]] = {}

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional["FallbackBPE"]:
        p = Path(path) if path else DEFAULT_MERGES
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf8"))
        return cls(d["alphabet"], [tuple(m) for m in d["merges"]])

    def save(self, path: Path) -> None:
        merges = [list(m) for m, _ in sorted(self.ranks.items(), key=lambda kv: kv[1])]
        Path(path).write_text(json.dumps({"alphabet": sorted(self.alphabet), "merges": merges},
                                         ensure_ascii=False), encoding="utf8")

    def encode_word(self, word: str) -> Optional[List[str]]:
        """So'zni bo'laklarga ajratadi; alifbodan tashqari belgi bo'lsa None."""
        if word in self._cache:
            return self._cache[word]
        if not word or any(c not in self.alphabet for c in word):
            self._cache[word] = None
            return None
        pieces = list(word)
        while len(pieces) > 1:
            best, best_rank = None, None
            for i in range(len(pieces) - 1):
                r = self.ranks.get((pieces[i], pieces[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            a, b = pieces[best], pieces[best + 1]
            out, i = [], 0
            while i < len(pieces):
                if i < len(pieces) - 1 and pieces[i] == a and pieces[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(pieces[i]); i += 1
            pieces = out
        if len(self._cache) < 100000:
            self._cache[word] = pieces
        return pieces
