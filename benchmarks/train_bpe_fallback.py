"""OOV zaxira BPE merjlarini paketdagi o'zaklar ro'yxatidan o'rgatadi.

    python benchmarks/train_bpe_fallback.py [--merges 4000] [--out uztoken/data/bpe_merges.json]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uztoken.fallback_bpe import train_merges, FallbackBPE, DEFAULT_MERGES
from uztoken.normalizer import normalize_uzbek

STEMS = Path(__file__).resolve().parent.parent / "uztoken" / "data" / "uz_UZ_stems.txt"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--merges", type=int, default=4000)
    ap.add_argument("--out", default=str(DEFAULT_MERGES)); a = ap.parse_args()
    words = []
    for line in STEMS.read_text(encoding="utf8").splitlines():
        w = normalize_uzbek(line.strip().split("/")[0])
        if w and all(c.isalpha() or c in "ʻʼ" for c in w):
            words.append(w)
    alphabet, merges = train_merges(words, a.merges)
    FallbackBPE(alphabet, merges).save(Path(a.out))
    print(f"words={len(set(words))} alphabet={len(alphabet)} merges={len(merges)} -> {a.out}")

if __name__ == "__main__":
    main()
