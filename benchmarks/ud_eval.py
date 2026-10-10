"""uztoken uchun takrorlanadigan o'lchov (UD gold lemmalari + encode ko'rsatkichlari).

Ishlatish:
    python benchmarks/ud_eval.py                 # UD repo'larini ~/.cache/uztoken_ud ga yuklaydi
    python benchmarks/ud_eval.py --ud-dir DIR    # tayyor nusxadan foydalanadi
    python benchmarks/ud_eval.py --json out.json

Ma'lumot: UniversalDependencies/UD_Uzbek-UT va UD_Uzbek-UzUDT (qo'lda belgilangan lemmalar).
Gold: so'z shakli lemma bilan boshlansa (davlatlar -> davlat), o'zak chegarasi = len(lemma).
Bo'linish: DEV = UzUDT train (sozlash uchun), TEST = UT test + UzUDT test (faqat yakuniy hisob).
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uztoken import create_tokenizer
from uztoken.normalizer import normalize_apostrophes
from gold_words import GOLD_SEGMENTATIONS

UD_REPOS = ["UD_Uzbek-UT", "UD_Uzbek-UzUDT"]


def fetch_ud(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for r in UD_REPOS:
        if not (root / r).exists():
            subprocess.run(["git", "clone", "-q", "--depth", "1",
                            f"https://github.com/UniversalDependencies/{r}.git", str(root / r)], check=True)
    return root


def read_conllu(path):
    sents, toks = [], []
    for line in open(path, encoding="utf8"):
        if line.startswith("# text ="):
            sents.append(line.split("=", 1)[1].strip())
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if "-" in f[0] or "." in f[0]:
            continue
        toks.append((f[1], f[2], f[3]))
    return sents, toks


def canon(s):
    return normalize_apostrophes(s).replace("ʻ", "'").replace("ʼ", "'").lower()


def build_pairs(tokens):
    infl, root = {}, {}
    for w, lem, pos in tokens:
        if pos not in ("NOUN", "VERB", "ADJ", "PROPN"):
            continue
        w, lem = canon(w), canon(lem)
        if not re.fullmatch(r"[a-z']+", w) or len(lem) < 3:
            continue
        if w == lem:
            root[w] = lem
        elif w.startswith(lem) and len(w) - len(lem) >= 2:
            infl[w] = lem
    return infl, root


def boundaries(tok, word):
    parts = [p.replace("Ġ", "").replace("##", "").lstrip("+").replace("ʻ", "'").replace("ʼ", "'")
             for p in tok.tokenize_to_strings(word, style="plus") if p.strip()]
    b, c = set(), 0
    for p in parts[:-1]:
        c += len(p)
        b.add(c)
    return b


def seg_metrics(tok, pairs):
    infl, root = pairs
    hit = intact = both = top1 = oracle = 0
    for w, lem in infl.items():
        b = boundaries(tok, w)
        h = len(lem) in b
        i = not any(x < len(lem) for x in b)
        hit += h; intact += i; both += (h and i)
        top1 += tok.analyze(w).root.replace("ʻ", "'").replace("ʼ", "'") == lem
        oracle += any(c.root.replace("ʻ", "'").replace("ʼ", "'") == lem
                      for c in tok._analyzer.analyze_all(w, max_results=8))
    n = len(infl)
    unsplit = sum(not boundaries(tok, w) for w in root) / max(1, len(root))
    return {"n_inflected": n, "n_roots": len(root),
            "boundary_and_root_intact": 100 * both / n, "boundary_hit": 100 * hit / n,
            "root_intact": 100 * intact / n, "root_top1": 100 * top1 / n,
            "root_oracle8": 100 * oracle / n, "uninflected_unsplit": 100 * unsplit}


def gold_f1(tok):
    tp = pp = gg = 0
    for w, segs in GOLD_SEGMENTATIONS.items():
        g, c = set(), 0
        for s in segs[:-1]:
            c += len(s); g.add(c)
        b = boundaries(tok, w)
        tp += len(b & g); pp += len(b); gg += len(g)
    p, r = tp / pp, tp / gg
    return 2 * p * r / (p + r)


def encode_metrics(tok, sentences):
    t0 = time.time(); ids = [tok.encode(s) for s in sentences]; dt = time.time() - t0
    words = sum(len(s.split()) for s in sentences)
    exact = sum(tok.decode(x) == normalize_apostrophes(s) for x, s in zip(ids, sentences))
    return {"sentences": len(sentences), "fertility": sum(map(len, ids)) / words,
            "sentences_per_sec": len(sentences) / dt, "roundtrip_exact_pct": 100 * exact / len(sentences)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ud-dir"); ap.add_argument("--json")
    a = ap.parse_args()
    root = Path(a.ud_dir) if a.ud_dir else fetch_ud(Path.home() / ".cache" / "uztoken_ud")
    dev_s, dev_t = read_conllu(root / "UD_Uzbek-UzUDT" / "uz_uzudt-ud-train.conllu")
    s1, t1 = read_conllu(root / "UD_Uzbek-UT" / "uz_ut-ud-test.conllu")
    s2, t2 = read_conllu(root / "UD_Uzbek-UzUDT" / "uz_uzudt-ud-test.conllu")
    tok = create_tokenizer(load_dict=True)
    res = {"dev": seg_metrics(tok, build_pairs(dev_t)), "test": seg_metrics(tok, build_pairs(t1 + t2)),
           "gold61_f1": gold_f1(tok), "encode_dev": encode_metrics(tok, dev_s[:600]), "encode_test": encode_metrics(tok, (s1 + s2)[:600])}
    for k, v in res.items():
        print(k, {a: round(b, 2) if isinstance(b, float) else b for a, b in v.items()} if isinstance(v, dict) else round(v, 3))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
