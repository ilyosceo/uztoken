"""Nomzod tanlagich (uztoken/ranker.py) og'irliklarini UD DEV (UzUDT train) dan o'rgatadi.

    python benchmarks/train_ranker.py --ud-dir DIR [--cv] [--lam 0.003]

Model: har so'z uchun nomzodlar ustida softmax (conditional logit), maqsad — to'g'ri o'zakli
(root == lemma) nomzodlar ehtimolini oshirish; L2 regulyarizatsiya, Adam. numpy kerak (faqat o'qitishda).
TEST bo'linmasi bu yerda ISHLATILMAYDI (faqat DEV). Natija: uztoken/data/ranker_weights.json
(UD ma'lumotlari CC BY-SA 4.0; og'irliklar shundan hosil qilingan).
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ud_eval import read_conllu, build_pairs, fetch_ud
from uztoken import create_tokenizer
from uztoken.normalizer import normalize_uzbek
from uztoken.ranker import candidate_features, FEATURE_NAMES, DEFAULT_WEIGHTS

fix = lambda r: r.replace("ʻ", "'").replace("ʼ", "'")


def build_groups(tok, pairs):
    a, stems = tok._analyzer, tok._dictionary.stems
    groups = []
    for w, lem in pairs.items():
        wl = normalize_uzbek(w)
        cs = a.analyze_all(wl, max_results=8)
        for c in cs:
            a._penalize_short_root(c)
        cs.sort(key=lambda c: c.confidence, reverse=True)
        if len(cs) < 2:
            continue
        X = np.array([candidate_features(wl, c, i, len(cs), stems) for i, c in enumerate(cs)])
        y = np.array([float(fix(c.root) == lem) for c in cs])
        if y.sum() > 0:
            groups.append((X, y))
    return groups


def fit(groups, lam, iters=400):
    allX = np.concatenate([g[0] for g in groups])
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    w = np.zeros(allX.shape[1]); m = np.zeros_like(w); v = np.zeros_like(w)
    for t in range(1, iters + 1):
        grad = lam * w
        for X, y in groups:
            Z = (X - mu) / sd; s = Z @ w; s -= s.max(); p = np.exp(s); p /= p.sum()
            q = p * y; q /= q.sum()
            grad -= Z.T @ (q - p) / len(groups)
        m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2
        w -= 0.05 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w, mu, sd


def acc(groups, model):
    w, mu, sd = model
    return 100 * sum(g[1][int(np.argmax(((g[0] - mu) / sd) @ w))] for g in groups) / len(groups)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ud-dir"); ap.add_argument("--cv", action="store_true")
    ap.add_argument("--lam", type=float, default=0.003); ap.add_argument("--out", default=str(DEFAULT_WEIGHTS))
    a = ap.parse_args()
    root = Path(a.ud_dir) if a.ud_dir else fetch_ud(Path.home() / ".cache" / "uztoken_ud")
    _, dev = read_conllu(root / "UD_Uzbek-UzUDT" / "uz_uzudt-ud-train.conllu")
    tok = create_tokenizer(load_dict=True); tok._analyzer._ranker = None  # nomzodlar eski tartibda
    infl, roots = build_pairs(dev)
    # qo'shimchasiz so'zlar ham o'qitishga kiradi: to'g'ri javob — butun so'z (bo'linmaslik)
    groups = build_groups(tok, infl) + build_groups(tok, roots); print("train groups:", len(groups))
    lam = a.lam
    if a.cv:
        random.seed(0); idx = list(range(len(groups))); random.shuffle(idx); folds = [set(idx[i::5]) for i in range(5)]
        best = None
        for l in [0.003, 0.03, 0.1, 0.3]:
            accs = [acc([groups[i] for i in folds[k]], fit([groups[i] for i in idx if i not in folds[k]], l, 250)) for k in range(5)]
            print(f"lam={l}: 5-fold CV top1={np.mean(accs):.1f}", flush=True)
            if best is None or np.mean(accs) > best[0]: best = (np.mean(accs), l)
        lam = best[1]
    w, mu, sd = fit(groups, lam)
    Path(a.out).write_text(json.dumps({"features": FEATURE_NAMES, "lam": lam, "w": w.tolist(), "mu": mu.tolist(),
                                        "sd": sd.tolist(), "trained_on": "UD_Uzbek-UzUDT train (types)"}, indent=1), encoding="utf8")
    print("saved", a.out, "| lam", lam, "| train-acc %.1f" % acc(groups, (w, mu, sd)))


if __name__ == "__main__":
    main()
