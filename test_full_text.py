# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from uztoken import create_tokenizer

TEXT = ("Kvantdagi neyro-chirog'imizdagi yong'og'zorlarimizdan sub'ektiv ob'ektlarimizning "
        "ma'lumotlar bazasidagi 2045-yilgi arxivlanmagan fayl'larni o'qiyotganimizda, "
        "ko'rsatib berolmaganingizdan kelib chiqib, dasturchilarimizning yozayotgan skriptlarini "
        "qayta ko'rib chiqmaganligingizni tushunmovchilikka yo'ydik. 3D-printerda bosilgan "
        "tuproqdan yasalgan ko'piksimon qopqoqchalarning 99,9%i yaroqsiz bo'lib chiqqanligi sababli, "
        "tasodifan yutib yuborgan mikrochiplarini qaytarganingizdami yoki shunchaki "
        "o'zgartirib bo'lmasligini bilganimizdami, aniq ayta olmaysiz; mas'uliyatsizlikning bu darajasi "
        "e'tiqodimizga zid!")

tok = create_tokenizer(load_dict=True)

def aff_str(a):
    if isinstance(a, dict):
        return f"{a.get('text')}({a.get('id')})"
    return f"{getattr(a,'text',str(a))}({getattr(a,'affix_id','')})"

words = TEXT.split()
n_ok = n_fail = 0
for w in words:
    res = tok.tokenize(w)
    morph = [t for t in res if t.is_morph and t.root]
    if morph:
        n_ok += 1
        m = morph[0]
        affs = m.suffixes or []
        chain = str(m.root) + "".join("+"+aff_str(x) for x in affs)
        note = ""
        if len(res) > 1:
            others = [str(t) for t in res[1:]]
            note = " [+ " + ", ".join(others) + "]"
        print(f"✅ {w!r:32s} -> {chain}{note}")
    else:
        n_fail += 1
        print(f"❌ {w!r:32s} -> {str(res[0]) if res else 'BO SH'}")

print("="*100)
print(f"Jami: {len(words)} | Morf tahlil topildi: {n_ok} | Topilmadi: {n_fail}")
