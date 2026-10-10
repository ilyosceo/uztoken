# Benchmarks

`python benchmarks/ud_eval.py` — UD (UD_Uzbek-UT, UD_Uzbek-UzUDT) qo'lda belgilangan lemmalar bo'yicha o'zak chegarasi,
tahlilchi top-1 / oracle@8, 61 so'zli gold F1 va `encode` ko'rsatkichlari (fertility, tezlik, decode aniqligi).

- **DEV** (UzUDT train): sozlamalarni tanlash uchun. **TEST** (UT test + UzUDT test): faqat yakuniy hisob.
- `root_oracle8 - root_top1` = ranker tuzatishi mumkin bo'lgan zaxira; `100 - root_oracle8` = lug'at/qoida yetishmasligi.
- `results/` ichida har muhim o'zgarishdan keyingi natijalar saqlanadi (PR'larda oldin/keyin solishtirish uchun).
- Cheklov: UD lemmasi ba'zan boshqa konventsiyada (masalan `o'yna` va `o'ynat`), shuning uchun 100% maqsad emas.
