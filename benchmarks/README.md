# Benchmarks

`python benchmarks/ud_eval.py` — UD (UD_Uzbek-UT, UD_Uzbek-UzUDT) qo'lda belgilangan lemmalar bo'yicha o'zak chegarasi,
tahlilchi top-1 / oracle@8, 61 so'zli gold F1 va `encode` ko'rsatkichlari (fertility, tezlik, decode aniqligi).

- **DEV** (UzUDT train): sozlamalarni tanlash uchun. **TEST** (UT test + UzUDT test): faqat yakuniy hisob.
- `root_oracle8 - root_top1` = ranker tuzatishi mumkin bo'lgan zaxira; `100 - root_oracle8` = lug'at/qoida yetishmasligi.
- `results/` ichida har muhim o'zgarishdan keyingi natijalar saqlanadi (PR'larda oldin/keyin solishtirish uchun).
- Cheklov: UD lemmasi ba'zan boshqa konventsiyada (masalan `o'yna` va `o'ynat`), shuning uchun 100% maqsad emas.

## Nomzod tanlagichni (ranker) qayta o'qitish
`python benchmarks/train_ranker.py --ud-dir DIR [--cv]` — `uztoken/data/ranker_weights.json` ni UD DEV (UzUDT train)
dan o'rgatadi (TEST ishlatilmaydi). Og'irliklar UD_Uzbek-UzUDT (CC BY-SA 4.0) ma'lumotlaridan hosil qilingan.
Belgilar ro'yxati `uztoken/ranker.py: FEATURE_NAMES` da; ro'yxat o'zgarsa og'irliklarni qayta o'qitish shart
(mos kelmasa ranker avtomatik o'chadi va eski ball tizimi ishlaydi).
