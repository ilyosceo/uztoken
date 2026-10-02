#!/usr/bin/env python3
"""
YonAI — Oddiy sun'iy intellekt chatboti (Python).

Imkoniyatlar:
  • Salomlashish va suhbat (kalit so'zlar asosida)
  • Matematik ifodalarni hisoblash ("2+2*3" kabi)
  • Vaqt/sana so'rovlari
  • So'zni teskari qilish, uzunligini aytish
  • Tasodifiy son / maslahat generatsiyasi
  • O'rganish rejimi: bilmasa, javobni yodlab oladi (memory.json)

Ishga tushirish:
    python3 ai.py
"""

import json
import os
import random
import re
from datetime import datetime

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")

MASLAHATLAR = [
    "Kichik qadamlar katta natijaga olib keladi.",
    "Har kuni 15 daqiqa o'qish miyani rivojlantiradi.",
    "Savol berishdan uyalmang — bilim shundan boshlanadi.",
    "Erta uxlash va erta turish sog'liq kaliti.",
    "Qiyin vazifani bo'laklarga bo'ling, osonlashadi.",
    "Suv ichishni unutmang — miya suvni yaxshi ko'radi!",
]


class YonAI:
    def __init__(self):
        self.name = "YonAI"
        self.knowledge = self._load_memory()

    # ---- Xotira (o'rganish) bilan ishlash -------------------------------
    def _load_memory(self) -> dict:
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_memory(self):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.knowledge, f, ensure_ascii=False, indent=2)

    def learn(self, trigger: str, answer: str):
        self.knowledge[trigger.lower().strip()] = answer.strip()
        self._save_memory()

    # ---- Asosiy javob generatori ----------------------------------------
    def respond(self, text: str) -> str:
        t = text.strip().lower()
        if not t:
            return "Hech narsa yozmadingiz. Biror savol bering!"

        # 1) O'rgatilgan javoblarni tekshirish
        for key, val in self.knowledge.items():
            if key and key in t:
                return val

        # 2) Matnni tozalab, matematik ifodani sinab ko'rish
        expr = re.sub(r"[xх×]", "*", t.replace(" ", "").replace(":", "/"))
        if re.fullmatch(r"[\d+\-*/().^%]+", expr or ""):
            try:
                safe_expr = expr.replace("^", "**")
                allowed = set("0123456789+-*/(). %*")
                if all(c in allowed for c in safe_expr):
                    result = eval(safe_expr, {"__builtins__": {}}, {})
                    return f"Javobi: {result}"
            except Exception:
                pass

        # 3) Maxsus buyruqlar (teskari qilish / uzunlik) — birinchi tekshiriladi
        if "teskari" in t or "reverse" in t:
            return self._reverse_from(text)
        if "uzunligi" in t or "nechta harf" in t or "necha ta belgi" in t:
            return self._length_from(text)

        # 4) Kalit so'zlar bo'yicha javoblar (tartib muhim: avtor maxsus so'zlar)
        rules = [
            (r"(isming(ni)?|sen kimsan|kimsan)",
             lambda: "Mening ismim YonAI — oddiy o'zbek tilidagi yordamchi bot."),
            (r"(soat necha|hozir vaqt|vaqt nechida)",
             lambda: f"Hozir vaqt: {datetime.now().strftime('%H:%M:%S')}"),
            (r"(sana|bugun qaysi kun|nechinchi kun)",
             lambda: f"Bugun: {datetime.now().strftime('%d.%m.%Y, %A')}"),
            (r"(tasodifiy son|random son|son o'ylab top)",
             lambda: f"Tasodifiy son: {random.randint(1, 100)}"),
            (r"(maslahat|tavsiya)",
             lambda: f"💡 {random.choice(MASLAHATLAR)}"),
            (r"(kulgu|kulgili|latifa|joke)",
             lambda: random.choice([
                 "Dasturchi nega qorong'uda ishlaydi? Chunki bug'lar yorug'likda yashirinadi!",
                 "— Kompyuter kasal bo'lsa nima qilamiz? — Restart 'deb davolaymiz!",
             ])),
            (r"(rahmat|tashakkur|thanks)",
             lambda: "Arzimaydi! Yana yordam kerak bo'lsa, bemalol so'rang."),
            (r"(yordam|help|nima qila olasiz)",
             lambda: ("Men quyidagilarni qila olaman:\n"
                      "• Matematika: \"2+2*3\" deb yozing\n"
                      "• Vaqt/sana: \"soat necha\"\n"
                      "• \"maslahat ber\", \"tasodifiy son\"\n"
                      "• \"<so'z> teskari qil\", \"<so'z> uzunligi\"\n"
                      "• Bilmaganimni o'rgating: \"o'rgit: savol | javob\"\n"
                      "• Chiqish: \"chiqish\"")),
            (r"\b(salom|assalomu[ ,]?alaykum|hello|hi|hey)\b",
             lambda: random.choice([
                 "Va alaykum assalom! Sizga qanday yordam bera olaman?",
                 "Salom! Men YonAIman. Biror narsa so'rang.",
             ])),
            (r"(hayr|khayr|bye|ko'rishguncha)",
             lambda: "Xayr! Ko'rishguncha 👋"),
        ]

        for pattern, handler in rules:
            if re.search(pattern, t):
                return handler()

        # 5) O'rgatish buyrug'i: "o'rgit: savol | javob"
        m = re.match(r"^o'?rgit\s*[:=]\s*(.+?)\s*\|\s*(.+)$", t)
        if m:
            answer = text.split("|", 1)[1].strip()
            self.learn(m.group(1), answer)
            return f"Yaxshi, '{m.group(1)}' haqida o'rgandim ✅"

        # 6) Noma'lum — o'rganishni taklif qilish
        return (f"Kechirasiz, '{text}' haqida hali bilmayman. "
                f"Agar xohlasangiz o'rgating: \"o'rgit: {text} | javob\"")

    # ---- Yordamchi funksiyalar ------------------------------------------
    @staticmethod
    def _extract_target(text: str) -> str:
        """'X so'zini teskari qil' / 'X uzunligi' dan X ni ajratib oladi."""
        # Qo'shtirnoqdagi so'zni ustima qilib olamiz
        quoted = re.search(r"['\"“”](.+?)['\"“”]", text)
        if quoted:
            return quoted.group(1)
        # "teskari"/"uzunligi" dan oldingi oxirgi so'zni olamiz
        m = re.search(r"(.+?)\s*(?:so.?zini|so'zining)?\s*(?:ni)?\s*"
                      r"(?:teskari|reverse|uzunligi|nechta)", text, re.IGNORECASE)
        if m:
            words = m.group(1).replace("'", "\u2018").split()
            for w in reversed(words):
                w = w.strip(".,!?;:")
                if w.lower() not in ("ayting", "ayt", "qil", "yozib", "bilish", "hisob"):
                    return w.replace("\u2018", "'")
        return ""

    @classmethod
    def _reverse_from(cls, text: str) -> str:
        word = cls._extract_target(text)
        return f"'{word}' → {word[::-1]}" if word else "Qaysi so'zni teskari qilay?"

    @classmethod
    def _length_from(cls, text: str) -> str:
        word = cls._extract_target(text)
        return f"'{word}' so'zi {len(word)} ta harfdan iborat." if word else "Qaysi so'zni sanay?"


def main():
    ai = YonAI()
    print("=" * 50)
    print(f"  🤖 {ai.name} — oddiy AI yordamchi")
    print("  Boshlash uchun biror narsa yozing. 'chiqish' = exit")
    print("=" * 50)
    while True:
        try:
            user = input("\nSiz: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nXayr! 👋")
            break
        if not user:
            continue
        if user.lower() in ("chiqish", "exit", "quit"):
            print(f"{ai.name}: Xayr! Ko'rishguncha 👋")
            break
        print(f"{ai.name}: {ai.respond(user)}")


if __name__ == "__main__":
    main()
