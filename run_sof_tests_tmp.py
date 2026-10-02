# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from uztoken import create_tokenizer

tok = create_tokenizer(load_dict=True)

def aff_str(a):
    if isinstance(a, dict):
        return f"{a.get('text')}({a.get('id')})"
    return f"{getattr(a,'text',str(a))}({getattr(a,'affix_id','')})"

GROUPS = {
 "G1": ["maktablarimizdagilardan","ko'chalaridagi","uylarnikida","qishloqlarimizdagilarga",
        "o'qituvchilarimizning","bolalarimiznikidan","kitoblarimizdagi","shaharlarimizdagilarning",
        "dasturchilarimizga","o'quvchilarimizdan","mashinalarimizniki","kompyuterlarimizdan",
        "telefonlarimizniki","hovlilarimizdagi","mevalarimizdan","daraxtlarimizdan",
        "gulzorlarimizga","talabalarimiznikida","ustozlarimizning","do'konlarimizdagi",
        "kutubxonalarimizdan"],
 "G2": ["qishlog'imizdan","bog'larimizda","tog'imizga","yong'og'imdan","qog'ozdan","tog'laridagi",
        "qo'ng'iroqni","sovg'alar","burchag'idan","dargoh","tuproqdan","yong'oqdan","yugurib",
        "yig'lagan","ko'pikdan"],
 "G3": ["o'zbekistonlik","oʻzbekistonlik","g'alaba","gʻalaba","a'lo","ma'no","mas'ul","san'a",
        "e'tibor","e'tiroz","e'tiqod","qis'm","g'urur","gʻurur","o'rgatmoq","oʻrgatmoq",
        "qo'shmoq","qoʻshmoq"],
 "G4": ["kelganingizdan","bormaganingizdagidek","yozdirib","ko'rmayapsizmi","o'qitayotganingizdan",
        "ko'rsatolmaganingizdan","tushuntirmoqchimisiz","eshitmaganingizday","ko'rmaganingizdek",
        "yozmaganingizdan","bilmaganingizni","kutayotganingizdan","o'ylab","ko'rmadingizmi",
        "qaytmaganingizdan","topmaganingizni","berolmaganingizdan"],
 "G5": ["Python'da","JavaScript'chi","serverda","GitHub'dan","Windows'ning","Linux'dagi",
        "ChatGPT'ga","YouTube'da","Instagram'chi","Telegram'dan"],
 "G6": ["2024-yilda","5-kurs","3-maktab","100-sonli","2,5","1-yanvar","1991-yil","23:45","$100","15%"],
 "G7": ["yoz","osh","bosh","qora","tush"],
 "G8": ["mas'uliyatsizlikdan","tushunmovchilikka","foydalanilmaganligidan","qoniqarsizlikdan",
        "e'tiborsizlikning","o'zgartirib","bo'lmasligidan","ko'rib","chiqilmaganligidan",
        "foydalanib","bo'lmaydigan","qo'llanilmaganligining","aniqlab","bo'lmaydiganlikdan"],
}

for g, words in GROUPS.items():
    print(f"\n===== {g} =====")
    for w in words:
        res = tok.tokenize(w)
        morph = [t for t in res if t.is_morph and t.root]
        if morph:
            m = morph[0]
            chain = str(m.root) + "".join("+"+aff_str(x) for x in (m.suffixes or []))
            extra = ""
            if len(res) > 1:
                extra = " [+ " + ", ".join(str(t) for t in res[1:]) + "]"
            conf = getattr(m, 'confidence', None)
            c = f" ({conf:.2f})" if isinstance(conf,(int,float)) else ""
            print(f"OK   {w!r:35s} -> {chain}{c}{extra}")
        else:
            print(f"FAIL {w!r:35s} -> {str(res[0]) if res else 'BO SH'}")
