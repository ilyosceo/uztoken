from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any
from enum import Enum

class AffixType(Enum):
    INFLECTIONAL = 'inflectional'
    DERIVATIONAL = 'derivational'
    PARTICLE = 'particle'

class AffixCategory(Enum):
    PLURAL = 'plural'           # Ko'plik
    POSSESSION = 'possession'    # Egalik/Tortishish
    CASE = 'case'               # Kelishik
    TENSE = 'tense'             # Zamon
    PERSON = 'person'           # Shaxs-son
    NEGATION = 'negation'       # Inkor
    VOICE = 'voice'             # Nisbat
    MOOD = 'mood'               # Mayl
    PARTICIPLE = 'participle'   # Sifatdosh
    GERUND = 'gerund'           # Ravishdosh
    WORD_FORMATION = 'word_formation'  # So'z yasash
    DIMINUTIVE = 'diminutive'   # Kichraytirish
    COMPARISON = 'comparison'   # Qiyoslash
    PARTICLE_CAT = 'particle_cat'  # Yuklama
    QUESTION = 'question'       # So'roq

@dataclass
class Affix:
    id: str
    base: str
    meaning_uz: str
    meaning_en: str
    type: AffixType
    category: AffixCategory
    variants: List[str]
    # To restrict which categories this suffix can attach to (suffix ordering):
    # e.g., Case can follow Plural, Possession, Word Formation, but NOT Person.
    follows: Optional[Set[AffixCategory]] = None 
    priority: int = 0
    # Rule for allomorphs (e.g. qa only after k,q)
    # If None, no restriction.
    phonetic_rules: Optional[Dict[str, str]] = None 

_AFFIXES_DB: List[Affix] = [
    # Noun - Plural
    Affix("pl_lar", "lar", "ko'plik", "plural", AffixType.INFLECTIONAL, AffixCategory.PLURAL, ["lar"], 
          follows={AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE, AffixCategory.POSSESSION, AffixCategory.CASE, AffixCategory.PARTICIPLE}, priority=10),
    
    # Noun - Possession (Egalik)
    # Can follow Plural or Word Formation
    Affix("poss_1sg", "m", "1-shaxs birlik egalik", "1st person singular possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["m", "im"], follows={AffixCategory.PLURAL, AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE, AffixCategory.PARTICIPLE, AffixCategory.MOOD}),
    Affix("poss_2sg", "ng", "2-shaxs birlik egalik", "2nd person singular possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["ng", "ing"], follows={AffixCategory.PLURAL, AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE, AffixCategory.PARTICIPLE, AffixCategory.MOOD}),
    Affix("poss_3sg", "si", "3-shaxs birlik egalik", "3rd person singular possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["i", "si"], follows={AffixCategory.PLURAL, AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE, AffixCategory.PARTICIPLE, AffixCategory.MOOD}),
    Affix("poss_1pl", "miz", "1-shaxs ko'plik egalik", "1st person plural possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["miz", "imiz"], follows={AffixCategory.PLURAL, AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE, AffixCategory.PARTICIPLE, AffixCategory.MOOD}),
    Affix("poss_2pl", "ngiz", "2-shaxs ko'plik egalik", "2nd person plural possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["ngiz", "ingiz"], follows={AffixCategory.PLURAL, AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE, AffixCategory.PARTICIPLE, AffixCategory.MOOD}),
    Affix("poss_3pl", "lari", "3-shaxs ko'plik egalik", "3rd person plural possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["lari"], follows={AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE}), # lari already includes plural meaning conceptually but grammatically attaches to stem
    # Egalik olmoshi (-niki): maktabnikida, telefonlarimizniki va h.k.
    Affix("poss_niki", "niki", "egalik olmoshi (-niki)", "possessive pronoun -niki", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["niki"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.WORD_FORMATION}, priority=6),
    
    # Noun - Cases (Kelishiklar)
    # Must be at the end of the noun phrase (after Plural, Possession)
    Affix("case_gen", "ning", "qaratqich", "genitive", AffixType.INFLECTIONAL, AffixCategory.CASE, ["ning"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.WORD_FORMATION, AffixCategory.PARTICIPLE, AffixCategory.MOOD}, priority=5),
    Affix("case_acc", "ni", "tushum", "accusative", AffixType.INFLECTIONAL, AffixCategory.CASE, ["ni"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.WORD_FORMATION, AffixCategory.PARTICIPLE, AffixCategory.MOOD}, priority=5),
    Affix("case_dat", "ga", "jo'nalish", "dative", AffixType.INFLECTIONAL, AffixCategory.CASE, ["ga", "ka", "qa"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.WORD_FORMATION, AffixCategory.PARTICIPLE, AffixCategory.MOOD}, priority=5),
    Affix("case_loc", "da", "o'rin-payt", "locative", AffixType.INFLECTIONAL, AffixCategory.CASE, ["da", "ta"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.WORD_FORMATION, AffixCategory.PARTICIPLE, AffixCategory.MOOD}, priority=5),
    Affix("case_abl", "dan", "chiqish", "ablative", AffixType.INFLECTIONAL, AffixCategory.CASE, ["dan", "tan"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.WORD_FORMATION, AffixCategory.PARTICIPLE, AffixCategory.MOOD}, priority=5),
    
    # Verb - Infinitive / mood / evidential bases (must be checked BEFORE Person/Tense)
    Affix("inf_moq", "moq", "ortiqcha fe'l (infinitiv)", "infinitive", AffixType.INFLECTIONAL, AffixCategory.MOOD, ["moq", "mok"], priority=12),
    Affix("cond_edi", "edi", "shart/kechiktirilgan", "past habitual/conditional", AffixType.INFLECTIONAL, AffixCategory.MOOD, ["edi", "edik", "edilar"], priority=12),
    Affix("evd_emish", "emish", "so'zlanuvchi (guvohlik)", "evidential", AffixType.INFLECTIONAL, AffixCategory.MOOD, ["emish", "emishlar"], priority=12),

    # Verb - Tense/Aspect
    Affix("tense_past", "di", "o'tgan zamon", "past tense", AffixType.INFLECTIONAL, AffixCategory.TENSE, ["di", "ti"], priority=9),
    Affix("tense_pres_yapti", "yapti", "hozirgi zamon", "present tense", AffixType.INFLECTIONAL, AffixCategory.TENSE, ["yapti", "yotir", "moqda", "yap"]),
    Affix("tense_fut_a", "a", "kelasi zamon", "future tense", AffixType.INFLECTIONAL, AffixCategory.TENSE, ["a", "y"]),
    Affix("tense_pres_ydi", "ydi", "hozirgi-kelasi zamon", "present-future tense", AffixType.INFLECTIONAL, AffixCategory.TENSE, ["ydi", "adi", "edi", "di"], priority=9),
    
    # Verb - Participles
    Affix("partic_gan", "gan", "sifatdosh (o'tgan)", "participle (past)", AffixType.INFLECTIONAL, AffixCategory.PARTICIPLE, ["gan", "kan", "qan"], priority=8),
    # Participial genitive: kelgan+ingiz, yozmagan+lari kabi egalik qatlamiga yo'l ochadi
    Affix("partic_gen", "ning", "sifatdosh qaratqich", "participial genitive", AffixType.INFLECTIONAL, AffixCategory.PARTICIPLE, ["ning"], priority=8),
    Affix("partic_digan", "digan", "sifatdosh (hozirgi/kelasi)", "participle (present/future)", AffixType.INFLECTIONAL, AffixCategory.PARTICIPLE, ["adigan", "ydigan", "digan"]),
    Affix("partic_ar", "ar", "sifatdosh", "participle", AffixType.INFLECTIONAL, AffixCategory.PARTICIPLE, ["ar"]),
    Affix("partic_mas", "mas", "sifatdosh (inkor)", "participle (negative)", AffixType.INFLECTIONAL, AffixCategory.PARTICIPLE, ["mas"]),
    
    # Verb - Gerunds
    Affix("gerund_lab", "lab", "ravishdosh (-lab/-ib/-lib/-yb)", "converb", AffixType.INFLECTIONAL, AffixCategory.GERUND, ["lab", "lib", "ib", "yb"], priority=6),
    Affix("gerund_qoyib", "qoyib", "ravishdosh (sura yo'li)", "converb (honorific)", AffixType.INFLECTIONAL, AffixCategory.GERUND, ["qo'yib", "qoyib"], priority=4),
    Affix("gerund_ib", "ib", "ravishdosh", "gerund", AffixType.INFLECTIONAL, AffixCategory.GERUND, ["ib"]),
    Affix("gerund_gach", "gach", "ravishdosh", "gerund", AffixType.INFLECTIONAL, AffixCategory.GERUND, ["gach", "kach", "qach"]),
    Affix("gerund_guncha", "guncha", "ravishdosh", "gerund", AffixType.INFLECTIONAL, AffixCategory.GERUND, ["guncha", "kuncha", "quncha"]),
    
    # Verb - Negation (Before tense/mood)
    Affix("neg_ma", "ma", "inkor", "negation", AffixType.INFLECTIONAL, AffixCategory.NEGATION, ["ma", "mas", "may", "magan", "maydi", "masin", "maysiz", "maymiz"]),
    
    # Person (Verbs & Predicative nouns)
    Affix("verb_pers_1sg", "man", "1-shaxs birlik", "1st person singular", AffixType.INFLECTIONAL, AffixCategory.PERSON, ["man", "im", "ym"], follows={AffixCategory.TENSE, AffixCategory.PARTICIPLE, AffixCategory.WORD_FORMATION}),
    Affix("verb_pers_2sg", "san", "2-shaxs birlik", "2nd person singular", AffixType.INFLECTIONAL, AffixCategory.PERSON, ["san", "sing", "ng", "ing"], follows={AffixCategory.TENSE, AffixCategory.PARTICIPLE, AffixCategory.WORD_FORMATION}),
    Affix("verb_pers_1pl", "miz", "1-shaxs ko'plik", "1st person plural", AffixType.INFLECTIONAL, AffixCategory.PERSON, ["miz", "imiz", "mizlar"], follows={AffixCategory.TENSE, AffixCategory.PARTICIPLE, AffixCategory.WORD_FORMATION}),
    Affix("verb_pers_2pl", "siz", "2-shaxs ko'plik", "2nd person plural", AffixType.INFLECTIONAL, AffixCategory.PERSON, ["siz", "sizlar", "ingiz", "qing", "larsiz"], follows={AffixCategory.TENSE, AffixCategory.PARTICIPLE, AffixCategory.WORD_FORMATION}),
    Affix("verb_pers_3pl", "dilar", "3-shaxs ko'plik", "3rd person plural", AffixType.INFLECTIONAL, AffixCategory.PERSON, ["lar", "dilar", "adi"], follows={AffixCategory.TENSE, AffixCategory.PARTICIPLE, AffixCategory.WORD_FORMATION}),
    
    # Word Formation
    Affix("deriv_chi", "chi", "shaxs oti yasovchi", "noun to noun", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["chi"], ),
    Affix("deriv_lik", "lik", "mavhum ot yasovchi", "noun to noun", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["lik", "lig"], ),
    Affix("deriv_li", "li", "sifat yasovchi", "noun to adj", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["li"], ),
    Affix("deriv_siz", "siz", "sifat yasovchi", "noun to adj", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["siz"], ),
    Affix("deriv_la", "la", "fe'l yasovchi", "noun/adj to verb", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["la", "lan", "lash", "lashtir"], ),
    Affix("deriv_ish", "ish", "harakat nomi", "verb to noun", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["ish", "sh"], ),
    
    # Pronominal / substantivizing suffixes (-dagi, -giki, -gi)
    Affix("deriv_dagi", "dagi", "olmosh qo'shimchasi (-dagi/-giki)", "pronominal -dagi", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["dagi", "giki", "gi", "daghi"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.CASE, AffixCategory.PARTICIPLE}, priority=7),
    Affix("deriv_gina", "gina", "ta'kidlash qo'shimchasi", "emphatic -gina", AffixType.PARTICLE, AffixCategory.PARTICLE_CAT, ["gina", "goi"], priority=3),

    # Verb voice / aspectual compound chunks (yordamchi fe'llar qatlam sifatida)
    Affix("voice_dir", "dir", "turtirish nisbati (-dir/-tir/-giz)", "causative", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["dir", "tir", "giz", "qiz", "adir", "ydir", "tdir"], priority=8),
    Affix("voice_il", "il", "o'zlik/o'tish nisbati (-il/-in)", "reflexive/passive", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["il", "in"], priority=7),
    Affix("voice_ish", "ish", "o'zaro nisbati", "reciprocal", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["ish"], priority=6),
    Affix("aux_kelib", "kelib", "yo'l fe'l shakli (-ib kelib)", "converb aux", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["kelib", "ketib", "qo'yib", "qoyib"], priority=6),
    Affix("aux_ola", "ola", "mumkinlik yordamchisi (-ol-)", "potential aux", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["oladigan", "olmaydigan", "olgan", "olmagan", "ola", "oli", "oloq"], priority=7),
    Affix("aux_kora", "ko'ra", "takrorlanish yordamchisi (-ib ko'ra)", "repetitive aux", AffixType.DERIVATIONAL, AffixCategory.VOICE, ["ko'ra", "koray", "bera", "beri", "qila", "qilay", "tura", "yura"], priority=6),

    # Davomlik zamon qatlamari: -ayotgan / -ayotganingiz (o'qitayotganingizdan)
    Affix("cont_yotgan", "yotgan", "davomlik sifatdoshi (-ayotgan/-oyotgan)", "progressive participle", AffixType.DERIVATIONAL, AffixCategory.PARTICIPLE, ["yotgan", "ayotgan", "oyotgan", "otgan"], priority=9),
    Affix("cont_yotib", "yotib", "ravishdosh (-ayotib/-ab borib)", "converb borib", AffixType.DERIVATIONAL, AffixCategory.GERUND, ["yotib", "ayotib", "borib", "yuborib"], priority=5),

    # Particles
    Affix("part_ku", "ku", "yuklama", "particle", AffixType.PARTICLE, AffixCategory.PARTICLE_CAT, ["ku"]),
    Affix("part_chi", "chi", "yuklama", "particle", AffixType.PARTICLE, AffixCategory.QUESTION, ["chi"]),
    Affix("part_da", "da", "yuklama", "particle", AffixType.PARTICLE, AffixCategory.PARTICLE_CAT, ["da"]),
    Affix("part_mi", "mi", "so'roq yuklama", "question particle", AffixType.PARTICLE, AffixCategory.QUESTION, ["mi", "mimu", "maymi"], priority=9),
]

def get_all_affixes() -> List[Affix]:
    return _AFFIXES_DB

def get_affix_by_id(affix_id: str) -> Optional[Affix]:
    for affix in _AFFIXES_DB:
        if affix.id == affix_id:
            return affix
    return None

def get_suffix_variants() -> List[Dict[str, Any]]:
    variants_list = []
    for affix in _AFFIXES_DB:
        for variant in affix.variants:
            variants_list.append({
                "variant": variant,
                "affix": affix
            })
    variants_list.sort(key=lambda x: (len(x["variant"]), x["affix"].priority), reverse=True)
    return variants_list
