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
          follows={AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE}, priority=10),
    
    # Noun - Possession (Egalik)
    # Can follow Plural or Word Formation
    Affix("poss_1sg", "m", "1-shaxs birlik egalik", "1st person singular possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["m", "im"], follows={AffixCategory.PLURAL, AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE}),
    Affix("poss_2sg", "ng", "2-shaxs birlik egalik", "2nd person singular possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["ng", "ing"], follows={AffixCategory.PLURAL, AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE}),
    Affix("poss_3sg", "si", "3-shaxs birlik egalik", "3rd person singular possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["i", "si"], follows={AffixCategory.PLURAL, AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE}),
    Affix("poss_1pl", "miz", "1-shaxs ko'plik egalik", "1st person plural possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["miz", "imiz"], follows={AffixCategory.PLURAL, AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE}),
    Affix("poss_2pl", "ngiz", "2-shaxs ko'plik egalik", "2nd person plural possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["ngiz", "ingiz"], follows={AffixCategory.PLURAL, AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE}),
    Affix("poss_3pl", "lari", "3-shaxs ko'plik egalik", "3rd person plural possession", AffixType.INFLECTIONAL, AffixCategory.POSSESSION, ["lari"], follows={AffixCategory.WORD_FORMATION, AffixCategory.DIMINUTIVE}), # lari already includes plural meaning conceptually but grammatically attaches to stem
    
    # Noun - Cases (Kelishiklar)
    # Must be at the end of the noun phrase (after Plural, Possession)
    Affix("case_gen", "ning", "qaratqich", "genitive", AffixType.INFLECTIONAL, AffixCategory.CASE, ["ning"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.WORD_FORMATION}, priority=5),
    Affix("case_acc", "ni", "tushum", "accusative", AffixType.INFLECTIONAL, AffixCategory.CASE, ["ni"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.WORD_FORMATION}, priority=5),
    Affix("case_dat", "ga", "jo'nalish", "dative", AffixType.INFLECTIONAL, AffixCategory.CASE, ["ga", "ka", "qa"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.WORD_FORMATION}, priority=5),
    Affix("case_loc", "da", "o'rin-payt", "locative", AffixType.INFLECTIONAL, AffixCategory.CASE, ["da", "ta"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.WORD_FORMATION}, priority=5),
    Affix("case_abl", "dan", "chiqish", "ablative", AffixType.INFLECTIONAL, AffixCategory.CASE, ["dan", "tan"], follows={AffixCategory.PLURAL, AffixCategory.POSSESSION, AffixCategory.WORD_FORMATION}, priority=5),
    
    # Verb - Tense/Aspect
    Affix("tense_past", "di", "o'tgan zamon", "past tense", AffixType.INFLECTIONAL, AffixCategory.TENSE, ["di", "ti"]),
    Affix("tense_pres_yapti", "yapti", "hozirgi zamon", "present tense", AffixType.INFLECTIONAL, AffixCategory.TENSE, ["yapti", "yotir", "moqda", "yap"]),
    Affix("tense_fut_a", "a", "kelasi zamon", "future tense", AffixType.INFLECTIONAL, AffixCategory.TENSE, ["a", "y"]),
    Affix("tense_pres_ydi", "ydi", "hozirgi-kelasi zamon", "present-future tense", AffixType.INFLECTIONAL, AffixCategory.TENSE, ["ydi", "di"]),
    
    # Verb - Participles
    Affix("partic_gan", "gan", "sifatdosh (o'tgan)", "participle (past)", AffixType.INFLECTIONAL, AffixCategory.PARTICIPLE, ["gan", "kan", "qan"]),
    Affix("partic_digan", "digan", "sifatdosh (hozirgi/kelasi)", "participle (present/future)", AffixType.INFLECTIONAL, AffixCategory.PARTICIPLE, ["adigan", "ydigan", "digan"]),
    Affix("partic_ar", "ar", "sifatdosh", "participle", AffixType.INFLECTIONAL, AffixCategory.PARTICIPLE, ["ar", "r"]),
    Affix("partic_mas", "mas", "sifatdosh (inkor)", "participle (negative)", AffixType.INFLECTIONAL, AffixCategory.PARTICIPLE, ["mas"]),
    
    # Verb - Gerunds
    Affix("gerund_ib", "ib", "ravishdosh", "gerund", AffixType.INFLECTIONAL, AffixCategory.GERUND, ["ib", "b"]),
    Affix("gerund_gach", "gach", "ravishdosh", "gerund", AffixType.INFLECTIONAL, AffixCategory.GERUND, ["gach", "kach", "qach"]),
    Affix("gerund_guncha", "guncha", "ravishdosh", "gerund", AffixType.INFLECTIONAL, AffixCategory.GERUND, ["guncha", "kuncha", "quncha"]),
    
    # Verb - Negation (Before tense/mood)
    Affix("neg_ma", "ma", "inkor", "negation", AffixType.INFLECTIONAL, AffixCategory.NEGATION, ["ma", "mas", "may"]),
    
    # Person (Verbs & Predicative nouns)
    Affix("verb_pers_1sg", "man", "1-shaxs birlik", "1st person singular", AffixType.INFLECTIONAL, AffixCategory.PERSON, ["man"], follows={AffixCategory.TENSE, AffixCategory.PARTICIPLE, AffixCategory.WORD_FORMATION}),
    Affix("verb_pers_2sg", "san", "2-shaxs birlik", "2nd person singular", AffixType.INFLECTIONAL, AffixCategory.PERSON, ["san"], follows={AffixCategory.TENSE, AffixCategory.PARTICIPLE, AffixCategory.WORD_FORMATION}),
    Affix("verb_pers_1pl", "miz", "1-shaxs ko'plik", "1st person plural", AffixType.INFLECTIONAL, AffixCategory.PERSON, ["miz"], follows={AffixCategory.TENSE, AffixCategory.PARTICIPLE, AffixCategory.WORD_FORMATION}),
    Affix("verb_pers_2pl", "siz", "2-shaxs ko'plik", "2nd person plural", AffixType.INFLECTIONAL, AffixCategory.PERSON, ["siz", "sizlar"], follows={AffixCategory.TENSE, AffixCategory.PARTICIPLE, AffixCategory.WORD_FORMATION}),
    Affix("verb_pers_3pl", "dilar", "3-shaxs ko'plik", "3rd person plural", AffixType.INFLECTIONAL, AffixCategory.PERSON, ["lar", "dilar"], follows={AffixCategory.TENSE, AffixCategory.PARTICIPLE, AffixCategory.WORD_FORMATION}),
    
    # Word Formation
    Affix("deriv_chi", "chi", "shaxs oti yasovchi", "noun to noun", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["chi"]),
    Affix("deriv_lik", "lik", "mavhum ot yasovchi", "noun to noun", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["lik", "lig"]),
    Affix("deriv_li", "li", "sifat yasovchi", "noun to adj", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["li"]),
    Affix("deriv_siz", "siz", "sifat yasovchi", "noun to adj", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["siz"]),
    Affix("deriv_la", "la", "fe'l yasovchi", "noun/adj to verb", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["la", "lan", "lash", "lashtir"]),
    Affix("deriv_ish", "ish", "harakat nomi", "verb to noun", AffixType.DERIVATIONAL, AffixCategory.WORD_FORMATION, ["ish", "sh"]),
    
    # Particles
    Affix("part_ku", "ku", "yuklama", "particle", AffixType.PARTICLE, AffixCategory.PARTICLE_CAT, ["ku"]),
    Affix("part_chi", "chi", "yuklama", "particle", AffixType.PARTICLE, AffixCategory.QUESTION, ["chi"]),
    Affix("part_da", "da", "yuklama", "particle", AffixType.PARTICLE, AffixCategory.PARTICLE_CAT, ["da"]),
    Affix("part_mi", "mi", "so'roq yuklama", "question particle", AffixType.PARTICLE, AffixCategory.QUESTION, ["mi"]),
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
