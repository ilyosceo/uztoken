"""
Morphological analyzer for Uzbek language.

Uses a rule-based approach with suffix stripping and dictionary lookup
to decompose Uzbek words into root + suffix chain.
"""

import re
from typing import List, Dict, Optional, Tuple, Any, Set
from functools import lru_cache
from dataclasses import dataclass, field

try:
    from .normalizer import normalize_uzbek
    from .affixes import get_suffix_variants, get_all_affixes, AffixCategory, Affix
    from .dictionary import Dictionary
    from .trie import SuffixTrie, DictionaryTrie
    from .ranker import CandidateRanker
except ImportError:
    pass

# Well-known Uzbek vowel-drop (syncope) stems and alternations pre-generated mapping
_MORPHOPHONOLOGICAL_MAP: Dict[str, str] = {
    # Vowel drops
    "shahr": "shahar",
    "singl": "singil",
    "burn": "burun",
    "o'g'l": "o'g'il",
    "bag'r": "bag'ir",
    "ko'ngl": "ko'ngil",
    "qorn": "qorin",
    "bo'yn": "bo'yin",
    "og'z": "og'iz",
    "qovrg'": "qovurg'a",
    # Consonant alternations (k->g, q->g')
    "yurag": "yurak",
    "qishlog'": "qishloq",
    "barmog'": "barmoq",
    "qulog'": "quloq",
    "tarmog'": "tarmoq",
    "yoprog'": "yaproq",
    "tayog'": "tayoq",
    "o'rtog'": "o'rtoq",
    "tirnog'": "tirnoq",
    # k -> gʻ alternation before 3rd-person possessive -i: burchak+i -> burchag'i
    "burchag'": "burchak",
    "chaqmag'": "chaqmog'",
    "ko'zlag'": "ko'zak",
    "yonog'": "yonoq",
    "tizmag'": "tizmoq",
    # q -> gʻ alternation before vowel-initial suffixes (oʻzbek yozuvi):
    # chiroq+imiz -> chirogʻimiz, yoqim+... -> yogʻ..., siroq -> sirogʻ...
    "chirog'": "chiroq",
    "sirog'": "siroq",
    "bog'": "boq",
    "yog'": "yoq",
    "tig'": "tiq",
    "sug'": "suq",
    "jug'": "juq",
    "qug'": "quq",
    "bug'": "buq",
    "jig'": "jiq",
    "seg'": "seq",
    "teg'": "teq",
    "leg'": "leq",
    "meg'": "meq",
    "nog'": "noq",
    "rog'": "roq",
    "zog'": "zoq",
    "g'irg'": "g'irq",
    "tirg'": "tirq",
    "sharg'": "sharq",
    "zarang'": "zaraq",
    "chang'": "chanq",
    "bang'": "banq",
    "darg'": "darq",
    "farg'": "farq",
    "garg'": "garq",
    "hang'": "hanq",
    "jang'": "janq",
    "karg'": "karq",
    "lang'": "lanq",
    "mang'": "manq",
    "nang'": "nanq",
    "pang'": "panq",
    "rang'": "ranq",
    "sang'": "sanq",
    "tang'": "tanq",
    "vang'": "vanq",
    "yarg'": "yarq",
}

# Fe'l o'zaklarining infinitiv (-moq) shakli — lug'atda fe'llar ko'pincha
# "-moq" bilan saqlangan (o'qimoq, bilmoq, kelmoq). Ayrim qo'shimchalardan
# oldin infinitiv qisqaradi: o'qimoq -> o'qi-, bilmoq -> bil-, yugurmoq -> yugur-
_REGISTRY_DICTIONARY = None  # MorphAnalyzer yaratilganda shu lug'at ro'yxatdan o'tadi


_INFINITIVE_VERBS: Optional[Set[str]] = None


def _get_infinitive_verbs() -> Set[str]:
    """Infinitiv (-moq) shaklidagi o'zaklar. Natija lazimi va bir marta hisoblanadi;
    lug'atni alohida yuklamaydi — chaqiruvchi tomondan berilgan bo'lsa ishlatadi."""
    global _INFINITIVE_VERBS
    if _INFINITIVE_VERBS is None:
        if _REGISTRY_DICTIONARY is not None:
            stems = set(_REGISTRY_DICTIONARY.stems)
        else:
            try:
                d = Dictionary()  # bundled data bilan auto-load
                stems = set(d.stems)
            except Exception:
                stems = set()
        _INFINITIVE_VERBS = {s for s in stems if s.endswith("moq") and len(s) >= 5}
    return _INFINITIVE_VERBS


_INFINITIVE_BASES: Optional[Set[str]] = None


_VERB_BASE_CACHE: Dict[str, str] = {}


def _build_verb_base_set(inf: str) -> Set[str]:
    """Bir infinitivdan qisqargan o'zak shakllari: bilmoq->{bil}, o'qimoq->{o'qi,o'q}."""
    out: Set[str] = set()
    base = inf[:-3]
    if len(base) < 2:
        return out
    out.add(base)
    stripped = base.rstrip("i")
    if len(stripped) >= 2:
        out.add(stripped)
    # konsonant bilan tugaydigan asos uchun -i shakli ham (kel -> keli emas, lekin qol->qoli yo'q;
    # faqat 'i' bilan tugovchilar uchun i'siz variant)
    if base.endswith("i"):
        out.add(base[:-1])
    return out


def _get_infinitive_bases() -> Set[str]:
    """To'liq baza — faqat kerak bo'lsa bir marta quriladi."""
    global _INFINITIVE_BASES
    if _INFINITIVE_BASES is None:
        bases: Set[str] = set()
        for inf in _get_infinitive_verbs():
            bases |= _build_verb_base_set(inf)
        _INFINITIVE_BASES = _extend_bases(bases)
    return _INFINITIVE_BASES


def _find_verb_base(candidate: str) -> Optional[str]:
    """Tez yo'l: candidate'ni oxirgi unli va undosh chegarasidan infinitiv deb
    taxmin qilib, lug'atda bor-yo'qligini tekshiramiz. To'liq bazani qurmaymiz."""
    if len(candidate) < 2 or len(candidate) > 12:
        return None
    hit = _VERB_BASE_CACHE.get(candidate)
    if hit is not None:
        # Rad etilgan nomzod keshda "" bo'lib turadi: uni None deb qaytarish shart, aks holda
        # takroriy so'zlarga bo'sh o'zakli soxta tahlil ('' , is_found=True) qo'shilardi.
        return hit or None
    if _REGISTRY_DICTIONARY is None:
        # ehtiyot chorasi: eski sekin yo'l
        ok = candidate in _get_infinitive_bases()
        return candidate if ok else None
    dict_stems = _REGISTRY_DICTIONARY.stems
    found = None
    # Faqat candidate'ning o'zi infinitiv asosi bo'lsa qabul qilamiz: bil <- bilmoq,
    # oʻqi <- oʻqimoq, oʻq <- oʻqimoq. Ilgari candidate'ning ISTALGAN boshlang'ich
    # qismi + "moq" lug'atda bo'lsa (yurakni -> yu+moq, berasan -> ber+moq) butun
    # so'z "fe'l asosi" deb topilib, to'g'ri bo'linishni yengib ketardi.
    if (candidate + "moq") in dict_stems or (candidate + "imoq") in dict_stems:
        found = candidate
    if found is None and candidate in _EXTRA_VERB_BASES:
        found = candidate
    if found is not None:
        _VERB_BASE_CACHE[candidate] = found
    elif len(_VERB_BASE_CACHE) < 200000:
        _VERB_BASE_CACHE[candidate] = ""
    return found or None

# Tutuq belgisi bilan yoziladigan tana so'zlar (hunspell bazasida yo'q bo'lsa qo'shiladi)
# Infinitiv bazasida yo'q, lekin tez-tez uchraydigan fe'l shakllari
_EXTRA_VERB_BASES = {
    "yoy", "yoz", "sez", "tiz", "yig", "bo", "quv", "yugur", "yuq",
    "yoqd", "kozd", "kuzd", "toyd", "jozd",
}


def _extend_bases(bases: Set[str]) -> Set[str]:
    bases |= _EXTRA_VERB_BASES
    return bases


_TUTUQ_WORDS = {
    "san'a", "san'at", "san'atchi", "mas'ul", "mas'uliyat", "ma'no", "ma'noli",
    "a'lo", "ba'zi", "la'nat", "sa'ol", "qa'ba", "qi'ya", "si'ra", "ti'ro",
    "e'tibor", "e'tiroz", "e'tiqod", "e'jodkor", "o'tkaz", "qis'm", "qism",
    "his'sira", "mal'a", "zal'a", "gul'ori",
}

@dataclass
class MorphToken:
    text: str
    affix_id: str
    meaning_uz: str
    meaning_en: str
    affix_type: str
    category: str
    affix_obj: Optional[Any] = None # Hold the original Affix object

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "id": self.affix_id,
            "meaning_uz": self.meaning_uz,
            "meaning_en": self.meaning_en,
            "type": self.affix_type,
            "category": self.category,
        }

@dataclass
class AnalysisResult:
    original: str
    root: str
    suffixes: List[MorphToken] = field(default_factory=list)
    is_found: bool = False
    is_compound: bool = False
    confidence: float = 0.0

    @property
    def status(self) -> str:
        return "found" if self.is_found else "unknown"

    @property
    def suffix_chain(self) -> str:
        if not self.suffixes:
            return ""
        return " -> ".join(f"+{s.text}({s.meaning_uz})" for s in self.suffixes)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "root": self.root,
            "status": self.status,
            "is_compound": self.is_compound,
            "confidence": round(self.confidence, 3),
            "suffixes": [s.to_dict() for s in self.suffixes],
            "suffix_chain": self.suffix_chain,
        }

class MorphAnalyzer:
    def __init__(
        self,
        dictionary: Dictionary,
        max_depth: int = 15,
        min_root_length: int = 2,
        use_trie: bool = True,
        cache_size: int = 50000,
        use_ranker: bool = True,
    ):
        self.dictionary = dictionary
        # O'rganilgan nomzod tanlagich (og'irliklar yo'q bo'lsa None: eski ball tizimi).
        self._ranker = CandidateRanker.load() if use_ranker else None
        global _REGISTRY_DICTIONARY
        if _REGISTRY_DICTIONARY is None or len(dictionary) > len(_REGISTRY_DICTIONARY):
            _REGISTRY_DICTIONARY = dictionary
        self.max_depth = max_depth
        self.min_root_length = min_root_length
        self._cache_size = cache_size
        self._use_trie = use_trie
        
        raw_variants = get_suffix_variants()
        self._suffix_list = []
        for item in raw_variants:
            affix_obj = item["affix"]
            entry = {
                "text": item["variant"],
                "id": affix_obj.id,
                "meaning_uz": affix_obj.meaning_uz,
                "meaning_en": affix_obj.meaning_en,
                "type": affix_obj.type.value,
                "category": affix_obj.category.value,
                "affix_obj": affix_obj
            }
            self._suffix_list.append(entry)

        self._suffix_trie = None
        if use_trie:
            self._suffix_trie = SuffixTrie()
            for suf in self._suffix_list:
                self._suffix_trie.insert(suf["text"], suf)

        self._cache: Dict[str, AnalysisResult] = {}
        # O'zak uzunligi bo'nusi: 2-3 harfli soxta o'zaklar (qo, bi, ko, yu)
        # aslida uzunroq o'zakning bir qismi — ularni qisqa deb jazolaymiz.
        self._SHORT_PENALTY_STEMS = {"qo", "bi", "ko", "yu", "yo", "bo", "to", "do"}

    def _penalize_short_root(self, res: 'AnalysisResult') -> None:
        if res.root in self._SHORT_PENALTY_STEMS and len(res.suffixes) >= 1:
            res.confidence -= 0.35

    def analyze(self, word: str) -> AnalysisResult:
        word_lower = normalize_uzbek(word)
        if word_lower in self._cache:
            return self._cache[word_lower]

        results = self.analyze_all(word_lower, max_results=8)
        # Qisqa soxta o'zaklarni jazolaymiz (qo+la+di emas, qol+adi kerak)
        for r in results:
            self._penalize_short_root(r)
        results.sort(key=lambda r: r.confidence, reverse=True)  # jarimadan keyin qayta saralash
        if results:
            result = results[0]
            if self._ranker is not None and len(results) > 1:
                result = self._ranker.best(word_lower, results, self.dictionary.stems)
        else:
            result = AnalysisResult(original=word, root=word_lower, is_found=False, confidence=0.0)

        if len(self._cache) >= self._cache_size:
            keys = list(self._cache.keys())
            for k in keys[: len(keys) // 2]:
                del self._cache[k]
        self._cache[word_lower] = result
        return result

    def analyze_all(self, word: str, max_results: int = 5) -> List[AnalysisResult]:
        word_lower = normalize_uzbek(word)
        results = []
        self._analyze_all_recursive(word_lower, word, [], results, 0, max_results)
        
        # In case we can't find anything, try BPE fallback (stripping suffixes even if root is OOV)
        if not results:
            self._analyze_all_recursive_oov(word_lower, word, [], results, 0, 5)
            
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:max_results]

    # Suffix formalarini keyinchalik _is_bare_suffix orqali tekshiramiz
    _SUFFIX_FORMS: Optional[Set[str]] = None

    def _is_bare_suffix(self, candidate: str) -> bool:
        """Agar butin so'z oddiy qo'shimcha bo'lsa (masalan 'lar', 'dan') True."""
        if MorphAnalyzer._SUFFIX_FORMS is None:
            MorphAnalyzer._SUFFIX_FORMS = {s["text"] for s in get_suffix_variants()}
        return candidate in MorphAnalyzer._SUFFIX_FORMS

    def _apostrophe_variants(self, candidate: str) -> List[str]:
        """oʻ/gʻ (U+02BB) va tutuq belgisi (U+02BC) farqi morfologik jihatdan
        muhim emas — ikkala holatda ham barcha apostrof kombinatsiyalarini
        sinab ko'ramiz. Masalan: bogʻlarimizda -> [bogʻ..., bogʼ...],
        maʼnoda -> [maʼno..., maʻno...]. Shuningdek apostrofsiz variant ham."""
        positions = [i for i, ch in enumerate(candidate) if ch in ("ʻ", "ʼ", "'")]
        if not positions:
            return []
        import itertools
        n = len(positions)
        # Kartezian kombinatsiyalar sonini cheklaymiz (timeout xavfsizligi):
        # 2^n <= 32 bo'lsa to'liad, aks holda faqat bittalik almashinuvlar.
        if n <= 5:
            combos = list(itertools.product(("ʻ", "ʼ"), repeat=n))
        else:
            combos = []
            all_first = tuple(["ʻ"] * n)
            all_second = tuple(["ʼ"] * n)
            seen = set()
            for base in (all_first, all_second):
                if base not in seen:
                    seen.add(base)
                    combos.append(base)
            for k in range(n):
                for sym in ("ʻ", "ʼ"):
                    lst = list(all_first)
                    lst[k] = sym
                    t = tuple(lst)
                    if t not in seen:
                        seen.add(t)
                        combos.append(t)
        variants: List[str] = []
        for combo in combos:
            chars = list(candidate)
            for pos, sym in zip(positions, combo):
                chars[pos] = sym
            variants.append("".join(chars))
        # 2) apostrofsiz variant (hunspell bazasi ba'zan tashlab yozgan bo'ladi)
        stripped = candidate.replace("ʻ", "").replace("ʼ", "").replace("'", "")
        if stripped != candidate:
            variants.append(stripped)
        return variants

    # Qattiq taqiqlangan soxta bo'linishlar (bug'lanish uchun):
    # masalan "yong'oq" so'zini "yong + oq" deb yorish noto'g'ri.
    # Umumiy blok: remaining undosh bilan tugasa va kesilayotgan qo'shimcha
    # shu undoshdan boshlansa ('yong'+'gʻogʻzor', 'qayta'+'rga', 'ko'+'rib') —
    # bu holda undosh o'zakka tegishli, qo'shimcha emas.
    _GAP_BLOCK_FIRST = set("bcdfghjklmnpqrstvwxyz")
    _BAD_SPLITS = {
        ("yong", "oʻ"), ("yong", "o"), ("yong", "og"), ("yong", "oʻg"),
        ("yon", "oq"), ("yon", "og"), ("bor", "oq"), ("bor", "og"),
        ("yoq", "oq"), ("tog", "oq"), ("bog", "oq"), ("rog", "oq"),
        ("qoq", "oq"), ("choq", "oq"), ("noq", "oq"), ("moq", "oq"),
        ("toq", "oq"), ("doq", "oq"), ("boq", "oq"), ("soq", "oq"),
    }

    # Butun so'z bo'lishi kerak bo'lgan, lekin qisqa o'zak + qo'shimcha
    # sifatida xato ajraladigan yopishqoq shakllar (fe'l birikmalari va
    # ravishlar). Ularni lug'atda bo'lsa ham bo'lishga qo'ymaymiz.
    _ATOMIC_WORDS = {
        "qayta", "keyin", "lekin", "chunki", "garchi", "ammo", "agar",
        "shunchaki", "aniq", "sodda", "murakkab", "chiroyli", "ayrim",
        "avval", "kemtin", "daf'atan", "birdaniga",
    }
    # So'z bo'linishi mumkin bo'lgan istisnolar: 'ko'rib -> ko'r+ib',
    # 'kelib -> kel+ib' — chunki bu fe'l o'zak + ravishdosh.
    _FORCED_SPLITS = {
        "koʻrib": ("koʻr", "ib"), "korib": ("kor", "ib"),
        "kelib": ("kel", "ib"), "ketib": ("ket", "ib"),
        "qolib": ("qol", "ib"), "kelib": ("kel", "ib"),
    }

    @staticmethod
    def _split_apostrophes(candidate: str) -> List[str]:
        """Apostrofsiz variantlarni qaytaradi: 'burchagʻidan' -> ['burchagidan'].
        Tutuq/oʻ belgilari qoʻshimchadan oldin turganda ularni tashlab
        koʻrish kerak: chirogʻimizda -> chirog+imiz... emas, balki
        remaining='chirogʻ' holatida 'chirog' varianti ham tekshiriladi."""
        out = []
        stripped = candidate.replace("ʻ", "").replace("ʼ", "")
        if stripped != candidate and len(stripped) >= 2:
            out.append(stripped)
        return out

    def _is_bad_split(self, root: str, suffixes: List[MorphToken]) -> bool:
        """Soxta bo'linishlarni aniqlaydi (faqat aniq ro'yxat bo'yicha)."""
        if not suffixes:
            return False
        leftmost = suffixes[-1]  # reversed order: chapdagi qo'shimcha
        key = (root, leftmost.text[0] if leftmost.text else "")
        if key in self._BAD_SPLITS:
            return True
        # 'yongʻogʻzor' kabi tutash ikki gʻ soxta o'zaklarini bloklaymiz
        if root.endswith("gʻogʻ") or root.endswith("ggʻ"):
            return True
        return False

    @staticmethod
    def _is_infinitive_base(candidate: str) -> bool:
        return _find_verb_base(candidate) is not None

    def _verb_stem_from_infinitive(self, candidate: str) -> Optional[str]:
        """Lug'atdagi infinitiv shakldan fe'l o'zakini tiklaydi:
        'bil'   <- bilmoq ('moq' tushadi)
        'oʻqi'  <- oʻqimoq ('moq' tushadi, 'i' saqlanadi: oʻqi+i... emas —
                 bu yerda 'oʻqi' ham, 'oʻq' ham qaytariladi)
        Faqat 'found=True' imkoniyati uchun, lug'atda alohida saqlanmagan bo'lsa."""
        return _find_verb_base(candidate)

    def _lookup_stem(self, candidate: str, allow_morphophonology: bool = True,
                     suffixes: Optional[List[MorphToken]] = None) -> Optional[str]:
        resolved = self._lookup_stem_raw(candidate, allow_morphophonology)
        if resolved is not None and self._is_bad_split(resolved, suffixes or []):
            return None
        return resolved

    def _lookup_stem_raw(self, candidate: str, allow_morphophonology: bool = True) -> Optional[str]:
        # Atomic so'zlar (qayta, lekin...) qisqa o'zak sifatida ishtimol
        # bo'linishiga yo'l qo'ymaymiz — ular faqat butun so'z bo'la oladi.
        if candidate in self._ATOMIC_WORDS and len(candidate) <= 4:
            return None
        if candidate in self.dictionary:
            return candidate

        if not allow_morphophonology:
            return None

        # 0. Apostrof variantlari: oʻ/gʻ vs tutuq belgisi farqini morfologik
        #    tekshiruvda yumshatamiz — lug'atdagi haqiqiy shaklni qaytaramiz.
        for alt in self._apostrophe_variants(candidate):
            if len(alt) >= 2 and alt in self.dictionary:
                return alt

        # 1. Morphophonological pre-generated map (shahr -> shahar, qishlog' -> qishloq)
        if candidate in _MORPHOPHONOLOGICAL_MAP:
            restored = _MORPHOPHONOLOGICAL_MAP[candidate]
            if restored in self.dictionary:
                return restored

        # 1.5. OOV nomlar/texnik so'zlar: apostrof + qo'shimcha (Python'da -> Python + da).
        #      Agar candidate ichida tutuq belgisi bor va undan oldingi qism lotin nom bo'lsa,
        #      apostrofdan keyingi qismni tashlab, asl nomni qaytarishga ruxsat beramiz.
        for ap in ("ʼ", "ʻ", "’"):
            if ap in candidate:
                head = candidate.split(ap)[0]
                if len(head) >= 3 and head.replace("'", "").isascii() and head not in self.dictionary:
                    return head  # OOV nom — tokenizer uni BPE/subtoken sifatida ishlatadi
        # 2. Dynamic alternation: -gʻ -> -q / -k, -g -> -k
        if candidate.endswith("gʻ") or candidate.endswith("g'"):
            alt_q = candidate[:-2] + "q"
            if alt_q in self.dictionary: return alt_q
            alt_k = candidate[:-2] + "k"
            if alt_k in self.dictionary: return alt_k
        if candidate.endswith("g"):
            alt_k = candidate[:-1] + "k"
            if alt_k in self.dictionary: return alt_k

        # 3. Fe'l infinitividan qisqargan o'zak: 'bil' <- bilmoq, 'oʻqi' <- oʻqimoq.
        if candidate.isascii() and re.fullmatch(r"[bcdfghjklmnpqrstvwxz]*[aeiou][aeiou]*", candidate):
            vb = _find_verb_base(candidate)
            if vb is not None:
                return candidate

        # 4. O'zbekcha so'zlarda tutuq belgisi (ʼ) ko'pincha tushirib yoziladi:
        #    maslahat+dan -> maslah'tan bo'linishida remaining='maslah' ni
        #    lug'atdagi 'maslahat' ga bog'lash uchun apostrofsiz variantni
        #    ham sinab ko'ramiz (faqat oddiy apostrof tashlab).
        stripped_ap = candidate.replace("ʼ", "")
        if stripped_ap != candidate and len(stripped_ap) >= 3:
            if stripped_ap in self.dictionary:
                return stripped_ap
            if self._verb_stem_from_infinitive(stripped_ap) is not None:
                return stripped_ap

        return None

    def _check_suffix_order(self, current_suffixes: List[MorphToken], new_suffix: MorphToken) -> bool:
        """
        Check if new_suffix can precede current_suffixes (since we strip from right to left,
        new_suffix is actually to the LEFT of current_suffixes in the word).
        e.g., word = maktab + lar(new) + imiz(current). 
        So new_suffix is attached BEFORE current_suffixes.
        """
        if not current_suffixes:
            return True
            
        right_suffix = current_suffixes[-1]
        right_category_name = right_suffix.category
        
        # --- Qattiq nom zanjiri qoidasi (ot => plural => poss => case => ...):
        # PLURAL/POSSESSION/CASE/PARTICIPLE/MOOD/PERSON kabi infleksion
        # qatlamlar faqat ruxsat etilgan tartibda kelishi kerak. Masalan
        # bog'larimizda: +lar(plural) ni +imiz(poss) boshqaradi — bu to'g'ri;
        # lekin 'bog'+a+r+imiz+da (tense+participle fe'l zanjiri) otga
        # nisbatan noto'g'ri — quyidagi qatlam tartibi tekshiruvi uni bloklaydi.
        _LAYER_ORDER = {
            "word_formation": 0,
            "diminutive": 0,
            "plural": 1,
            "possession": 2,
            "case": 3,
            "participle": 4,
            "gerund": 4,
            "tense": 5,
            "mood": 6,
            "negation": 6,
            "voice": 7,
            "person": 8,
            "question": 9,
            "particle_cat": 9,
        }
        _NOUN_LAYERS = {"word_formation", "diminutive", "plural", "possession", "case"}
        new_layer = _LAYER_ORDER.get(new_suffix.category)
        right_layer = _LAYER_ORDER.get(right_category_name)
        if new_layer is not None and right_layer is not None and \
           new_suffix.category in _NOUN_LAYERS and right_category_name in _NOUN_LAYERS:
            # new_suffix CHAP tomonda turadi, ya'ni zanjirda oldinroq keladi
            if new_layer > right_layer:
                return False
            # Bir xil qatlamdagi takrorlanishga faqat plural=>possessive
            # aralash zanjirida (+lar+imiz+) ruxsat: oddiy ikki marta
            # bir xil qatlamni (case+case) taqiqlaymiz.
            if new_layer == right_layer and new_suffix.category == right_category_name \
               and new_suffix.category in ("case", "tense"):
                return False

        right_affix_obj = right_suffix.affix_obj
        if right_affix_obj and right_affix_obj.follows is not None:
            # new_suffix must be in the follows list of right_suffix
            new_cat_enum = None
            for c in AffixCategory:
                if c.value == new_suffix.category:
                    new_cat_enum = c
                    break
            
            if new_cat_enum is not None and new_cat_enum not in right_affix_obj.follows:
                # debug logging removed — kasb etish tartibi tekiruvi natijasi yashirin
                return False
                
        return True

    def _check_allomorphs(self, remaining: str, suffix_text: str) -> bool:
        """Check vowel harmony and assimilation rules."""
        if suffix_text in ["qa", "qan", "qach"]:
            return remaining.endswith("k") or remaining.endswith("q")
        if suffix_text in ["ka", "kan", "kach"]:
            return remaining.endswith("k") or remaining.endswith("q")
        if suffix_text in ["ga", "gan", "gach"]:
            if remaining.endswith("k") or remaining.endswith("q"):
                return False
        # -layotgan/-ayotgan: 'l' faqat fe'l o'zagi -la/-lan bilan tugasa ruxsat
        # (yozmoq -> yoza+l+yotgan EMAS; haydamoq -> hayda+la+yotgan TO'G'RI).
        # Aks holda 'chiqmaganli', 'berolmagani' kabi soxta kesimlar chiqadi.
        if suffix_text == "l" and remaining[-1:] in ("a", "e", "i", "o", "u", "oʻ"):
            if not self._is_verbish_stem(remaining):
                return False
        return True

    def _is_verbish_stem(self, remaining: str) -> bool:
        """O'zak '-la/-lan/-lashtir' yasovchisi bilan tugayotganini tekshiradi:
        remaining='haydala' -> 'hayda' lug'atda bo'lsa True."""
        for cut, suf in ((2, "la"), (3, "lan")):
            if remaining.endswith(suf):
                base = remaining[:-cut]
                if len(base) >= 2 and self._lookup_stem(base, allow_morphophonology=False):
                    return True
        return False

    def _analyze_all_recursive(
        self,
        word_lower: str,
        original: str,
        current_suffixes: List[MorphToken],
        results: List[AnalysisResult],
        depth: int,
        max_results: int,
    ) -> None:
        if len(results) >= max_results * 2: # Gather some extra for sorting
            return

        resolved_root = self._lookup_stem(word_lower, allow_morphophonology=(depth > 0),
                                          suffixes=current_suffixes)
        if resolved_root is None and depth == 0:
            # Fe'l infinitividan qisqargan o'zak (bilmoq -> bil-, kelmoq -> kel-)
            resolved_root = self._verb_stem_from_infinitive(word_lower)
        if resolved_root is not None:
            valid = True
            if current_suffixes:
                valid = self._check_allomorphs(resolved_root, current_suffixes[-1].text)
            if valid and self._is_bad_split(resolved_root, current_suffixes):
                valid = False
                
            if valid:
                confidence = self._compute_confidence(resolved_root, current_suffixes, depth, True)
                res = AnalysisResult(
                        original=original,
                        root=resolved_root,
                        suffixes=list(reversed(current_suffixes)), # Reverse to left-to-right order
                        is_found=True,
                        confidence=confidence,
                    )
                results.append(res)

        if depth >= self.max_depth:
            return

        candidates = self._suffix_trie.find_suffixes(word_lower) if self._use_trie else self._find_suffixes_linear(word_lower)

        for suffix_text, suffix_data in candidates:
            remaining = word_lower[: -len(suffix_text)]
            if len(remaining) < self.min_root_length:
                continue

            # O'zbekcha o'zaklar unli bilan tugashi kerak (bo'g'in qonuni).
            # Undosh bilan tugagan kesimlar faqat ruxsat etilgan holatlarda:
            #  - i/u bilan boshlanadigan egalik/qo'shimcha (kitob+imiz),
            #  - lug'atdagi tutuqdan keyin keladigan tortishish shakllari
            #    (tutuq qo'shimcha tarkibida: boʻyin+da -> remaining 'boʻy'),
            #  - morfofonologik map qisqarishlari (shahr -> shahar),
            #  - -lari 3-shaxs ko'plik egalik (kitob+lari).
            _VOWS = "aeiou" + "\u02bb\u02bc"
            if remaining[-1:] not in _VOWS and not remaining.endswith("o\u02bb"):
                ok_cons = False
                # Egalik/ot qo'shimchalari undosh o'zakka bemalol qo'shiladi:
                # kitob+imiz, maktab+da, talaba+lar, uy+ning, dastur+chi...
                if suffix_text[:1] in ("i", "u", "d", "c", "n", "l", "g", "q", "k", "m", "b", "t", "s"):
                    ok_cons = True
                if not ok_cons and remaining in _MORPHOPHONOLOGICAL_MAP:
                    ok_cons = True
                if not ok_cons and suffix_text == "lari":
                    ok_cons = True
                if not ok_cons and remaining.endswith("\u02bb") and \
                   self._lookup_stem(remaining + "\u02bc", allow_morphophonology=False):
                    ok_cons = True
                if not ok_cons:
                    continue

            if not self._check_allomorphs(remaining, suffix_text):
                continue

            # Bo'shliq (root-gap) tekshiruvi: chapda qolgan qism morfologik
            # jihatdan "tugallanmagan" bo'lmasin. Soxta kesimlarning belgisi:
            # remaining oxiri undosh va u endi kesilgan qo'shimchaning BIRINCHI
            # harfi bilan bir xil — ya'ni qo'shimcha o'zakning oxirgi harfini
            # o'z ichiga olayotgandek ko'rinadi ('lar' -> 'la'+'r', 'chi'+'lar').
            if len(remaining) >= 2 and remaining[-1] not in _VOWS and \
               remaining[-1] not in "\u02bb\u02bc" and suffix_text[0] == remaining[-1]:
                # 'qaytargan' -> qayta+r+gan kabi kesimlarda aslida birinchi
                # harfi remaining oxiriga to'g'ri keladigan UZUNROQ variant
                # mavjud bo'lsa ('r' o'rniga 'rga'), qisqasini bloklaymiz.
                longer_exists = any(
                    suf != suffix_text and suf.startswith(suffix_text[0]) and
                    word_lower.endswith(suf) and len(suf) > len(suffix_text)
                    for suf, _ in candidates)
                if longer_exists or True:
                    continue

            token = MorphToken(
                text=suffix_text,
                affix_id=suffix_data["id"],
                meaning_uz=suffix_data["meaning_uz"],
                meaning_en=suffix_data["meaning_en"],
                affix_type=suffix_data["type"],
                category=suffix_data["category"],
                affix_obj=suffix_data["affix_obj"]
            )
            
            if not self._check_suffix_order(current_suffixes, token):
                continue

            self._analyze_all_recursive(
                remaining,
                original,
                current_suffixes + [token],
                results,
                depth + 1,
                max_results,
            )

    def _analyze_all_recursive_oov(self, word_lower, original, current_suffixes, results, depth, max_results):
        """Fallback for OOV words. Same logic but allows any remaining string as root."""
        if depth > 0 and len(word_lower) >= self.min_root_length:
            confidence = self._compute_confidence(word_lower, current_suffixes, depth, False)
            results.append(
                AnalysisResult(
                    original=original,
                    root=word_lower,
                    suffixes=list(reversed(current_suffixes)),
                    is_found=False,
                    confidence=confidence,
                )
            )

        if depth >= 5: return # Limit depth for OOV

        candidates = self._suffix_trie.find_suffixes(word_lower) if self._use_trie else self._find_suffixes_linear(word_lower)
        for suffix_text, suffix_data in candidates:
            remaining = word_lower[: -len(suffix_text)]
            if len(remaining) < self.min_root_length: continue
            
            token = MorphToken(
                text=suffix_text, affix_id=suffix_data["id"], meaning_uz=suffix_data["meaning_uz"],
                meaning_en=suffix_data["meaning_en"], affix_type=suffix_data["type"], category=suffix_data["category"],
                affix_obj=suffix_data["affix_obj"]
            )
            
            if not self._check_suffix_order(current_suffixes, token): continue
            
            self._analyze_all_recursive_oov(remaining, original, current_suffixes + [token], results, depth + 1, max_results)

    def _find_suffixes_linear(self, word: str) -> List[Tuple[str, dict]]:
        return [(suf["text"], suf) for suf in self._suffix_list if word.endswith(suf["text"])]

    def _compute_confidence(self, root: str, suffixes: List[MorphToken], depth: int, is_found: bool) -> float:
        score = 0.0
        if is_found: score += 0.5
        # O'zak uzunligi bo'nusi: kichik o'zak + ulkan qo'shimcha bloki
        # sun'iy ravishda yuqori ball olmasin — faqat 4 harfdan boshlab.
        score += min(0.25, max(0, len(root) - 4) * 0.03)

        num_suffixes = len(suffixes)
        # Har bir tasniflangan (follows-tartib tekshiruvidan o'tgan)
        # qo'shimchaga alohida ball — chuqur morf zanjirni mukofotlaydi.
        score += 0.08 * num_suffixes
        if num_suffixes == 0: score += 0.15
        elif num_suffixes <= 2: score += 0.10
        elif num_suffixes <= 5: score += 0.05
        # Soxta sayoz bo'linishlarni jazolaymiz: agar chapdagi qo'shimcha
        # o'zakka "yopishib" qolgan bo'lsa (kelgani+ngiz emas, kelgani+ingiz
        # kerak — ya'ni remaining oxiri unli va qo'shimcha i/ng/bilan
        # boshlansa ham aslida u boshqa qo'shimchaning bo'lagi).
        covered = sum(len(s.text) for s in suffixes)
        if num_suffixes <= 2 and covered >= 8:
            score -= 0.12
        # O'zak infinitiv bazasi bilan aniq mos kelsa (bil, o'qi, kel) — bonus
        if is_found and self._is_infinitive_base(root):
            score += 0.06
        # Fe'l shakl qo'shimchalari (zamon/sifatdosh/shaxs) ketma-ketligi:
        # +di/+gan/+man/-siz/-yotgan kabi aniq fe'l qo'shimchalari zanjirida
        # bo'linishni mukofotlaymiz ('qoladi -> qo+la+di' kabi soxta kesimlar
        # 'qol+adi' variantidan pastda qolsin).
        _VERBY = {"di", "di", "gan", "kan", "moq", "yotgan", "ayotgan", "oyotgan",
                  "man", "miz", "siz", "san", "sz", "ar", "or", "may", "mas",
                  "ding", "tigan", "adigan", "ishi", "ib", "lab", "lib", "yb"}
        verb_chain = sum(1 for s in suffixes if s.text in _VERBY)
        if verb_chain >= 2:
            score += 0.05 * verb_chain
        elif verb_chain == 1 and num_suffixes >= 1:
            score += 0.03
        # Uzun, tartibga mos zanjirlarni qo'shimcha mukofotlaymiz
        if num_suffixes >= 4:
            score += 0.10
        elif num_suffixes == 3:
            score += 0.06
        
        # Add a tiny amount for priority to break ties (e.g. lar plural vs lar person)
        for s in suffixes:
            if s.affix_obj:
                score += s.affix_obj.priority * 0.001

        return max(0.0, min(score, 1.0))

def create_analyzer(dictionary=None, load_hunspell=True, max_depth=15, use_trie=True) -> MorphAnalyzer:
    if dictionary is None:
        dictionary = Dictionary()
        if load_hunspell:
            try: dictionary.load_hunspell()
            except Exception: pass
    return MorphAnalyzer(dictionary=dictionary, max_depth=max_depth, use_trie=use_trie)
