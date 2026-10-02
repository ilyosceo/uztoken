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
    ):
        self.dictionary = dictionary
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

    def analyze(self, word: str) -> AnalysisResult:
        word_lower = normalize_uzbek(word)
        if word_lower in self._cache:
            return self._cache[word_lower]

        results = self.analyze_all(word_lower, max_results=1)
        if results:
            result = results[0]
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

    def _lookup_stem(self, candidate: str, allow_morphophonology: bool = True) -> Optional[str]:
        if candidate in self.dictionary:
            return candidate

        if not allow_morphophonology:
            return None

        # 1. Morphophonological pre-generated map (shahr -> shahar, qishlog' -> qishloq)
        if candidate in _MORPHOPHONOLOGICAL_MAP:
            restored = _MORPHOPHONOLOGICAL_MAP[candidate]
            if restored in self.dictionary:
                return restored

        # 2. Dynamic alternation: -gʻ -> -q, -g -> -k
        if candidate.endswith("gʻ") or candidate.endswith("g'"):
            alt_q = candidate[:-2] + "q"
            if alt_q in self.dictionary: return alt_q
        if candidate.endswith("g"):
            alt_k = candidate[:-1] + "k"
            if alt_k in self.dictionary: return alt_k

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
        
        right_affix_obj = right_suffix.affix_obj
        if right_affix_obj and right_affix_obj.follows is not None:
            # new_suffix must be in the follows list of right_suffix
            new_cat_enum = None
            for c in AffixCategory:
                if c.value == new_suffix.category:
                    new_cat_enum = c
                    break
            
            if new_cat_enum is not None and new_cat_enum not in right_affix_obj.follows:
                print(f"REJECTED: {new_suffix.text} ({new_cat_enum}) cannot precede {right_suffix.text} (follows={right_affix_obj.follows})")
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
        return True

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

        resolved_root = self._lookup_stem(word_lower, allow_morphophonology=(depth > 0))
        if word_lower == 'maktab':
            print(f"DEBUG: _lookup_stem('maktab') = {resolved_root}")
        if resolved_root is not None:
            valid = True
            if current_suffixes:
                valid = self._check_allomorphs(resolved_root, current_suffixes[-1].text)
                
            if valid:
                confidence = self._compute_confidence(resolved_root, current_suffixes, depth, True)
                res = AnalysisResult(
                        original=original,
                        root=resolved_root,
                        suffixes=list(reversed(current_suffixes)), # Reverse to left-to-right order
                        is_found=True,
                        confidence=confidence,
                    )
                print(f"ADDED: {res}")
                results.append(res)

        if depth >= self.max_depth:
            return

        candidates = self._suffix_trie.find_suffixes(word_lower) if self._use_trie else self._find_suffixes_linear(word_lower)

        for suffix_text, suffix_data in candidates:
            remaining = word_lower[: -len(suffix_text)]
            if len(remaining) < self.min_root_length:
                continue
                
            if not self._check_allomorphs(remaining, suffix_text):
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
        score += min(0.3, len(root) * 0.05)
        
        num_suffixes = len(suffixes)
        if num_suffixes == 0: score += 0.2
        elif num_suffixes <= 2: score += 0.15
        elif num_suffixes <= 4: score += 0.1
        
        # Add a tiny amount for priority to break ties (e.g. lar plural vs lar person)
        for s in suffixes:
            if s.affix_obj:
                score += s.affix_obj.priority * 0.001
                
        return score

def create_analyzer(dictionary=None, load_hunspell=True, max_depth=15, use_trie=True) -> MorphAnalyzer:
    if dictionary is None:
        dictionary = Dictionary()
        if load_hunspell:
            try: dictionary.load_hunspell()
            except Exception: pass
    return MorphAnalyzer(dictionary=dictionary, max_depth=max_depth, use_trie=use_trie)
