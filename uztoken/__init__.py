"""
UzToken — O'zbek tili uchun morfologik tokenizatsiya kutubxonasi.

Uzbek Morphological Tokenizer Library.

A hybrid tokenizer that combines rule-based morphological analysis
with BPE subword tokenization for complete Uzbek text processing.

Usage:
    >>> import uztoken
    >>> tok = uztoken.UzTokenizer()
    >>> tok.load_dictionary()
    >>> tokens = tok.tokenize("Maktablarimizda o'qiyapmiz")
    >>> for t in tokens:
    ...     print(t)
"""

__version__ = "0.1.0"
__author__ = "UzToken Team"

# Core classes
from .tokenizer import UzTokenizer, Token, create_tokenizer, quick_tokenize
from .morphology import MorphAnalyzer, AnalysisResult, MorphToken, create_analyzer
from .dictionary import Dictionary
from .bpe import BPETokenizer
from .normalizer import (
    normalize_uzbek,
    normalize_apostrophes,
    cyrillic_to_latin,
    clean_text,
)
from .trie import SuffixTrie, DictionaryTrie
from .affixes import (
    Affix,
    AffixType,
    AffixCategory,
    get_all_affixes,
    get_affix_by_id,
    get_suffix_variants,
)

__all__ = [
    # Main tokenizer
    "UzTokenizer",
    "Token",
    "create_tokenizer",
    "quick_tokenize",
    # Morphology
    "MorphAnalyzer",
    "AnalysisResult",
    "MorphToken",
    "create_analyzer",
    # Dictionary
    "Dictionary",
    # BPE
    "BPETokenizer",
    # Normalizer
    "normalize_uzbek",
    "normalize_apostrophes",
    "cyrillic_to_latin",
    "clean_text",
    "is_uzbek_word",
    # Data structures
    "SuffixTrie",
    "DictionaryTrie",
    # Affixes
    "Affix",
    "AffixType",
    "AffixCategory",
    "get_all_affixes",
    "get_affix_by_id",
    "get_affixes_by_category",
    "get_suffix_variants",
]
