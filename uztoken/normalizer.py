import html
import re
import unicodedata
from typing import Dict, List, Optional, Pattern, Set, Tuple, Any

# ============================================================================
# Apostrophe Variants & Translation Table
# ============================================================================

# All known Unicode apostrophe and quotation variants mapped to standard ASCII '
APOSTROPHE_VARIANTS: Tuple[str, ...] = (
    "\u02BB",  # ʻ MODIFIER LETTER TURNED COMMA (official Uzbek Latin Oʻ, Gʻ)
    "\u02BC",  # ʼ MODIFIER LETTER APOSTROPHE (official Uzbek tutuq belgisi)
    "\u2019",  # ’ RIGHT SINGLE QUOTATION MARK (most common curly apostrophe)
    "\u2018",  # ‘ LEFT SINGLE QUOTATION MARK (curly quote)
    "\u0060",  # ` GRAVE ACCENT (common keyboard substitute)
    "\u02B9",  # ʹ MODIFIER LETTER PRIME
    "\u2032",  # ′ PRIME
    "\u02BF",  # ʿ MODIFIER LETTER LEFT HALF RING (Arabic 'ayn in translit)
    "\u0027",  # ' APOSTROPHE (standard ASCII single quote)
    "\u00B4",  # ´ ACUTE ACCENT
    "\u02BE",  # ʾ MODIFIER LETTER RIGHT HALF RING (Arabic hamza in translit)
    "\u201B",  # ‛ SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u201A",  # ‚ SINGLE LOW-9 QUOTATION MARK
    "\u2035",  # ‵ REVERSED PRIME
    "\uFF07",  # ＇ FULLWIDTH APOSTROPHE
    "\u02CA",  # ˊ MODIFIER LETTER ACUTE ACCENT
    "\u02CB",  # ˋ MODIFIER LETTER GRAVE ACCENT
    "\u02C8",  # ˈ MODIFIER LETTER VERTICAL LINE
    "\u02BD",  # ʽ MODIFIER LETTER REVERSED COMMA
    "\u0092",  # ’ Windows-1252 right quote
)

_APOSTROPHE_TRANS_TABLE: Dict[int, str] = str.maketrans(
    {char: "'" for char in APOSTROPHE_VARIANTS}
)

_RE_COMBINING_APOSTROPHE: Pattern = re.compile(r"([ogOG])[\u0300-\u036F]+")
_RE_ORPHANED_APOSTROPHE: Pattern = re.compile(r"\b([ogOG])\s+['\u02BB\u02BC]\s*([a-zA-Z])")
_RE_ORPHANED_APOSTROPHE_TRAILING: Pattern = re.compile(r"\b([ogOG])\s+['\u02BB\u02BC](?!\w)")

# Regex for applying O'/G' vs Tutuq belgisi rules
# Match [oOgG] followed by apostrophe -> ʻ
_RE_O_G_APOS: Pattern = re.compile(r"([oOgG])'")
# Match other vowel followed by apostrophe -> ʼ (simplified rule)
_RE_TUTUQ_APOS: Pattern = re.compile(r"([aAeEuUiI])'")
# Remaining apostrophes in words might also be tutuq belgisi (e.g., san'at -> sanʼat)
_RE_OTHER_APOS: Pattern = re.compile(r"([b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z])'")

# ============================================================================
# Whitespace & Cleaning Regular Expressions
# ============================================================================
_RE_WHITESPACE: Pattern = re.compile(
    r"[\s\u00A0\u1680\u2000-\u200A\u200B\u200C\u200D\u2028\u2029\u202F\u205F\u3000\uFEFF]+"
)

_RE_HTML_TAGS: Pattern = re.compile(r"<[^>]+>")
_RE_URLS: Pattern = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s<>\"]+?(?=[.,!?;:]?(?:\s|$|\Z))",
    re.IGNORECASE,
)
_RE_EMAILS: Pattern = re.compile(
    r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"
)
_RE_DECORATIVE_SYMBOLS: Pattern = re.compile(r"[-=~*#^_]{2,}")
_RE_EXCESSIVE_DOTS: Pattern = re.compile(r"\.{4,}")
_RE_TWO_DOTS: Pattern = re.compile(r"(?<!\.)\.\.(?!\.)")
_RE_EXCESSIVE_PUNCT: Pattern = re.compile(r"([!?,;:—–])\1+")
_RE_MIXED_EXCL_QUEST: Pattern = re.compile(r"(?:!\?|\?!)[!?]+")
_RE_REPEATED_QUOTES: Pattern = re.compile(r"\"{2,}|'{2,}")
_RE_SPACE_BEFORE_PUNCT: Pattern = re.compile(r"\s+([.,!?:;])")

# ============================================================================
# Cyrillic rules
# ============================================================================
CYRILLIC_VOWELS: Set[str] = set("аеёиоуэюяўАЕЁИОУЭЮЯЎ")
CYRILLIC_UPPERCASE: Set[str] = set("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЎҚҒҲ")

BASE_CYRILLIC_TO_LATIN: Dict[str, str] = {
    "а": "a",   "б": "b",   "в": "v",   "г": "g",   "д": "d",
    "ж": "j",   "з": "z",   "и": "i",   "й": "y",   "к": "k",
    "л": "l",   "м": "m",   "н": "n",   "о": "o",   "п": "p",
    "р": "r",   "с": "s",   "т": "t",   "у": "u",   "ф": "f",
    "х": "x",   "ы": "i",   "ь": "",    "э": "e",   "қ": "q",
    "ҳ": "h",   "ъ": "ʼ",   "ў": "oʻ",  "ғ": "gʻ",
    "ӯ": "oʻ",  "ґ": "gʻ",  "і": "i",
    "А": "A",   "Б": "B",   "В": "V",   "Г": "G",   "Д": "D",
    "Ж": "J",   "З": "Z",   "И": "I",   "Й": "Y",   "К": "K",
    "Л": "L",   "М": "M",   "Н": "N",   "О": "O",   "П": "P",
    "Р": "R",   "С": "S",   "Т": "T",   "У": "U",   "Ф": "F",
    "Х": "X",   "Ы": "I",   "Ь": "",    "Э": "E",   "Қ": "Q",
    "Ҳ": "H",   "Ъ": "ʼ",   "Ў": "Oʻ",  "Ғ": "Gʻ",
    "Ӯ": "Oʻ",  "Ѓ": "Gʻ",  "І": "I",
}

CYRILLIC_DIGRAPHS: Dict[str, Tuple[str, str, str]] = {
    "ё": ("yo", "Yo", "YO"),
    "ц": ("ts", "Ts", "TS"),
    "ч": ("ch", "Ch", "CH"),
    "ш": ("sh", "Sh", "SH"),
    "щ": ("sh", "Sh", "SH"),
    "ю": ("yu", "Yu", "YU"),
    "я": ("ya", "Ya", "YA"),
}

def normalize_apostrophes_lossless(text: str) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Standardize apostrophes to ʻ (U+02BB) for oʻ/gʻ and ʼ (U+02BC) for tutuq belgisi.
    Returns (normalized_text, [(offset, original_char)]) for lossless decoding.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    if not text:
        return "", []

    # Fast normalization to ascii ' just to unify processing
    temp_chars = list(text)
    replacements = []
    
    # 1. First pass: find all variants and remember their original positions
    for i, char in enumerate(temp_chars):
        if char in APOSTROPHE_VARIANTS:
            replacements.append((i, char))
            temp_chars[i] = "'"
            
    temp_text = "".join(temp_chars)
    
    # 2. Rule-based application of ʻ and ʼ
    # Rule 1: o, g, O, G + ' -> ʻ (U+02BB)
    temp_text = _RE_O_G_APOS.sub(r"\1ʻ", temp_text)
    
    # Rule 2: other vowels/consonants + ' -> ʼ (U+02BC)
    temp_text = _RE_TUTUQ_APOS.sub(r"\1ʼ", temp_text)
    temp_text = _RE_OTHER_APOS.sub(r"\1ʼ", temp_text)
    
    # Finally, any leftover ' we convert to ʼ as a fallback for tutuq belgisi inside words,
    # or keep as ' if it's external punctuation (but for now let's just use ʼ if it's between letters)
    temp_text = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "ʼ", temp_text)
    
    return temp_text, replacements


def normalize_apostrophes(text: str) -> str:
    """Standardize apostrophes based on Uzbek rules."""
    norm_text, _ = normalize_apostrophes_lossless(text)
    return norm_text


def restore_apostrophes_lossless(normalized_text: str, replacements: List[Tuple[int, str]]) -> str:
    """Restores the exact original apostrophes into the text based on offset maps."""
    chars = list(normalized_text)
    for offset, orig_char in replacements:
        if offset < len(chars):
            chars[offset] = orig_char
    return "".join(chars)


def normalize_uzbek(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_apostrophes(text)
    text = text.lower()
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text

def cyrillic_to_latin(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    result = []
    n = len(text)
    def _is_upper(c): return c in CYRILLIC_UPPERCASE or ("A" <= c <= "Z")

    for i, char in enumerate(text):
        is_upper_char = _is_upper(char)
        is_all_caps = False
        if is_upper_char:
            if i + 1 < n and _is_upper(text[i + 1]):
                is_all_caps = True
            elif i > 0 and _is_upper(text[i - 1]):
                if i + 1 >= n or not text[i + 1].islower():
                    is_all_caps = True

        if char in ("е", "Е"):
            is_word_start = i == 0 or not (text[i - 1].isalpha() or text[i - 1] in "ўЎқҚғҒҳҲ")
            is_after_vowel_or_sign = i > 0 and (text[i - 1] in CYRILLIC_VOWELS or text[i - 1] in "'ʼ`ʻъЪьЬ\u02BB\u02BC\u2019")
            if is_word_start or is_after_vowel_or_sign:
                if char == "е": result.append("ye")
                else: result.append("YE" if is_all_caps else "Ye")
            else:
                result.append("e" if char == "е" else "E")
            continue

        lower_char = char.lower()
        if lower_char in CYRILLIC_DIGRAPHS:
            low, title, upper = CYRILLIC_DIGRAPHS[lower_char]
            if char.islower(): result.append(low)
            else: result.append(upper if is_all_caps else title)
            continue

        if char in BASE_CYRILLIC_TO_LATIN:
            result.append(BASE_CYRILLIC_TO_LATIN[char])
        else:
            result.append(char)

    return "".join(result)

def clean_text(text: str) -> str:
    if not text: return ""
    text = html.unescape(text)
    text = _RE_HTML_TAGS.sub(" ", text)
    text = _RE_URLS.sub(" ", text)
    text = _RE_EMAILS.sub(" ", text)
    text = _RE_DECORATIVE_SYMBOLS.sub(" ", text)
    text = _RE_EXCESSIVE_DOTS.sub("...", text)
    text = _RE_TWO_DOTS.sub(".", text)
    text = _RE_MIXED_EXCL_QUEST.sub("?!", text)
    text = _RE_EXCESSIVE_PUNCT.sub(r"\1", text)
    text = _RE_REPEATED_QUOTES.sub("'", text)
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text

__all__ = [
    "normalize_apostrophes",
    "normalize_apostrophes_lossless",
    "restore_apostrophes_lossless",
    "normalize_uzbek",
    "cyrillic_to_latin",
    "clean_text",
    "APOSTROPHE_VARIANTS",
]
