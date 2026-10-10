"""
UzToken Hybrid Tokenizer — Main entry point.

Combines morphological analysis with BPE fallback to provide
complete, zero-failure tokenization of Uzbek text.

Architecture:
    Word → Morphological Analyzer → Found? → [root + suffixes]
                                  → Not found? → BPE subword split
"""

import re
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .normalizer import normalize_uzbek, normalize_apostrophes, clean_text, cyrillic_to_latin
from .fallback_bpe import FallbackBPE
from .dictionary import Dictionary
from .morphology import MorphAnalyzer, AnalysisResult, create_analyzer
from .bpe import BPETokenizer
from .trie import SuffixTrie, DictionaryTrie


# ============================================================
# Token types
# ============================================================
TOKEN_WORD = "word"
TOKEN_PUNCT = "punctuation"
TOKEN_NUMBER = "number"
TOKEN_SPACE = "whitespace"
TOKEN_UNKNOWN = "unknown"

# Regex for Uzbek-aware tokenization
_TOKENIZE_PATTERN = re.compile(
    r"""
    (?P<word>[\w'ʻʼ'']+)        |   # Words (including apostrophes)
    (?P<number>\d[\d.,]*\d|\d)    |   # Numbers
    (?P<punct>[^\w\s'ʻʼ'']+)     |   # Punctuation
    (?P<space>\s+)                    # Whitespace
    """,
    re.VERBOSE | re.UNICODE,
)


_RE_CYRILLIC = re.compile("[\u0400-\u04FF]")


@dataclass
class Token:
    """Represents a single token in the output."""
    text: str
    token_type: str                     # 'word', 'punctuation', 'number', 'whitespace'
    morph: Optional[AnalysisResult] = None  # Morphological analysis (if word)
    subtokens: Optional[List[str]] = None   # BPE subtokens (if fallback)
    token_ids: Optional[List[int]] = None   # Numeric IDs for ML
    is_morph: bool = False              # True if morphologically analyzed
    is_bpe: bool = False                # True if BPE was used (fallback)

    @property
    def root(self) -> Optional[str]:
        """Get the root/stem if morphologically analyzed."""
        if self.morph and self.morph.is_found:
            return self.morph.root
        return None

    @property
    def suffixes(self) -> List[dict]:
        """Get suffix chain as list of dicts."""
        if self.morph and self.morph.suffixes:
            return [s.to_dict() for s in self.morph.suffixes]
        return []

    def to_dict(self) -> dict:
        result = {
            "text": self.text,
            "type": self.token_type,
            "is_morph": self.is_morph,
            "is_bpe": self.is_bpe,
        }
        if self.morph:
            result["morph"] = self.morph.to_dict()
        if self.subtokens:
            result["subtokens"] = self.subtokens
        if self.token_ids:
            result["token_ids"] = self.token_ids
        return result

    def __repr__(self) -> str:
        if self.is_morph and self.morph:
            if self.morph.is_found or self.morph.suffixes:
                if self.morph.suffixes:
                    return f"Token('{self.text}' → {self.morph.root} {self.morph.suffix_chain})"
                return f"Token('{self.text}' → {self.morph.root} [bare])"
        if self.is_bpe and self.subtokens:
            return f"Token('{self.text}' → BPE{self.subtokens})"
        return f"Token('{self.text}', {self.token_type})"

    def to_subwords(self, style: str = "hybrid") -> List[str]:
        """Convert token into subwords with boundary markers.

        Styles:
        - 'hybrid': Ġroot, ##suffix
        - 'bert': root, ##suffix
        - 'gpt': Ġroot, suffix
        - 'sentencepiece':  root, suffix
        - 'plus': root, +suffix
        """
        if self.token_type != TOKEN_WORD:
            return [self.text]

        if style == "hybrid":
            word_prefix, cont_prefix = "Ġ", "##"
        elif style == "bert":
            word_prefix, cont_prefix = "", "##"
        elif style == "gpt":
            word_prefix, cont_prefix = "Ġ", ""
        elif style == "sentencepiece":
            word_prefix, cont_prefix = " ", ""
        elif style == "plus":
            word_prefix, cont_prefix = "", "+"
        else:
            word_prefix, cont_prefix = "", ""

        # Case A: Partial BPE + Morphology (OOV Root with known suffixes)
        if self.is_morph and self.is_bpe and self.subtokens and self.morph:
            # We assume BPE correctly preserved case in subtokens, so just append suffixes
            parts = [f"{word_prefix}{self.subtokens[0]}"]
            for sub in self.subtokens[1:]:
                parts.append(f"{cont_prefix}{sub}")
            for s in self.morph.suffixes:
                parts.append(f"{cont_prefix}{s.text}")
            return parts

        # Case B: Pure Morphology
        if self.is_morph and self.morph:
            # We use it if is_found is True OR if we have suffixes (OOV fallback)
            if self.morph.is_found or self.morph.suffixes:
                root = self.morph.root
                # Restore case
                if self.text.istitle():
                    root = root.capitalize()
                elif self.text.isupper():
                    root = root.upper()
                    
                parts = [f"{word_prefix}{root}"]
                for s in self.morph.suffixes:
                    parts.append(f"{cont_prefix}{s.text}")
                return parts

        # Case C: Pure BPE
        if self.is_bpe and self.subtokens:
            parts = [f"{word_prefix}{self.subtokens[0]}"]
            for sub in self.subtokens[1:]:
                parts.append(f"{cont_prefix}{sub}")
            return parts

        # Case D: Bare/Unknown word
        return [f"{word_prefix}{self.text}"]


class UzTokenizer:
    """
    Hybrid Uzbek Tokenizer.

    Combines rule-based morphological analysis with BPE subword
    tokenization as fallback.

    Flow:
        1. Text → clean & normalize
        2. Split into raw tokens (words, punctuation, numbers)
        3. For each word token:
           a. Try morphological analysis → if found, return root + suffixes
           b. If not found → use BPE to split into subwords
        4. Return list of Token objects

    Usage:
        >>> tok = UzTokenizer()
        >>> tok.load_dictionary()
        >>> tokens = tok.tokenize("Maktablarimizda o'qiyapmiz")
        >>> for t in tokens:
        ...     print(t)
        Token('maktablarimizda' → maktab +lar(Ko'plik) → +imiz(1-shaxs) → +da(O'rin-payt))
        Token('o'qiyapmiz' → o'qi +yap(Hozirgi zamon) → +miz(1-shaxs ko'pl.))

    Performance:
        - SuffixTrie: O(n) suffix matching per word
        - DictionaryTrie: O(n) dictionary lookup per word
        - LRU cache: repeated words are instant
        - NumPy batch: vectorized operations for large texts
        - Optional C extension: 3-5x faster trie operations
    """

    def __init__(
        self,
        dictionary: Optional[Dictionary] = None,
        bpe: Optional[BPETokenizer] = None,
        max_depth: int = 15,
        use_trie: bool = True,
        cache_size: int = 50000,
    ):
        # Dictionary
        self._dictionary = dictionary or Dictionary()
        self._dict_loaded = dictionary is not None

        # Morphological analyzer
        self._analyzer = MorphAnalyzer(
            dictionary=self._dictionary,
            max_depth=max_depth,
            use_trie=use_trie,
            cache_size=cache_size,
        )

        # BPE fallback
        self._bpe = bpe
        self._bpe_trained = bpe is not None
        # Lug'atda yo'q so'zlar uchun yo'qotishsiz BPE zaxirasi (data/bpe_merges.json; yo'q bo'lsa None)
        self._fb = FallbackBPE.load()

        # Stats
        self._stats = {
            "total_tokens": 0,
            "morph_found": 0,
            "morph_not_found": 0,
            "bpe_used": 0,
            "punct_tokens": 0,
            "num_tokens": 0,
        }

    # ============================================================
    # Dictionary & BPE Setup
    # ============================================================

    def load_dictionary(
        self,
        source: str = "hunspell",
        path: Optional[str] = None,
        extra_stems: Optional[List[str]] = None,
    ) -> "UzTokenizer":
        """
        Load dictionary from various sources.

        Args:
            source: 'hunspell' (online), 'file' (local file), or 'auto'
            path: File path (required for 'file' source)
            extra_stems: Additional stems to add

        Returns:
            self (for chaining)
        """
        if source == "hunspell" or source == "auto":
            try:
                self._dictionary.load_hunspell()
            except Exception:
                pass

        if source == "file" and path:
            self._dictionary.load_from_file(path)

        if extra_stems:
            self._dictionary.add_stems(set(extra_stems))

        # Rebuild analyzer with updated dictionary
        self._analyzer = MorphAnalyzer(
            dictionary=self._dictionary,
            max_depth=self._analyzer.max_depth,
            use_trie=self._analyzer._use_trie,
            cache_size=self._analyzer._cache_size,
        )

        self._dict_loaded = True
        return self

    def train_bpe(
        self,
        texts: List[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> "UzTokenizer":
        """
        Train BPE tokenizer on provided texts.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum pair frequency for merging

        Returns:
            self (for chaining)
        """
        self._bpe = BPETokenizer(vocab_size=vocab_size)
        self._bpe.train(texts, min_frequency=min_frequency)
        self._bpe_trained = True
        return self

    def load_bpe(self, path: str) -> "UzTokenizer":
        """Load pre-trained BPE model from file."""
        self._bpe = BPETokenizer.from_file(path)
        self._bpe_trained = True
        return self

    def save_bpe(self, path: str) -> None:
        """Save trained BPE model to file."""
        if self._bpe:
            self._bpe.save(path)

    # ============================================================
    # Core Tokenization
    # ============================================================

    def tokenize(
        self,
        text: str,
        clean: bool = True,
        include_spaces: bool = False,
        include_punct: bool = True,
    ) -> List[Token]:
        """
        Tokenize text using hybrid morphological + BPE approach.

        Args:
            text: Input text to tokenize
            clean: If True, clean text before tokenizing (remove HTML, etc.)
            include_spaces: If True, include whitespace tokens
            include_punct: If True, include punctuation tokens

        Returns:
            List of Token objects
        """
        # Step 1: Normalize
        if clean:
            text = clean_text(text)
        text = normalize_apostrophes(text)

        # Step 2: Split into raw tokens
        raw_tokens = self._split_text(text)

        # Step 3: Process each token
        result = []
        for raw_text, raw_type in raw_tokens:
            if raw_type == TOKEN_SPACE:
                if include_spaces:
                    result.append(Token(text=raw_text, token_type=TOKEN_SPACE))
                continue

            if raw_type == TOKEN_PUNCT:
                self._stats["punct_tokens"] += 1
                if include_punct:
                    result.append(Token(text=raw_text, token_type=TOKEN_PUNCT))
                continue

            if raw_type == TOKEN_NUMBER:
                self._stats["num_tokens"] += 1
                result.append(Token(text=raw_text, token_type=TOKEN_NUMBER))
                continue

            # Word token — try morphological analysis first
            self._stats["total_tokens"] += 1
            token = self._process_word(raw_text)
            result.append(token)

        return result

    def tokenize_words(self, text: str) -> List[Token]:
        """Tokenize and return only word tokens (no punctuation/spaces)."""
        return [t for t in self.tokenize(text) if t.token_type == TOKEN_WORD]

    def tokenize_to_strings(
        self,
        text: str,
        style: str = "plus",
        include_punct: bool = True,
    ) -> List[str]:
        """Tokenize and return flat list of string tokens (roots + suffixes + BPE).

        Args:
            text: Input text string.
            style: 'plus' (default), 'hybrid' (Ġroot, ##suffix), 'bert', 'gpt', 'sentencepiece'.
            include_punct: If True, include punctuation tokens.
        """
        result = []
        for token in self.tokenize(text, include_punct=include_punct, include_spaces=False):
            result.extend(token.to_subwords(style=style))
        return result

    def tokenize_with_boundaries(
        self,
        text: str,
        style: str = "hybrid",
        include_punct: bool = True,
    ) -> List[str]:
        """Tokenize with modern LLM subword boundary markers (default: hybrid Ġroot, ##suffix)."""
        return self.tokenize_to_strings(text, style=style, include_punct=include_punct)

    def tokenize_batch(self, texts: List[str]) -> List[List[Token]]:
        """Tokenize multiple texts."""
        return [self.tokenize(t) for t in texts]

    def build_unified_vocab(self) -> Dict[str, int]:
        """Build unified vocabulary mapping all tokens to unique integer IDs.

        Structure:
        - 0..4: Special tokens (<pad>, <unk>, <s>, </s>, <mask>)
        - Common punctuation & symbols
        - Morphological affixes (##lar, ##dan, etc.)
        - Morphological roots (Ġmaktab, Ġkitob, etc.)
        - BPE subwords (if trained)
        """
        key = (
            len(self._dictionary),
            len(self._bpe.vocab) if (self._bpe and self._bpe.vocab) else 0,
            len(self._fb.pieces) if self._fb else 0,
        )
        cached = getattr(self, "_vocab_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "<mask>": 4,
            # encode() yordamchi belgilari: <cap> keyingi so'z bosh harf bilan,
            # <upper> to'liq katta harf bilan, <nsp> oldidagi belgi bilan probelsiz
            # tutashgan so'z, <w> lug'atda yo'q (harfma-harf kodlangan) so'z boshi.
            "<cap>": 5,
            "<upper>": 6,
            "<nsp>": 7,
            "<w>": 8,
        }
        curr_id = len(vocab)

        # Common punctuation & numbers
        for char in ".,!?:;-\"'/\\()[]{}0123456789":
            if char not in vocab:
                vocab[char] = curr_id
                curr_id += 1

        # Full Latin alphabet (upper + lower) and Uzbek apostrophe variants,
        # so every character has an ID and encode/decode round-trips losslessly.
        import string as _string
        for char in _string.ascii_letters + "ʻʼ‘’":
            if char not in vocab:
                vocab[char] = curr_id
                curr_id += 1

        # Affixes (with ##)
        from .affixes import get_suffix_variants
        for s in get_suffix_variants():
            affix_tok = f"##{s['variant']}"
            if affix_tok not in vocab:
                vocab[affix_tok] = curr_id
                curr_id += 1

        # Roots (with Ġ)
        for stem in sorted(self._dictionary):
            stem_tok = f"Ġ{stem}"
            if stem_tok not in vocab:
                vocab[stem_tok] = curr_id
                curr_id += 1

        # BPE subword tokens
        if self._bpe and self._bpe.vocab:
            for bpe_tok in sorted(self._bpe.vocab.keys()):
                if bpe_tok not in vocab:
                    vocab[bpe_tok] = curr_id
                    curr_id += 1

        if self._fb is not None:
            # BPE bo'laklari so'z boshida (Ġ) va ichida (##) shaklida; mavjud o'zak/qo'shimcha
            # tokenlari bilan mos kelsa, bir xil ID ishlatiladi.
            for piece in self._fb.pieces:
                for pref in ("\u0120", "##"):
                    t = pref + piece
                    if t not in vocab:
                        vocab[t] = curr_id
                        curr_id += 1

        self._vocab_cache = (key, vocab)
        self._vocab_inv_cache = None
        return vocab

    def save_vocab_json(self, filepath: str) -> Dict[str, int]:
        """Export unified vocab.json for LLMs and deep learning models."""
        vocab = self.build_unified_vocab()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        return vocab

    # ------------------------------------------------------------------------
    # Lossless character-level sub-vocabulary helpers
    # ------------------------------------------------------------------------

    _CHAR_TOKENS: str = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " .,!?;:-()\"'\u02bb\u02bc\u2018\u2019/%\u0120#"
    )

    def _char_vocab(self) -> Dict[str, int]:
        """Small fixed alphabet mapping every encodable character to an ID.

        IDs start at 1_000_000 so they never collide with morphological/BPE
        IDs of the unified vocabulary (~72k). ``decode`` recognizes this
        range and reconstructs the original surface text exactly
        (lossless round-trip).
        """
        cv = getattr(self, "_char_vocab_cache", None)
        if cv is None:
            cv = {ch: 1_000_000 + i for i, ch in enumerate(self._CHAR_TOKENS)}
            self._char_vocab_cache = cv
        return cv

    _BYTE_BASE = 2_000_000

    def _encode_charstring(self, s: str) -> List[int]:
        """Harfma-harf kodlash. Belgi _CHAR_TOKENS da bo'lmasa (kirill, emoji, ...)
        UTF-8 baytlari ID = 2_000_000 + bayt sifatida kodlanadi (yo'qotishsiz)."""
        cv = self._char_vocab()
        ids: List[int] = []
        for ch in s:
            cid = cv.get(ch)
            if cid is not None:
                ids.append(cid)
            else:
                ids.extend(self._BYTE_BASE + b for b in ch.encode("utf-8"))
        return ids

    @staticmethod
    def _case_flag(word: str) -> Optional[str]:
        """'cap' (Toshkent), 'upper' (TOSHKENT), None (toshkent) yoki 'mixed' (iPhone).
        str.istitle() ishlatilmaydi: u 'Oʻzbek' ni noto'g'ri baholaydi (ʻ — kichik harf emas)."""
        letters = [c for c in word if c.isalpha()]
        if not letters or word.islower():
            return None
        if len(letters) > 1 and word.isupper():
            return "upper"
        if word[0].isupper() and word[1:].islower():
            return "cap"
        if len(word) == 1 and word.isupper():
            return "cap"
        return "mixed"

    @staticmethod
    def _lower_subword(sw: str) -> str:
        for pref in ("\u0120", "##"):
            if sw.startswith(pref):
                return pref + sw[len(pref):].lower()
        return sw.lower()

    @staticmethod
    def _join_subwords(subs: List[str]) -> str:
        return "".join(sw[1:] if sw.startswith("\u0120") else sw[2:] if sw.startswith("##") else sw
                       for sw in subs)

    def _fallback_subwords(self, lower: str, vocab: Dict[str, int]) -> Optional[List[str]]:
        """Lug'atda to'liq topilmagan so'z uchun BPE bo'laklari (yo'qotishsiz) yoki None.
        (Faqat noma'lum o'zakni BPE qilib qo'shimchalarni saqlash varianti sinaldi: foyda bermadi.)"""
        fb = self._fb
        if fb is None:
            return None
        pieces = fb.encode_word(lower)
        if not pieces:
            return None
        cand = ["\u0120" + pieces[0]] + ["##" + p for p in pieces[1:]]
        return cand if all(sw in vocab for sw in cand) else None

    def encode(
        self,
        text: str,
        style: str = "hybrid",
        canonical_apostrophes: bool = True,
        transliterate_cyrillic: bool = True,
    ) -> List[int]:
        """Matnni token ID larga aylantiradi (``decode`` bilan qaytariladi).

        Kodlash sxemasi (yo'qotishsiz, kam tokenli):
        - Lug'atdagi so'zlar morfemalarga bo'linadi: ``Ġroot``, ``##suffix``.
        - Bosh harf ``<cap>``, to'liq katta harf ``<upper>`` belgisi bilan beriladi;
          so'zning o'zi kichik harfda qidiriladi (Toshkent, TOSHKENT lug'atga tushadi).
        - Oddiy bitta probel alohida token emas: so'z boshi (``Ġ``) probelni anglatadi.
          Boshqa bo'shliqlar (yangi qator, ikkita probel, ...) va so'z oldidagi
          tutashuv (``(kitob``) ``<nsp>`` / belgi tokenlari bilan saqlanadi.
        - Lug'atda yo'q so'z ``<w>`` + harflar (kerak bo'lsa UTF-8 baytlar) bo'ladi.

        Args:
            style: moslik uchun saqlangan; lug'at doim 'hybrid' (Ġroot, ##suffix).
            canonical_apostrophes: True bo'lsa barcha apostrof variantlari
                (' ’ ‘ ʼ ` ...) bitta kanonik shaklga (ʻ / ʼ) keltiriladi — bir xil
                so'z bir xil tokenlar beradi; ``decode`` kanonik shaklni qaytaradi.
                False bo'lsa apostrofi o'zgargan so'zlar harfma-harf saqlanadi.
            transliterate_cyrillic: True bo'lsa kirill matn avval lotinga o'giriladi
                (``decode`` lotin matnni qaytaradi). False bo'lsa kirill baytlar bilan
                saqlanadi (to'liq yo'qotishsiz, lekin juda qimmat).
        """
        vocab = self.build_unified_vocab()
        CAP, UPPER, NSP, WSTART = vocab["<cap>"], vocab["<upper>"], vocab["<nsp>"], vocab["<w>"]
        if transliterate_cyrillic and _RE_CYRILLIC.search(text):
            text = cyrillic_to_latin(text)
        norm = normalize_apostrophes(text)
        if len(norm) != len(text):  # himoya: pozitsiyalar mos kelmasa, normallashtirmaymiz
            norm = text
        pieces = [(m.lastgroup, m.start(), m.end()) for m in _TOKENIZE_PATTERN.finditer(norm)]
        n = len(pieces)

        def is_word_unit(k: int) -> bool:
            kind, st, en = pieces[k]
            return kind == "word" and not norm[st:en].isdigit()

        ids: List[int] = []
        pos = 0
        for i, (kind, st, en) in enumerate(pieces):
            if st > pos:  # regeks qoplamagan belgilar (himoya)
                ids.extend(self._encode_charstring(text[pos:st]))
            pos = en
            raw = text[st:en]
            if kind == "space":
                if raw == " " and 0 < i < n - 1 and is_word_unit(i + 1):
                    continue  # yagona probel — keyingi so'z boshi (Ġ) anglatadi
                ids.extend(self._encode_charstring(raw))
                continue
            if not is_word_unit(i):
                ids.extend(self._encode_charstring(raw))
                continue
            if i > 0 and pieces[i - 1][0] != "space":
                ids.append(NSP)
            word = norm[st:en]
            if not canonical_apostrophes and raw != word:
                ids.append(WSTART)
                ids.extend(self._encode_charstring(raw))
                continue
            flag = self._case_flag(word)
            if flag != "mixed":
                tok = self._process_word(word.lower())
                subs = [self._lower_subword(x) for x in tok.to_subwords(style="hybrid")]
                # Morfofonologik tiklash (shahrida -> shahar+i+da) matnni o'zgartiradi:
                # bunday so'zlar yo'qotishsizlik uchun harfma-harf kodlanadi.
                exact = self._join_subwords(subs) == word.lower()
                if not (exact and all(sw in vocab for sw in subs)):
                    subs = self._fallback_subwords(word.lower(), vocab)
                    exact = subs is not None
                if exact:
                    if flag == "cap":
                        ids.append(CAP)
                    elif flag == "upper":
                        ids.append(UPPER)
                    ids.extend(vocab[sw] for sw in subs)
                    continue
            ids.append(WSTART)
            ids.extend(self._encode_charstring(word if canonical_apostrophes else raw))
        if pos < len(text):
            ids.extend(self._encode_charstring(text[pos:]))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Token ID larni matnga qaytaradi (``encode`` ning teskarisi)."""
        vocab = self.build_unified_vocab()
        inv = getattr(self, "_vocab_inv_cache", None)
        if inv is None:
            inv = {v: k for k, v in vocab.items()}
            self._vocab_inv_cache = inv
        cv_inv = getattr(self, "_char_inv_cache", None)
        if cv_inv is None:
            cv_inv = {v: k for k, v in self._char_vocab().items()}
            self._char_inv_cache = cv_inv
        CAP, UPPER, NSP, WSTART = vocab["<cap>"], vocab["<upper>"], vocab["<nsp>"], vocab["<w>"]
        skip = {vocab["<pad>"], vocab["<unk>"], vocab["<s>"], vocab["</s>"], vocab["<mask>"]}

        out: List[str] = []
        cur: Optional[List[str]] = None
        flag: Optional[str] = None
        pending: Optional[str] = None
        nsp = False
        bbuf = bytearray()

        def last_char() -> str:
            return out[-1][-1] if out else ""

        def flush_bytes():
            if bbuf:
                out.append(bytes(bbuf).decode("utf-8", errors="replace"))
                bbuf.clear()

        def flush_word():
            nonlocal cur, flag
            if cur is not None:
                w = "".join(cur)
                if flag == "cap":
                    w = w[:1].upper() + w[1:]
                elif flag == "upper":
                    w = w.upper()
                if w:
                    out.append(w)
                cur, flag = None, None

        def start_word():
            nonlocal nsp
            lc = last_char()
            if lc and not lc.isspace() and not nsp:
                out.append(" ")
            nsp = False

        for i in ids:
            if i >= self._BYTE_BASE:
                flush_word()
                bbuf.append(i - self._BYTE_BASE)
                continue
            flush_bytes()
            if i >= 1_000_000:
                ch = cv_inv.get(i)
                if ch is not None:
                    flush_word()
                    out.append(ch)
                continue
            if i in skip:
                continue
            if i == CAP or i == UPPER:
                flush_word()
                pending = "cap" if i == CAP else "upper"
                continue
            if i == NSP:
                flush_word()
                nsp = True
                continue
            if i == WSTART:
                flush_word()
                start_word()
                continue
            tok = inv.get(i)
            if not tok:
                continue
            if tok.startswith("\u0120"):
                flush_word()
                start_word()
                cur, flag, pending = [tok[1:]], pending, None
            elif tok.startswith("##"):
                if cur is not None:
                    cur.append(tok[2:])
                else:
                    out.append(tok[2:])
            else:
                flush_word()
                if len(tok) == 1 and not tok.isalnum():
                    out.append(tok)
                else:
                    start_word()
                    out.append(tok)
        flush_word()
        flush_bytes()
        return "".join(out)

    # ============================================================
    # Analysis Methods
    # ============================================================

    def analyze(self, word: str) -> AnalysisResult:
        """Analyze a single word morphologically."""
        return self._analyzer.analyze(word)

    def analyze_all(self, word: str, max_results: int = 5) -> List[AnalysisResult]:
        """Find all possible analyses of a word (ambiguity)."""
        return self._analyzer.analyze_all(word, max_results)

    def analyze_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Full morphological analysis of a text.
        Returns detailed dict for each token.
        """
        tokens = self.tokenize(text)
        return [t.to_dict() for t in tokens]

    # ============================================================
    # Internal Methods
    # ============================================================

    def _split_text(self, text: str) -> List[Tuple[str, str]]:
        """Split text into (text, type) pairs using regex."""
        tokens = []
        for match in _TOKENIZE_PATTERN.finditer(text):
            if match.group("word"):
                tokens.append((match.group("word"), TOKEN_WORD))
            elif match.group("number"):
                tokens.append((match.group("number"), TOKEN_NUMBER))
            elif match.group("punct"):
                tokens.append((match.group("punct"), TOKEN_PUNCT))
            elif match.group("space"):
                tokens.append((match.group("space"), TOKEN_SPACE))
        return tokens

    def _process_word(self, word: str) -> Token:
        """
        Process a single word: try morphological analysis,
        fall back to BPE if not found.
        """
        # Try morphological analysis
        morph_result = self._analyzer.analyze(word)

        if morph_result.is_found:
            self._stats["morph_found"] += 1
            return Token(
                text=word,
                token_type=TOKEN_WORD,
                morph=morph_result,
                is_morph=True,
            )

        # Morphology failed or it's an OOV root with known suffixes
        self._stats["morph_not_found"] += 1

        if self._bpe and self._bpe_trained:
            self._stats["bpe_used"] += 1
            if not morph_result.is_found and morph_result.suffixes:
                # OOV root but has valid suffixes (e.g. Pythonda -> Python + da)
                root_subtokens = self._bpe.tokenize(morph_result.root)
                # We mock it as a BPE token but with morphological suffixes appended
                return Token(
                    text=word,
                    token_type=TOKEN_WORD,
                    morph=morph_result,
                    subtokens=root_subtokens,
                    is_morph=True, # We did find suffixes
                    is_bpe=True,
                )
            else:
                subtokens = self._bpe.tokenize(word)
                return Token(
                    text=word,
                    token_type=TOKEN_WORD,
                    subtokens=subtokens,
                    is_bpe=True,
                )

        # No BPE available — return as is
        # If we found suffixes but no BPE, just return the morph result anyway
        if not morph_result.is_found and morph_result.suffixes:
            return Token(
                text=word,
                token_type=TOKEN_WORD,
                morph=morph_result,
                is_morph=True,
                is_bpe=False,
            )

        return Token(
            text=word,
            token_type=TOKEN_WORD,
            morph=morph_result,
            is_morph=False,
            is_bpe=False,
        )

    # ============================================================
    # Stats & Info
    # ============================================================

    @property
    def stats(self) -> Dict[str, Any]:
        """Get tokenization statistics."""
        total = self._stats["total_tokens"]
        found = self._stats["morph_found"]
        return {
            **self._stats,
            "morph_accuracy": f"{found / total * 100:.1f}%" if total > 0 else "N/A",
            "dictionary_size": len(self._dictionary),
            "bpe_trained": self._bpe_trained,
            "cache_size": len(self._analyzer._cache),
        }

    def reset_stats(self) -> None:
        """Reset tokenization statistics."""
        for key in self._stats:
            self._stats[key] = 0

    @property
    def dictionary_size(self) -> int:
        return len(self._dictionary)

    @property
    def has_bpe(self) -> bool:
        return self._bpe_trained

    def info(self) -> str:
        """Return human-readable info about the tokenizer."""
        lines = [
            "UzTokenizer - O'zbek tili uchun gibrid tokenizator",
            "=" * 50,
            f"Lug'at:          {len(self._dictionary)} ta o'zak",
            f"BPE:             {'Trained' if self._bpe_trained else 'Not trained'}",
            f"Trie:            {'Enabled' if self._analyzer._use_trie else 'Disabled'}",
            f"Max depth:       {self._analyzer.max_depth}",
            f"Cache:           {len(self._analyzer._cache)} ta so'z",
        ]
        if self._stats["total_tokens"] > 0:
            lines.extend([
                f"",
                f"Statistika:",
                f"  Jami so'zlar:    {self._stats['total_tokens']}",
                f"  Morph topildi:   {self._stats['morph_found']}",
                f"  BPE ishlatildi:  {self._stats['bpe_used']}",
                f"  Aniqlik:         {self.stats['morph_accuracy']}",
            ])
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"UzTokenizer(dict_size={len(self._dictionary)}, "
            f"bpe={'yes' if self._bpe_trained else 'no'})"
        )

    # ============================================================
    # Save/Load
    # ============================================================

    def save(self, directory: str) -> None:
        """
        Save tokenizer state to a directory.
        Saves dictionary, BPE model, and config.
        """
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "version": "0.1.0",
            "max_depth": self._analyzer.max_depth,
            "use_trie": self._analyzer._use_trie,
            "cache_size": self._analyzer._cache_size,
            "has_bpe": self._bpe_trained,
        }
        with open(dirpath / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # Save dictionary
        self._dictionary.save(str(dirpath / "dictionary.txt"))

        # Save BPE
        if self._bpe and self._bpe_trained:
            self._bpe.save(str(dirpath / "bpe_model.json"))

        # Save unified vocab.json for LLMs
        self.save_vocab_json(str(dirpath / "vocab.json"))

    @classmethod
    def load(cls, directory: str) -> "UzTokenizer":
        """Load tokenizer from a saved directory."""
        dirpath = Path(directory)

        # Load config
        with open(dirpath / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # Load dictionary
        dictionary = Dictionary.from_file(str(dirpath / "dictionary.txt"))

        # Load BPE
        bpe = None
        bpe_path = dirpath / "bpe_model.json"
        if bpe_path.exists() and config.get("has_bpe"):
            bpe = BPETokenizer.from_file(str(bpe_path))

        return cls(
            dictionary=dictionary,
            bpe=bpe,
            max_depth=config.get("max_depth", 15),
            use_trie=config.get("use_trie", True),
            cache_size=config.get("cache_size", 50000),
        )


# ============================================================
# Convenience functions
# ============================================================

def create_tokenizer(
    load_dict: bool = True,
    bpe_vocab_size: int = 8000,
    max_depth: int = 15,
) -> UzTokenizer:
    """
    Create a ready-to-use UzTokenizer with sensible defaults.

    Args:
        load_dict: If True, downloads and loads Hunspell dictionary
        bpe_vocab_size: BPE vocabulary size (not trained until train_bpe called)
        max_depth: Maximum morphological analysis depth

    Returns:
        Configured UzTokenizer
    """
    tok = UzTokenizer(max_depth=max_depth)
    if load_dict:
        tok.load_dictionary(source="auto")
    return tok


def quick_tokenize(text: str) -> List[str]:
    """
    Quick one-shot tokenization without creating a persistent tokenizer.
    Uses only the built-in dictionary (no Hunspell download).
    """
    tok = UzTokenizer(max_depth=10)
    return tok.tokenize_to_strings(text)
