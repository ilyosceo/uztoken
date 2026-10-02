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

from .normalizer import normalize_uzbek, normalize_apostrophes, clean_text
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
        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "<mask>": 4,
        }
        curr_id = len(vocab)

        # Common punctuation & numbers
        for char in ".,!?:;-\"'/\\()[]{}0123456789":
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

        return vocab

    def save_vocab_json(self, filepath: str) -> Dict[str, int]:
        """Export unified vocab.json for LLMs and deep learning models."""
        vocab = self.build_unified_vocab()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        return vocab

    def encode(self, text: str, style: str = "hybrid") -> List[int]:
        """
        Encode text to numeric token IDs using unified vocabulary.
        Seamlessly maps morphological subwords and BPE tokens to integers.
        """
        vocab = self.build_unified_vocab()
        subwords = self.tokenize_with_boundaries(text, style=style)
        unk_id = vocab.get("<unk>", 1)
        return [vocab.get(sw, unk_id) for sw in subwords]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to human-readable text."""
        vocab = self.build_unified_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        tokens = [id_to_token.get(i, "<unk>") for i in ids]

        # Reconstruct text by joining boundary markers
        text = ""
        for t in tokens:
            if t in ("<pad>", "<unk>", "<s>", "</s>", "<mask>"):
                continue
            if t.startswith("Ġ"):
                text += (" " if text else "") + t[1:]
            elif t.startswith("##"):
                text += t[2:]
            elif t in ".,!?:;":
                text += t
            else:
                text += (" " if text else "") + t
        return text

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
