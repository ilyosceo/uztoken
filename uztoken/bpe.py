"""Byte Pair Encoding (BPE) tokenizer for the uztoken library.

This module provides a production-grade Byte Pair Encoding (BPE) subword tokenizer
tailored specifically for the Uzbek language. It functions both as an independent
subword tokenizer and as the fallback mechanism for the morphological analyzer
when encountering out-of-vocabulary (OOV) words, loanwords, or neologisms.

Key Features:
-------------
1. Uzbek Orthographic Awareness:
   - Preserves digraph letters O' (o') and G' (g') during pre-tokenization.
   - Treats tutuq belgisi (apostrophe of separation / glottal stop) inside words
     (e.g., 'ma'no', 'san'at', 'a'lo') as valid word-internal characters.
   - Normalizes all apostrophe variants (ʻ, ʼ, ’, ‘, `, ʹ, etc.) to standard ASCII (').
   - Fully supports both Uzbek Latin and Uzbek Cyrillic alphabets.

2. Algorithmic Efficiency & Optimizations:
   - Frequency-weighted pre-tokenization using `collections.Counter`.
   - Inverted pair-to-word index for O(k) incremental pair frequency updates during training.
   - Rank-based priority merging for fast subword segmentation.
   - Built-in caching (`_token_cache`) for repeated word tokenization.
   - NumPy batch encoding acceleration with automatic padding when NumPy is available.

3. Standard Special Tokens:
   - <pad>: 0 (Padding token)
   - <unk>: 1 (Unknown token)
   - <bos>: 2 (Beginning of sequence)
   - <eos>: 3 (End of sequence)
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# Attempt optional NumPy import for accelerated batch encoding
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Attempt relative import from normalizer if available, otherwise define standalone fallback
try:
    from .normalizer import normalize_apostrophes
except ImportError:
    try:
        from uztoken.normalizer import normalize_apostrophes
    except ImportError:
        _APOSTROPHE_VARIANTS = "ʻʼ’‘`ʹ′ʿ´ʾ‛‚‵＇ˊˋˈʽ\u0092"
        _APOS_TRANS = str.maketrans({c: "'" for c in _APOSTROPHE_VARIANTS})

        def normalize_apostrophes(text: str) -> str:
            """Normalize various Unicode apostrophe characters to ASCII single quote."""
            if not text:
                return ""
            return text.translate(_APOS_TRANS)


# ============================================================================
# Special Token Constants
# ============================================================================

PAD_TOKEN: str = "<pad>"
UNK_TOKEN: str = "<unk>"
BOS_TOKEN: str = "<bos>"
EOS_TOKEN: str = "<eos>"

DEFAULT_SPECIAL_TOKENS: Dict[str, int] = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1,
    BOS_TOKEN: 2,
    EOS_TOKEN: 3,
}


# ============================================================================
# Pre-tokenization Regular Expressions
# ============================================================================

# Matches Unicode letters (excluding digits and underscore)
_LETTER_PATTERN = r"[^\W\d_]"

# Uzbek word pattern:
# Allows sequences of Unicode letters, with apostrophes allowed:
# 1. Between letters (tutuq belgisi, e.g. ma'no, ta'lim, a'lo)
# 2. Immediately following o, O, g, G even at word end (e.g. bog', tog', to'g'ri, o'sha)
_UZBEK_WORD_PATTERN = (
    rf"{_LETTER_PATTERN}+(?:'{_LETTER_PATTERN}+)*(?:(?<=[oOgG])')?"
)

# Combined pre-tokenization pattern:
# Matches Uzbek words, numbers, and individual punctuation / non-whitespace characters
PRE_TOKENIZE_REGEX = re.compile(
    rf"{_UZBEK_WORD_PATTERN}|\d+|[^\s\w]",
    re.UNICODE,
)

# Pattern preserving whitespace chunks for lossless reconstruction
ROUNDTRIP_REGEX = re.compile(
    rf"{_UZBEK_WORD_PATTERN}|\d+|[^\s\w]|\s+",
    re.UNICODE,
)


# ============================================================================
# Helper Functions
# ============================================================================

def _merge_pieces(pieces: Sequence[str], pair: Tuple[str, str], new_token: str) -> List[str]:
    """Merge all adjacent occurrences of `pair` in `pieces` into `new_token`."""
    p0, p1 = pair
    new_pieces: List[str] = []
    i = 0
    n = len(pieces)
    while i < n:
        if i < n - 1 and pieces[i] == p0 and pieces[i + 1] == p1:
            new_pieces.append(new_token)
            i += 2
        else:
            new_pieces.append(pieces[i])
            i += 1
    return new_pieces


# ============================================================================
# BPETokenizer Class
# ============================================================================

class BPETokenizer:
    """Byte Pair Encoding (BPE) subword tokenizer for the Uzbek language.

    This tokenizer learns a vocabulary of subword units by iteratively merging
    the most frequent adjacent pairs of symbols. It includes specialized pre-tokenization
    rules for Uzbek orthography (O', G', and tutuq belgisi), fast inverted-index pair
    frequency tracking, and serialization to/from JSON.

    Attributes:
        vocab_size (int): Target vocabulary size (including special tokens).
        merges (List[Tuple[str, str]]): Ordered list of learned merge pairs.
        vocab (Dict[str, int]): Token string to integer ID mapping.
        inverse_vocab (Dict[int, str]): Integer ID to token string mapping.
    """

    def __init__(self, vocab_size: int = 8000) -> None:
        """Initialize BPETokenizer.

        Args:
            vocab_size: Maximum vocabulary size to learn during training.
                Defaults to 8000.
        """
        if vocab_size < len(DEFAULT_SPECIAL_TOKENS):
            raise ValueError(
                f"vocab_size must be at least {len(DEFAULT_SPECIAL_TOKENS)}, got {vocab_size}"
            )

        self.vocab_size: int = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = dict(DEFAULT_SPECIAL_TOKENS)
        self.inverse_vocab: Dict[int, str] = {
            idx: tok for tok, idx in self.vocab.items()
        }

        # Internal lookups & caches
        self._ranks: Dict[Tuple[str, str], int] = {}
        self._token_cache: Dict[str, List[str]] = {}
        self._token_pattern: re.Pattern = PRE_TOKENIZE_REGEX

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------

    @property
    def pad_token_id(self) -> int:
        """Token ID for padding (<pad>)."""
        return self.vocab.get(PAD_TOKEN, 0)

    @property
    def unk_token_id(self) -> int:
        """Token ID for unknown tokens (<unk>)."""
        return self.vocab.get(UNK_TOKEN, 1)

    @property
    def bos_token_id(self) -> int:
        """Token ID for beginning of sequence (<bos>)."""
        return self.vocab.get(BOS_TOKEN, 2)

    @property
    def eos_token_id(self) -> int:
        """Token ID for end of sequence (<eos>)."""
        return self.vocab.get(EOS_TOKEN, 3)

    # ------------------------------------------------------------------------
    # Pre-tokenization
    # ------------------------------------------------------------------------

    def pre_tokenize(self, text: str, include_whitespace: bool = False) -> List[str]:
        """Pre-tokenize text by splitting on whitespace and punctuation.

        Preserves Uzbek digraphs (o', g') and internal apostrophes (tutuq belgisi).
        All apostrophe variants (ʻ, ʼ, ’, etc.) are normalized to standard ASCII (').

        Args:
            text: Input string.
            include_whitespace: If True, preserves whitespace as separate tokens.

        Returns:
            List of pre-tokenized string segments.

        Examples:
            >>> bpe = BPETokenizer()
            >>> bpe.pre_tokenize("O'zbekiston - go'zal yurt!")
            ["O'zbekiston", '-', "go'zal", 'yurt', '!']
            >>> bpe.pre_tokenize("Ma'no va san'at, 100% bog'da.")
            ["Ma'no", 'va', "san'at", ',', '100', '%', "bog'da", '.']
        """
        if not text:
            return []
        normalized = normalize_apostrophes(text)
        pattern = ROUNDTRIP_REGEX if include_whitespace else self._token_pattern
        return pattern.findall(normalized)

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    def train(self, texts: List[str], min_frequency: int = 2) -> None:
        """Train BPE on a list of texts.

        Algorithm:
        1. Pre-tokenize all texts using Uzbek-aware regex and count word frequencies.
        2. Initialize the vocabulary with special tokens and all unique characters.
        3. Split all words into sequences of characters.
        4. Count all adjacent pairs weighted by word frequencies.
        5. Iteratively merge the most frequent adjacent pair until `vocab_size`
           is reached or the maximum pair frequency drops below `min_frequency`.

        Args:
            texts: List of training text strings.
            min_frequency: Minimum frequency threshold for a pair to be merged.
                Defaults to 2.

        Raises:
            ValueError: If texts is empty or min_frequency < 1.
        """
        if not texts:
            raise ValueError("Training corpus 'texts' cannot be empty.")
        if min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {min_frequency}")

        # Reset vocabulary and merge tables
        self.vocab = dict(DEFAULT_SPECIAL_TOKENS)
        self.inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}
        self.merges = []
        self._ranks = {}
        self._token_cache.clear()

        # Step 1: Count pre-token frequencies
        word_freqs: Counter[str] = Counter()
        for text in texts:
            if not text or not isinstance(text, str):
                continue
            tokens = self.pre_tokenize(text, include_whitespace=False)
            word_freqs.update(tokens)

        if not word_freqs:
            return

        # Step 2: Extract base characters and add to vocabulary
        base_chars: Set[str] = set()
        for word in word_freqs:
            base_chars.update(word)

        # Ensure space is available in base characters
        base_chars.add(" ")

        for char in sorted(base_chars):
            if char not in self.vocab and len(self.vocab) < self.vocab_size:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char

        # Step 3: Split each word into characters
        word_splits: Dict[str, List[str]] = {w: list(w) for w in word_freqs}

        # Step 4: Build initial pair counts & inverted index (pair -> words)
        pair_counts: Counter[Tuple[str, str]] = Counter()
        pair_to_words: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for word, freq in word_freqs.items():
            pieces = word_splits[word]
            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(word)

        # Step 5: Iterative merging with cached frequency updates
        while len(self.vocab) < self.vocab_size and pair_counts:
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break

            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)

            if new_token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[new_token] = new_id
                self.inverse_vocab[new_id] = new_token

            # Identify words containing best_pair
            affected_words = list(pair_to_words[best_pair])
            del pair_counts[best_pair]
            del pair_to_words[best_pair]

            for word in affected_words:
                freq = word_freqs[word]
                old_pieces = word_splits[word]
                new_pieces = _merge_pieces(old_pieces, best_pair, new_token)
                word_splits[word] = new_pieces

                # Compute pair differences for this word
                old_pairs = Counter(
                    (old_pieces[i], old_pieces[i + 1])
                    for i in range(len(old_pieces) - 1)
                )
                new_pairs = Counter(
                    (new_pieces[i], new_pieces[i + 1])
                    for i in range(len(new_pieces) - 1)
                )

                # Decrement counts of destroyed pairs
                for pair, count in old_pairs.items():
                    if pair == best_pair:
                        continue
                    diff = count - new_pairs.get(pair, 0)
                    if diff > 0:
                        pair_counts[pair] -= diff * freq
                        if pair_counts[pair] <= 0:
                            del pair_counts[pair]
                        if pair not in new_pairs:
                            pair_to_words[pair].discard(word)
                            if not pair_to_words[pair]:
                                del pair_to_words[pair]

                # Increment counts of newly created pairs
                for pair, count in new_pairs.items():
                    if pair == best_pair:
                        continue
                    diff = count - old_pairs.get(pair, 0)
                    if diff > 0:
                        pair_counts[pair] += diff * freq
                        pair_to_words[pair].add(word)

        # Build rank lookup table for fast inference
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}

    # ------------------------------------------------------------------------
    # Subword Tokenization
    # ------------------------------------------------------------------------

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply learned BPE merges to a single word string using rank lookup."""
        if not word:
            return []
        if word in self._token_cache:
            return self._token_cache[word]
        if len(word) <= 1:
            return [word]

        pieces = list(word)

        while len(pieces) > 1:
            # Find the candidate pair with the lowest rank (earliest merge)
            min_rank = float("inf")
            best_pair: Optional[Tuple[str, str]] = None

            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                rank = self._ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            if best_pair is None or min_rank == float("inf"):
                break

            pieces = _merge_pieces(pieces, best_pair, best_pair[0] + best_pair[1])

        self._token_cache[word] = pieces
        return pieces

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword strings.

        Splits text into words/punctuation and segments each token into
        the longest subwords learned during BPE training.

        Args:
            text: Input text string to tokenize.

        Returns:
            List of subword strings.

        Examples:
            >>> bpe.tokenize("maktablarimizda")
            ['maktab', 'lar', 'imiz', 'da']
            >>> bpe.tokenize("O'zbekiston - go'zal yurt!")
            ["O'zbekiston", '-', "go'zal", 'yurt', '!']
        """
        if not text:
            return []

        pre_tokens = self.pre_tokenize(text, include_whitespace=False)
        subwords: List[str] = []
        for token in pre_tokens:
            subwords.extend(self._tokenize_word(token))
        return subwords

    # ------------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs.

        Converts text into subwords and maps each subword to its corresponding
        vocabulary integer ID. Tokens or characters not found in the vocabulary
        are mapped to `<unk>` (ID 1). Whitespace is preserved as space token(s)
        to allow exact text reconstruction in decode.

        Args:
            text: Input text string.
            add_bos: Whether to prepend the `<bos>` token ID (2).
            add_eos: Whether to append the `<eos>` token ID (3).

        Returns:
            List of integer token IDs.

        Examples:
            >>> ids = bpe.encode("Salom dunyo!")
            >>> isinstance(ids, list)
            True
        """
        if not text:
            result: List[int] = []
            if add_bos:
                result.append(self.bos_token_id)
            if add_eos:
                result.append(self.eos_token_id)
            return result

        # Pre-tokenize preserving whitespace chunks for full reconstruction
        chunks = self.pre_tokenize(text, include_whitespace=True)
        unk_id = self.unk_token_id

        ids: List[int] = []
        for chunk in chunks:
            if chunk.isspace():
                # Handle space chunks
                for ch in chunk:
                    ids.append(self.vocab.get(ch, unk_id))
            else:
                for subword in self._tokenize_word(chunk):
                    ids.append(self.vocab.get(subword, unk_id))

        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)

        return ids

    def encode_batch(
        self,
        texts: List[str],
        pad_to_max: bool = True,
        max_length: Optional[int] = None,
        add_bos: bool = False,
        add_eos: bool = False,
        return_numpy: bool = True,
    ) -> Union[List[List[int]], Any]:
        """Encode a batch of texts into token IDs with optional padding.

        Uses NumPy if available and requested, returning a 2D ndarray padded
        with the `<pad>` token ID (0).

        Args:
            texts: List of text strings to encode.
            pad_to_max: If True, pads all sequences to equal length.
            max_length: Optional maximum sequence length (truncates if exceeded).
            add_bos: Whether to prepend `<bos>` token ID.
            add_eos: Whether to append `<eos>` token ID.
            return_numpy: Whether to return a NumPy ndarray if NumPy is installed.

        Returns:
            2D NumPy array if return_numpy and NumPy is available,
            otherwise List[List[int]].
        """
        encoded_list: List[List[int]] = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts
        ]

        if max_length is not None:
            encoded_list = [seq[:max_length] for seq in encoded_list]

        if not pad_to_max:
            return encoded_list

        max_len = 0
        if encoded_list:
            max_len = max(len(seq) for seq in encoded_list)
        if max_length is not None:
            max_len = min(max_len, max_length)

        pad_id = self.pad_token_id
        padded: List[List[int]] = [
            seq + [pad_id] * (max_len - len(seq)) for seq in encoded_list
        ]

        if HAS_NUMPY and return_numpy:
            return np.array(padded, dtype=np.int64)

        return padded

    # ------------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------------

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of integer token IDs.
            skip_special_tokens: If True, omits special tokens (<pad>, <unk>,
                <bos>, <eos>) from the decoded output. Defaults to True.

        Returns:
            Reconstructed string text.

        Examples:
            >>> text = "Salom, dunyo! O'zbekiston kelajagi buyukdir."
            >>> ids = bpe.encode(text)
            >>> bpe.decode(ids)
            "Salom, dunyo! O'zbekiston kelajagi buyukdir."
        """
        if not ids:
            return ""

        special_ids = {
            self.pad_token_id,
            self.unk_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }

        tokens: List[str] = []
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            tok = self.inverse_vocab.get(token_id, UNK_TOKEN)
            tokens.append(tok)

        return "".join(tokens)

    # ------------------------------------------------------------------------
    # Persistence (save / load / from_file)
    # ------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save trained BPE model to file in JSON format.

        The JSON payload contains model metadata, vocabulary mappings,
        and the ordered list of merge operations.

        Args:
            path: Destination file path.

        Raises:
            IOError: If writing to the file fails.
        """
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        data = {
            "version": "1.0",
            "model_type": "bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": {
                PAD_TOKEN: self.pad_token_id,
                UNK_TOKEN: self.unk_token_id,
                BOS_TOKEN: self.bos_token_id,
                EOS_TOKEN: self.eos_token_id,
            },
            "vocab": self.vocab,
            "merges": self.merges,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load trained BPE model from file.

        Restores vocabulary, inverse vocabulary, and merge operations.

        Args:
            path: Path to the JSON model file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file content is invalid JSON or missing required fields.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"BPE model file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse BPE model JSON from {path}: {e}") from e

        if "vocab" not in data or "merges" not in data:
            raise ValueError(
                f"Invalid BPE model format in {path}: must contain 'vocab' and 'merges' keys."
            )

        self.vocab_size = data.get("vocab_size", len(data["vocab"]))
        self.vocab = {str(k): int(v) for k, v in data["vocab"].items()}
        self.inverse_vocab = {int(v): str(k) for k, v in self.vocab.items()}
        self.merges = [tuple(pair) for pair in data["merges"]]
        self._ranks = {tuple(pair): i for i, pair in enumerate(self.merges)}
        self._token_cache.clear()

    @classmethod
    def from_file(cls, path: str) -> BPETokenizer:
        """Create and load a BPETokenizer instance from a saved model file.

        Args:
            path: Path to the saved JSON model file.

        Returns:
            Configured and loaded BPETokenizer instance.
        """
        tokenizer = cls()
        tokenizer.load(path)
        return tokenizer

    # ------------------------------------------------------------------------
    # Magic Methods
    # ------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the current size of the vocabulary."""
        return len(self.vocab)

    def __contains__(self, token: str) -> bool:
        """Check if a token exists in the vocabulary."""
        return token in self.vocab

    def __getitem__(self, item: Union[str, int]) -> Union[int, str]:
        """Access token ID by string or token string by integer ID."""
        if isinstance(item, int):
            if item in self.inverse_vocab:
                return self.inverse_vocab[item]
            raise KeyError(f"Token ID {item} not found in vocabulary.")
        if isinstance(item, str):
            if item in self.vocab:
                return self.vocab[item]
            raise KeyError(f"Token '{item}' not found in vocabulary.")
        raise TypeError(f"Index must be str or int, got {type(item).__name__}")

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"BPETokenizer(vocab_size={len(self.vocab)}/{self.vocab_size}, "
            f"merges={len(self.merges)})"
        )
