"""High-performance Trie (prefix tree) implementation for Uzbek NLP and tokenization.

This module provides data structures tailored for agglutinative morphology in Uzbek:
- TrieNode: Memory-efficient node with __slots__.
- SuffixTrie: Reverse prefix tree for O(n) suffix matching (affix stripping/analysis).
- DictionaryTrie: Fast prefix tree for lexicon lookup, stem checking, and autocompletion.
- Batch operations: NumPy-accelerated batch lookup and batch suffix extraction.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

# Optional NumPy import for batch acceleration
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "TrieNode",
    "SuffixTrie",
    "DictionaryTrie",
    "batch_lookup",
    "batch_find_suffixes",
    "HAS_NUMPY",
]


class TrieNode:
    """A node in a Trie (prefix/suffix tree) data structure.

    Optimized for memory efficiency and cache friendliness using __slots__.

    Attributes:
        children: Dictionary mapping single-character strings to child TrieNode instances.
        is_end: Boolean indicating whether a complete word or suffix ends at this node.
        data: Optional payload dictionary or metadata associated with the entry
              (e.g., grammatical tags, affix class, POS, harmonic constraints).
    """

    __slots__ = ("children", "is_end", "data")

    def __init__(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        is_end: bool = False,
    ) -> None:
        """Initialize a TrieNode.

        Args:
            data: Optional metadata list associated with this node.
            is_end: Boolean indicating whether this node represents the end of an entry.
        """
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = is_end
        self.data: List[Dict[str, Any]] = data if data is not None else []

    def __repr__(self) -> str:
        return (
            f"TrieNode(is_end={self.is_end}, "
            f"children_count={len(self.children)}, "
            f"data={self.data!r})"
        )


class SuffixTrie:
    """A reverse Trie specialized for fast suffix matching in Uzbek morphology.

    Because Uzbek is an agglutinative language, affixes (suffixes) are attached
    to the right side of the root word (stem). By storing suffixes reversed,
    finding all matching suffixes at the end of a word is accomplished by traversing
    the word backwards from its last character.

    This guarantees an O(n) time complexity where n is the length of the query word,
    completely independent of the total number of suffixes m stored in the trie (O(n) vs O(n*m)).
    """

    def __init__(
        self,
        suffixes: Optional[Iterable[Union[Tuple[str, Dict[str, Any]], str]]] = None,
    ) -> None:
        """Initialize the SuffixTrie.

        Args:
            suffixes: Optional iterable of (suffix, data_dict) tuples or suffix strings.
        """
        self.root: TrieNode = TrieNode()
        self._size: int = 0
        if suffixes is not None:
            self.bulk_insert(suffixes)

    def insert(self, suffix: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Insert a suffix into the trie.

        The suffix is stored in reverse order internally so that backwards scanning
        from the end of any word matches valid suffixes efficiently.

        Args:
            suffix: Suffix string to insert (e.g., 'lar', 'imizda', 'dan').
            data: Optional dictionary containing morphological properties of the suffix
                  (e.g., {'type': 'plural', 'pos': 'noun'}). Defaults to empty dict.

        Raises:
            TypeError: If suffix is not a string or data is not a dict.
        """
        if not isinstance(suffix, str):
            raise TypeError(f"Suffix must be a string, got {type(suffix).__name__}")
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            raise TypeError(f"Data must be a dict, got {type(data).__name__}")

        curr = self.root
        for char in reversed(suffix):
            nxt = curr.children.get(char)
            if nxt is None:
                nxt = TrieNode()
                curr.children[char] = nxt
            curr = nxt

        if not curr.is_end:
            curr.is_end = True
            self._size += 1
        curr.data.append(data)

    def find_suffixes(self, word: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Find ALL suffixes of the word that exist in the trie.

        The search scans backwards from the end of the word in O(n) time complexity
        (where n is the word length). Matches are returned sorted by suffix length
        in descending order (longest matching suffix first).

        Args:
            word: The input word to extract suffixes from (e.g., 'kitoblarimizda').

        Returns:
            List of (suffix, data) tuples sorted by suffix length (longest first).
        """
        if not isinstance(word, str) or not word:
            if word == "" and self.root.is_end:
                return [("", self.root.data if self.root.data is not None else {})]
            return []

        results: List[Tuple[str, Dict[str, Any]]] = []
        curr = self.root

        # Scan backwards from the last character
        for i in range(len(word) - 1, -1, -1):
            char = word[i]
            curr = curr.children.get(char)
            if curr is None:
                break
            if curr.is_end:
                matched_suffix = word[i:]
                if curr.data:
                    for item in curr.data:
                        results.append((matched_suffix, item))
                else:
                    results.append((matched_suffix, {}))

        # Check if root represents an empty suffix
        if self.root.is_end:
            if self.root.data:
                for item in self.root.data:
                    results.append(("", item))
            else:
                results.append(("", {}))

        # Backwards scan visits shorter suffixes first.
        # Reversing puts longest matches first in O(k) where k <= n.
        results.reverse()
        return results

    def bulk_insert(
        self,
        items: Iterable[Union[Tuple[str, Dict[str, Any]], str]],
    ) -> int:
        """Bulk insert multiple suffixes into the trie.

        Args:
            items: Iterable of either (suffix, data) tuples or plain suffix strings.

        Returns:
            Count of newly added unique suffixes.
        """
        added = 0
        for item in items:
            if isinstance(item, tuple):
                suffix, data = item
            else:
                suffix, data = item, {}
            before = self._size
            self.insert(suffix, data)
            if self._size > before:
                added += 1
        return added

    def batch_find_suffixes(
        self,
        words: List[str],
    ) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Find suffixes for a batch of words.

        Args:
            words: List of words to analyze.

        Returns:
            List of suffix match lists, one per input word.
        """
        return batch_find_suffixes(self, words)

    def search(self, suffix: str) -> bool:
        """Check whether a suffix exists in the trie.

        Args:
            suffix: Suffix string to search for.

        Returns:
            True if suffix exists, False otherwise.
        """
        if not isinstance(suffix, str):
            return False
        curr = self.root
        for char in reversed(suffix):
            curr = curr.children.get(char)
            if curr is None:
                return False
        return curr.is_end

    def get(self, suffix: str, default: Any = None) -> Any:
        """Get the data dictionary for a suffix, or default if not found.

        Args:
            suffix: Suffix to look up.
            default: Default value if not found.

        Returns:
            The data dictionary or default.
        """
        if not isinstance(suffix, str):
            return default
        curr = self.root
        for char in reversed(suffix):
            curr = curr.children.get(char)
            if curr is None:
                return default
        if curr.is_end:
            return curr.data if curr.data is not None else {}
        return default

    def __contains__(self, suffix: str) -> bool:
        """Support 'in' operator for checking suffix presence."""
        return self.search(suffix)

    def __len__(self) -> int:
        """Return the number of unique suffixes stored in the trie."""
        return self._size

    def __iter__(self) -> Iterator[str]:
        """Yield all stored suffixes in the trie."""
        stack: List[Tuple[TrieNode, List[str]]] = [(self.root, [])]
        while stack:
            node, chars = stack.pop()
            if node.is_end:
                yield "".join(reversed(chars))
            for ch in sorted(node.children.keys(), reverse=True):
                stack.append((node.children[ch], chars + [ch]))

    def items(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield all (suffix, data) pairs stored in the trie."""
        stack: List[Tuple[TrieNode, List[str]]] = [(self.root, [])]
        while stack:
            node, chars = stack.pop()
            if node.is_end:
                suffix = "".join(reversed(chars))
                data = node.data if node.data is not None else {}
                yield (suffix, data)
            for ch in sorted(node.children.keys(), reverse=True):
                stack.append((node.children[ch], chars + [ch]))

    def clear(self) -> None:
        """Clear all entries from the trie."""
        self.root = TrieNode()
        self._size = 0

    def __repr__(self) -> str:
        return f"SuffixTrie(size={self._size})"


class DictionaryTrie:
    """A Trie data structure for fast dictionary/stem lookup and prefix matching.

    Supports exact matching, prefix testing, autocompletion (retrieving all words
    with a given prefix), and bulk insertion for Uzbek vocabularies and stems.
    """

    def __init__(self, words: Optional[Iterable[str]] = None) -> None:
        """Initialize the DictionaryTrie.

        Args:
            words: Optional iterable of words to pre-populate the trie with.
        """
        self.root: TrieNode = TrieNode()
        self._size: int = 0
        if words is not None:
            self.bulk_insert(words)

    def insert(self, word: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add a word to the trie.

        Args:
            word: The word string to insert.
            data: Optional dictionary containing metadata associated with the word
                  (e.g., POS tag, stem frequency, inflection paradigm).

        Raises:
            TypeError: If word is not a string.
        """
        if not isinstance(word, str):
            raise TypeError(f"Word must be a string, got {type(word).__name__}")

        curr = self.root
        for char in word:
            nxt = curr.children.get(char)
            if nxt is None:
                nxt = TrieNode()
                curr.children[char] = nxt
            curr = nxt

        if not curr.is_end:
            curr.is_end = True
            self._size += 1
        if data is not None:
            curr.data = data

    def search(self, word: str) -> bool:
        """Check whether the word exists as an exact match in the trie.

        Args:
            word: Word to search for.

        Returns:
            True if the word exists, False otherwise.
        """
        if not isinstance(word, str):
            return False

        curr = self.root
        for char in word:
            curr = curr.children.get(char)
            if curr is None:
                return False
        return curr.is_end

    def starts_with(self, prefix: str) -> bool:
        """Check whether there is any word in the trie that starts with prefix.

        Args:
            prefix: Prefix string to test.

        Returns:
            True if any word starts with prefix, False otherwise.
        """
        if not isinstance(prefix, str):
            return False
        if not prefix:
            return self._size > 0

        curr = self.root
        for char in prefix:
            curr = curr.children.get(char)
            if curr is None:
                return False
        return True

    def get_words_with_prefix(
        self,
        prefix: str,
        limit: Optional[int] = 100,
    ) -> List[str]:
        """Get all words starting with the given prefix, up to limit.

        Words are returned in deterministic lexicographical order.

        Args:
            prefix: Prefix to match.
            limit: Maximum number of words to return. If None, returns all matches.

        Returns:
            List of words starting with prefix.
        """
        if not isinstance(prefix, str):
            return []
        if limit is not None and limit <= 0:
            return []

        curr = self.root
        for char in prefix:
            curr = curr.children.get(char)
            if curr is None:
                return []

        results: List[str] = []

        # Iterative DFS for recursion-safe and fast traversal
        stack: List[Tuple[TrieNode, List[str]]] = [(curr, list(prefix))]
        while stack:
            node, chars = stack.pop()
            if node.is_end:
                results.append("".join(chars))
                if limit is not None and len(results) >= limit:
                    break

            # Reverse sort so smaller character keys are popped first (A-Z)
            for ch in sorted(node.children.keys(), reverse=True):
                stack.append((node.children[ch], chars + [ch]))

        return results

    def bulk_insert(self, words: Iterable[str]) -> int:
        """Fast batch insertion of multiple words.

        Args:
            words: Iterable of string words to insert.

        Returns:
            The number of newly added unique words.
        """
        added = 0
        root = self.root
        for word in words:
            if not isinstance(word, str):
                continue
            curr = root
            for char in word:
                nxt = curr.children.get(char)
                if nxt is None:
                    nxt = TrieNode()
                    curr.children[char] = nxt
                curr = nxt
            if not curr.is_end:
                curr.is_end = True
                added += 1
        self._size += added
        return added

    def find_prefixes(
        self,
        word: str,
    ) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """Find all valid dictionary words that are prefixes of the input word.

        Useful for stem matching and morphological decomposition of Uzbek words.
        Returned sorted by length descending (longest prefix first).

        Args:
            word: The input word to decompose.

        Returns:
            List of (prefix, data) tuples sorted longest first.
        """
        if not isinstance(word, str) or not word:
            if word == "" and self.root.is_end:
                return [("", self.root.data)]
            return []

        results: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        curr = self.root

        for i, char in enumerate(word):
            curr = curr.children.get(char)
            if curr is None:
                break
            if curr.is_end:
                results.append((word[: i + 1], curr.data))

        results.reverse()
        return results

    def longest_prefix(self, word: str) -> Optional[str]:
        """Find the longest word in the trie that is a prefix of the given word.

        Args:
            word: Input word to check.

        Returns:
            The longest matching prefix string, or None if no prefix matches.
        """
        prefixes = self.find_prefixes(word)
        return prefixes[0][0] if prefixes else None

    def get(self, word: str, default: Any = None) -> Any:
        """Get the metadata associated with a word, or default if not found.

        Args:
            word: The word to look up.
            default: Default value if not found.

        Returns:
            Stored metadata or default.
        """
        if not isinstance(word, str):
            return default
        curr = self.root
        for char in word:
            curr = curr.children.get(char)
            if curr is None:
                return default
        if curr.is_end:
            return curr.data
        return default

    def batch_lookup(self, words: List[str]) -> np.ndarray:
        """Perform fast batch lookup for a list of words using NumPy.

        Args:
            words: List of words to check.

        Returns:
            1D boolean NumPy array indicating whether each word is present.
        """
        return batch_lookup(self, words)

    def __contains__(self, word: str) -> bool:
        """Support 'in' operator for checking word presence."""
        return self.search(word)

    def __len__(self) -> int:
        """Return the number of words stored in the trie."""
        return self._size

    def __iter__(self) -> Iterator[str]:
        """Yield all words stored in the trie in lexicographical order."""
        stack: List[Tuple[TrieNode, List[str]]] = [(self.root, [])]
        while stack:
            node, chars = stack.pop()
            if node.is_end:
                yield "".join(chars)
            for ch in sorted(node.children.keys(), reverse=True):
                stack.append((node.children[ch], chars + [ch]))

    def clear(self) -> None:
        """Clear all entries from the trie."""
        self.root = TrieNode()
        self._size = 0

    def __repr__(self) -> str:
        return f"DictionaryTrie(size={self._size})"


def batch_lookup(trie: DictionaryTrie, words: List[str]) -> np.ndarray:
    """Perform fast batch lookup for a list of words against a DictionaryTrie.

    Uses NumPy to construct an optimized boolean array representing membership.

    Args:
        trie: DictionaryTrie instance to look up words in.
        words: List of word strings to test.

    Returns:
        1D boolean NumPy array where index i is True if words[i] is in trie.

    Raises:
        ImportError: If NumPy is not installed in the environment.
        TypeError: If trie is not a DictionaryTrie.
    """
    if not HAS_NUMPY:
        raise ImportError(
            "NumPy is required for batch_lookup. Please install numpy (e.g. 'pip install numpy')."
        )
    if not isinstance(trie, DictionaryTrie):
        raise TypeError(f"Expected DictionaryTrie, got {type(trie).__name__}")

    n = len(words)
    if n == 0:
        return np.empty(0, dtype=bool)

    search = trie.search
    return np.fromiter((search(w) for w in words), dtype=bool, count=n)


def batch_find_suffixes(
    suffix_trie: SuffixTrie,
    words: List[str],
) -> List[List[Tuple[str, Dict[str, Any]]]]:
    """Perform batch suffix finding for a list of words against a SuffixTrie.

    Args:
        suffix_trie: SuffixTrie instance containing suffix definitions.
        words: List of words to analyze for suffixes.

    Returns:
        List of suffix match lists, where each entry contains (suffix, data) tuples
        sorted by suffix length descending (longest first).

    Raises:
        TypeError: If suffix_trie is not a SuffixTrie.
    """
    if not isinstance(suffix_trie, SuffixTrie):
        raise TypeError(f"Expected SuffixTrie, got {type(suffix_trie).__name__}")

    find = suffix_trie.find_suffixes
    return [find(w) for w in words]
