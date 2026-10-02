"""
Dictionary management module for uztoken.

Handles loading, caching, and querying stem/root word dictionaries
from various sources (Hunspell, custom files, online).
"""

import os
import urllib.request
import hashlib
from pathlib import Path
from typing import Set, Optional, List, Union
from functools import lru_cache

try:
    from .normalizer import normalize_uzbek
except ImportError:
    # Fallback for standalone usage
    def normalize_uzbek(text: str) -> str:
        replacements = {
            '\u02bb': "'", '\u02bc': "'", '\u2019': "'", '\u2018': "'",
            '`': "'", '\u02b9': "'", '\u2032': "'", '\u02bf': "'",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.lower().strip()


# Default data directory
_DATA_DIR = Path(__file__).parent / "data"

# Default Hunspell dictionary URL
_HUNSPELL_URL = "https://raw.githubusercontent.com/u2b3k/uz-hunspell/master/uz_UZ.dic"

# Additional well-known stems that Hunspell might miss
_EXTRA_STEMS = {
    # Common borrowed/international words
    "mexanizatsiya", "internatsionalizatsiya", "elektromexanizatsiya",
    "industrializatsiya", "demokratizatsiya", "privatizatsiya",
    "modernizatsiya", "globalizatsiya", "liberalizatsiya",
    "urbanizatsiya", "militarizatsiya", "kolonizatsiya",
    "civilizatsiya", "organizatsiya", "realizatsiya",
    "optimizatsiya", "minimizatsiya", "maksimizatsiya",
    "stabilizatsiya", "mobilizatsiya", "sterilizatsiya",
    "transformatsiya", "informatsiya", "kommunikatsiya",
    "klassifikatsiya", "identifikatsiya", "diversifikatsiya",
    "intensifikatsiya", "ratifikatsiya", "kvalifikatsiya",
    "sertifikatsiya", "modifikatsiya", "kodifikatsiya",
    "elektrifikatsiya", "gazifikatsiya", "kompyuterizatsiya",
    "funksiya", "konstitutsiya", "revolyutsiya", "evolyutsiya",
    "konstruktsiya", "instruktsiya", "operatsiya", "federatsiya",
    "delegatsiya", "deklaratsiya", "immigratsiya", "emigratsiya",
    "navigatsiya", "aviatsiya", "motivatsiya", "innovatsiya",

    # Important words the analyzer needs
    "muvaffaqiyat", "tashkilot", "jamiyat", "huquq", "qaror",
    "munosabat", "taraqqiyot", "iqtisodiyot", "siyosat", "hukumat",
    "ta'lim", "tarbiya", "sog'liq", "xavfsizlik", "mustaqillik",
    "ozodlik", "erkinlik", "adolat", "tinchlik", "barqarorlik",
    "rivojlanish", "o'zgarish", "muammo", "yechim", "natija",
    "sabab", "maqsad", "reja", "loyiha", "dastur", "tizim",
    "usul", "uslub", "tamoyil", "qoida", "tartib", "kun",

    # Common Uzbek verbs (infinitive stems)
    "ishla", "o'qi", "yoz", "ol", "ber", "kel", "ket", "bor",
    "ko'r", "bil", "de", "ayt", "qil", "tush", "chiq", "kir",
    "tur", "o'tir", "yot", "yur", "ye", "ich", "sur", "ur",
    "och", "yop", "sol", "o'rnat", "boshla", "tugat", "davom",
    "et", "bo'l", "qol", "tushun", "angla", "o'rgan", "o'rgat",
    "kechir", "tushir", "chiqar", "kirgiz", "ko'tar", "tushir",
    "ishlat", "o'qit", "yozdir", "oldir", "berdir", "keldir",

    # Common nouns
    "maktab", "universitet", "kitob", "dars", "til", "so'z",
    "odam", "bola", "ota", "ona", "aka", "uka", "opa", "singil",
    "do'st", "dushman", "shahar", "shahir", "qishloq", "mamlakat", "davlat",
    "xalq", "millat", "madaniyat", "fan", "texnologiya",
    "kompyuter", "dastur", "telefon", "internet", "fayl", "kod",
    "ma'lumot", "axborot", "xabar", "yangilik", "maqola", "hikoya",
    "she'r", "qo'shiq", "musiqa", "san'at", "sport", "futbol",
    "uy", "xona", "ko'cha", "yo'l", "daryo", "tog'", "vodiy", "bog'",
    "katak", "stol", "stul", "divan", "shkaf", "deraza", "eshik",
    "yurak", "burun", "o'g'il", "bag'ir", "ko'ngil", "qorin", "bo'yin",
    "og'iz", "o'rtoq", "tayoq", "buloq", "quloq", "oyaq", "oyoq", "tilak",

    # Adjectives
    "katta", "kichik", "yangi", "eski", "yaxshi", "yomon",
    "chiroyli", "xunuk", "baland", "past", "uzun", "qisqa",
    "keng", "tor", "qalin", "yupqa", "og'ir", "yengil",
    "issiq", "sovuq", "qattiq", "yumshoq", "tez", "sekin",
    "to'liq", "bo'sh", "xursand", "g'amgin", "qiziq",

    # Pronouns and function words
    "men", "sen", "u", "biz", "siz", "ular",
    "bu", "shu", "o'sha", "ana", "mana", "qaysi",
    "kim", "nima", "qayer", "qachon", "qanday", "necha",
    "ekan", "emish", "edi", "imoq", "bo'lmoq",

    # Common adverbs
    "hozir", "kecha", "bugun", "ertaga", "har", "doim",
    "hech", "hali", "endi", "keyin", "oldin", "yuqori",
    "pastda", "ichida", "tashqari", "orqali", "bilan",
    "uchun", "haqida", "sababli", "ko'ra", "qadar",
    "butunlay", "xatosiz",

    # Numbers
    "bir", "ikki", "uch", "to'rt", "besh", "olti",
    "yetti", "sakkiz", "to'qqiz", "o'n", "yuz", "ming",
    "million", "milliard", "bitta", "ikkita",

    # Common particles/conjunctions
    "va", "yoki", "lekin", "ammo", "chunki", "agar",
    "garchi", "hamda", "shuningdek", "biroq", "balki",
}


class Dictionary:
    """
    Manages stem/root word dictionary for morphological analysis.

    Supports loading from Hunspell .dic files, plain text files,
    and programmatic additions.

    Usage:
        dict = Dictionary()
        dict.load_hunspell()           # Load from Hunspell online
        dict.add_stems({"yangi", "so'z"})  # Add custom stems
        "maktab" in dict               # Check if stem exists
    """

    def __init__(self, stems: Optional[Set[str]] = None, auto_load_bundled: bool = True):
        self._stems: Set[str] = set()
        if stems:
            self._stems = {normalize_uzbek(s) for s in stems}
        # Always add extra known stems
        self._stems.update(normalize_uzbek(s) for s in _EXTRA_STEMS)

        # Automatically load bundled stems dictionary if available
        if auto_load_bundled:
            default_file = _DATA_DIR / "uz_UZ_stems.txt"
            if default_file.exists():
                try:
                    self.load_from_file(str(default_file))
                except Exception:
                    pass

    def __contains__(self, word: str) -> bool:
        """Check if a word is in the dictionary."""
        return normalize_uzbek(word) in self._stems

    def __len__(self) -> int:
        """Number of stems in dictionary."""
        return len(self._stems)

    def __iter__(self):
        return iter(self._stems)

    @property
    def stems(self) -> Set[str]:
        """Get the full set of stems (read-only copy)."""
        return self._stems.copy()

    def add_stem(self, stem: str) -> None:
        """Add a single stem."""
        self._stems.add(normalize_uzbek(stem))

    def add_stems(self, stems: Union[Set[str], List[str]]) -> None:
        """Add multiple stems at once."""
        self._stems.update(normalize_uzbek(s) for s in stems)

    def remove_stem(self, stem: str) -> None:
        """Remove a stem."""
        self._stems.discard(normalize_uzbek(stem))

    def lookup(self, word: str) -> bool:
        """Check if a word exists in dictionary."""
        return normalize_uzbek(word) in self._stems

    def load_from_file(self, filepath: str, encoding: str = "utf-8") -> int:
        """
        Load stems from a plain text file (one stem per line).

        Returns:
            Number of new stems added.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dictionary file not found: {filepath}")

        count_before = len(self._stems)
        with open(filepath, "r", encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    self._stems.add(normalize_uzbek(line))

        return len(self._stems) - count_before

    def load_hunspell(
        self,
        url: str = _HUNSPELL_URL,
        cache_file: Optional[str] = None,
        force_download: bool = False,
    ) -> int:
        """
        Load stems from a Hunspell .dic file (online or cached).

        Args:
            url: URL of the Hunspell .dic file
            cache_file: Path to cache the downloaded file. If None, uses default.
            force_download: If True, re-downloads even if cached.

        Returns:
            Number of new stems added.
        """
        if cache_file is None:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            cache_file = _DATA_DIR / "uz_UZ_stems.txt"
        else:
            cache_file = Path(cache_file)

        # Try loading from cache first
        if cache_file.exists() and not force_download:
            return self.load_from_file(str(cache_file))

        # Download from URL
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                lines = response.read().decode("utf-8").splitlines()

            count_before = len(self._stems)

            # Hunspell .dic format: first line is count, rest are word/flags
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                # Remove Hunspell flags after /
                stem = line.split("/")[0].strip()
                if stem:
                    self._stems.add(normalize_uzbek(stem))

            # Cache to file
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(self._stems)))

            return len(self._stems) - count_before

        except Exception as e:
            raise ConnectionError(
                f"Failed to download Hunspell dictionary from {url}: {e}"
            ) from e

    def save(self, filepath: str) -> None:
        """Save current dictionary to a file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for stem in sorted(self._stems):
                f.write(stem + "\n")

    @classmethod
    def from_file(cls, filepath: str) -> "Dictionary":
        """Create a Dictionary from a text file."""
        d = cls()
        d.load_from_file(filepath)
        return d

    @classmethod
    def from_hunspell(cls, url: str = _HUNSPELL_URL, cache_file: Optional[str] = None) -> "Dictionary":
        """Create a Dictionary by loading from Hunspell."""
        d = cls()
        d.load_hunspell(url=url, cache_file=cache_file)
        return d

    def get_stats(self) -> dict:
        """Get dictionary statistics."""
        lengths = [len(s) for s in self._stems]
        return {
            "total_stems": len(self._stems),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "with_apostrophe": sum(1 for s in self._stems if "'" in s),
        }
