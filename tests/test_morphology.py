"""Tahlilchi barqarorligi va soxta nomzodlar regressiya testlari."""
import pytest
from uztoken import create_tokenizer

@pytest.fixture(scope="module")
def tok():
    return create_tokenizer(load_dict=True)

WORDS = ["kishilarga", "aytib", "hayotining", "soladigan", "yurakni", "kitoblarimizdan"]

@pytest.mark.parametrize("word", WORDS)
def test_analysis_is_idempotent(tok, word):
    """Bir xil so'zni qayta tahlil qilish natijani o'zgartirmasligi kerak
    (rad etilgan fe'l asosi keshda "" bo'lib qolib, bo'sh o'zakli nomzod qo'shardi)."""
    sig = lambda: [(c.root, tuple(s.text for s in c.suffixes)) for c in tok._analyzer.analyze_all(word, max_results=8)]
    assert sig() == sig() == sig()

@pytest.mark.parametrize("word", WORDS)
def test_no_empty_root_candidates(tok, word):
    for _ in range(2):
        assert all(c.root for c in tok._analyzer.analyze_all(word, max_results=8))


def test_ranker_is_loaded_and_matches_feature_list(tok):
    # Og'irliklar fayli FEATURE_NAMES bilan mos bo'lmasa CandidateRanker.load() None qaytaradi.
    assert tok._analyzer._ranker is not None

@pytest.mark.parametrize("word", ["juda", "qiyin"])
def test_lexical_words_are_not_split(tok, word):
    assert tok.tokenize_to_strings(word, style="plus") == [word]

@pytest.mark.parametrize("word,root", [("yurakni", "yurak"), ("soladigan", "sol"), ("kitoblarimizdan", "kitob")])
def test_ranker_picks_correct_root(tok, word, root):
    assert tok.analyze(word).root == root
