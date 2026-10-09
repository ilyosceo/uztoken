import pytest

from uztoken.dictionary import Dictionary
from uztoken.morphology import MorphAnalyzer


@pytest.fixture(params=[True, False], ids=["trie", "linear"])
def analyzer(request):
    return MorphAnalyzer(
        Dictionary(auto_load_bundled=False),
        use_trie=request.param,
    )


@pytest.mark.parametrize(
    ("word", "root", "suffixes"),
    [
        ("maktablarimizda", "maktab", ["lar", "imiz", "da"]),
        ("kitoblarimizni", "kitob", ["lar", "imiz", "ni"]),
        ("qishlog'imiz", "qishloq", ["imiz"]),
        ("yuragimda", "yurak", ["im", "da"]),
        ("shahrimiz", "shahar", ["imiz"]),
        ("singlimga", "singil", ["im", "ga"]),
        ("burni", "burun", ["i"]),
    ],
)
def test_analyzes_documented_uzbek_morphology(analyzer, word, root, suffixes):
    result = analyzer.analyze(word)

    assert result.is_found
    assert result.root == root
    assert [suffix.text for suffix in result.suffixes] == suffixes


@pytest.mark.parametrize(
    ("word", "expected_found"),
    [
        ("yurakka", True),
        ("yurakqa", False),
        ("qishloqqa", True),
        ("qishloqka", False),
    ],
)
def test_dative_allomorph_matches_final_k_or_q(analyzer, word, expected_found):
    assert analyzer.analyze(word).is_found is expected_found


def test_analysis_does_not_write_debug_output(analyzer, capsys):
    analyzer.analyze("maktablarimizda")

    assert capsys.readouterr().out == ""
