"""encode()/decode() va morfologik tahlil regressiya testlari."""
import pytest
from uztoken import create_tokenizer
from uztoken.normalizer import normalize_apostrophes

@pytest.fixture(scope="module")
def tok():
    return create_tokenizer(load_dict=True)

ROUNDTRIP = [
    "Ular menga bir nechta savol berishni xohlashdi.",
    "Toshkent shahrida (kitob) 2026-yil, TOSHKENT!",
    "Salom\n\ndunyo  ikki probel",
    "Python'da dastur yozyapman 😀 ok",
    "iPhone va McDonald's",
    "  bosh va oxirida bo'shliq  ",
    "shahrida",  # morfofonologik tiklash (shahar) matnni buzmasligi kerak
]

@pytest.mark.parametrize("text", ROUNDTRIP)
def test_roundtrip_is_exact_up_to_apostrophe_normalization(tok, text):
    assert tok.decode(tok.encode(text)) == normalize_apostrophes(text)

def test_apostrophe_variants_give_identical_ids(tok):
    variants = ["O'qituvchi keldi", "Oʻqituvchi keldi", "O‘qituvchi keldi", "O’qituvchi keldi"]
    ids = [tok.encode(v) for v in variants]
    assert all(x == ids[0] for x in ids)

def test_capitalized_word_costs_one_marker_not_characters(tok):
    lower, cap = tok.encode("ular keldi"), tok.encode("Ular keldi")
    assert len(cap) == len(lower) + 1
    assert all(i < 1_000_000 for i in cap)  # harfma-harf fallback yo'q

def test_single_space_is_not_a_token(tok):
    assert all(i < 1_000_000 for i in tok.encode("men kitob oldim"))

def test_cyrillic_is_transliterated_by_default(tok):
    assert tok.decode(tok.encode("Китоб")) == "Kitob"

def test_cyrillic_bytes_roundtrip_when_transliteration_disabled(tok):
    ids = tok.encode("Китоб", transliterate_cyrillic=False)
    assert tok.decode(ids) == "Китоб"

def test_vocab_is_cached(tok):
    assert tok.build_unified_vocab() is tok.build_unified_vocab()

@pytest.mark.parametrize("word,root", [("yurakni", "yurak"), ("yashagan", "yasha")])
def test_word_starting_with_verb_stem_is_not_whole_stem(tok, word, root):
    # Regressiya: _find_verb_base so'zning istalgan boshlang'ich qismini (yu+moq) qabul qilardi.
    assert tok.analyze(word).root == root
