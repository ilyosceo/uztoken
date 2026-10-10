"""Lug'atda yo'q so'zlar uchun BPE zaxirasi testlari."""
import pytest
from uztoken import create_tokenizer
from uztoken.fallback_bpe import FallbackBPE, train_merges
from uztoken.normalizer import normalize_apostrophes

@pytest.fixture(scope="module")
def tok():
    return create_tokenizer(load_dict=True)

OOV = ["Neyrontarmoqlarni tokenizatsiyalash qiyin.", "Python'da haybatlanib keldi", "Samarqand va Buxoro"]

@pytest.mark.parametrize("text", OOV)
def test_oov_words_roundtrip_without_char_fallback(tok, text):
    ids = tok.encode(text)
    assert tok.decode(ids) == normalize_apostrophes(text)
    assert all(i < 1_000_000 for i in ids)  # harfma-harf / bayt zaxirasiga tushmadi

def test_oov_costs_far_fewer_tokens_than_characters(tok):
    word = "tokenizatsiyalash"
    assert len(tok.encode(word)) <= len(word) // 2

def test_bpe_rejects_characters_outside_alphabet():
    bpe = FallbackBPE.load()
    assert bpe is not None
    assert bpe.encode_word("кітоб") is None and bpe.encode_word("abc123") is None

def test_train_merges_is_deterministic_and_merges_frequent_pairs():
    words = ["kitob", "kitobim", "kitoblar"] * 3 + ["maktab"]
    a1, m1 = train_merges(words, 10)
    a2, m2 = train_merges(words, 10)
    assert (a1, m1) == (a2, m2) and ("k", "i") in m1[:3]

def test_pieces_concatenate_to_original(tok):
    bpe = FallbackBPE.load()
    for w in ["neyrontarmoqlarni", "samarqand", "oʻzbekiston", "haybatlanib"]:
        assert "".join(bpe.encode_word(w)) == w
