import pytest
from src.tokenizer import CharTokenizer

def test_round_trip_basic():
    tok = CharTokenizer()
    tok.fit("hello\n", include_specials=True)
    ids = tok.encode("hello\n")
    text = tok.decode(ids)  # by default skips specials (none present)
    assert text == "hello\n"

def test_decode_skips_specials():
    tok = CharTokenizer()
    tok.fit("ab", include_specials=True)
    sample = [tok.pad_id, tok.stoi["a"], tok.unk_id, tok.stoi["b"]]
    assert tok.decode(sample, skip_specials=True) == "ab"
    assert tok.decode(sample, skip_specials=False) == "<PAD>a<UNK>b"

def test_decode_unknown_id_raises():
    tok = CharTokenizer()
    tok.fit("ab", include_specials=True)
    bad = [999999]
    with pytest.raises(KeyError):
        tok.decode(bad)
