"""RED tests for spans.py — trim_span narrows a NER span to the entity CORE (veridicità:
trailing punctuation / bled function-words stay OUT of the redacted range). Pure string logic.
"""
from dp_it_giurisprudenza_anonymize.spans import trim_span


def test_trailing_period_excluded():
    text = "Campese."
    assert trim_span(text, 0, 8) == (0, 7)
    assert text[0:7] == "Campese"


def test_bled_trailing_stopword_excluded():
    text = "Napoli che"
    assert trim_span(text, 0, 10) == (0, 6)
    assert text[0:6] == "Napoli"


def test_trailing_comma_excluded():
    text = "Lo Giudice,"
    start, end = trim_span(text, 0, len(text))
    assert text[start:end] == "Lo Giudice"


def test_surrounding_spaces_trimmed():
    text = "  Roma "
    start, end = trim_span(text, 0, len(text))
    assert text[start:end] == "Roma"


def test_clean_multiword_entity_unchanged():
    text = "MOCARELLI GIOVANNI"
    assert trim_span(text, 0, len(text)) == (0, len(text))


def test_all_punctuation_span_returns_none():
    assert trim_span(" .,; ", 0, 5) is None


def test_internal_period_and_apostrophe_preserved():
    text = "via Bisenton. 37."
    start, end = trim_span(text, 0, len(text))
    assert text[start:end] == "via Bisenton. 37"   # only the TRAILING period trimmed
    text2 = "DELL'UTRI MARCELLO,"
    start2, end2 = trim_span(text2, 0, len(text2))
    assert text2[start2:end2] == "DELL'UTRI MARCELLO"


def test_trimmed_slice_has_no_edge_whitespace():
    text = "  imputato ROSSI MARIO ed  "
    start, end = trim_span(text, 0, len(text))
    seg = text[start:end]
    assert seg == seg.strip()
    assert seg == text[start:end]
    assert not seg.endswith("ed")                  # trailing stopword run stripped
