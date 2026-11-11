"""Span trimming for veridicità: narrow a NER span to the entity CORE so redaction replaces ONLY the
identifier. Model spans may include trailing punctuation ("Campese.") or bleed a trailing lowercase
function-word into narrative ("Napoli che") — those bytes must stay in the output untouched.
"""
_TRAIL_PUNCT = set(".,;:()[]{}\"'’«»-–— \t\n")
_STOPWORDS = {"che", "di", "del", "della", "delle", "dei", "e", "ed", "in", "il", "la", "lo",
              "le", "gli", "un", "una", "a", "da", "per", "con", "su", "tra", "fra", "al", "alla"}


def trim_span(text, start, end):
    """Return (start, end) narrowed to the entity core, or None when nothing remains."""
    while start < end and text[start] in _TRAIL_PUNCT:
        start += 1
    while end > start and text[end - 1] in _TRAIL_PUNCT:
        end -= 1
    # strip a trailing run of lowercase stopword tokens the model bled into the span
    changed = True
    while changed and end > start:
        changed = False
        seg = text[start:end]
        toks = seg.rsplit(None, 1)
        if len(toks) == 2 and toks[1].strip(".,;:").lower() in _STOPWORDS:
            end = start + len(toks[0])                         # cut back to end of first part
            while end > start and text[end - 1] in _TRAIL_PUNCT:
                end -= 1
            changed = True
    if end <= start:
        return None
    return (start, end)
