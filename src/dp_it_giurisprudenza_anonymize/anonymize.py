"""Anonymizer (text-in / text-out) for Cassazione sentences. Ties together, in priority order:
  1) the deterministic heading gazetteer (100%-reliable party names) — consistent [PERSONA_n] tokens,
     applied in TWO passes: every FULL party name first (longest first), then surname-alone components,
     so a shared surname never orphans a given name;
  2) the DeepMount domain NER (ner_engine) mapped through the configurable policy (policy.AnonConfig):
     persons → shared [PERSONA_n] map, typed identifiers → fixed tokens, everything else KEPT in clear;
     absolute-category docs (minors/family/personal status) additionally over-block LUOGO/STATO_CIVILE.
Every redacted span is narrowed via spans.trim_span (veridicità: trailing punctuation and bled
function-words stay untouched); replacements apply right-to-left by character offset. Returns
(text_anon, report) with report = {entities, absoluteCategory, gazetteerHits, model}.
"""
import re

from dp_it_giurisprudenza_anonymize import gazetteer
from dp_it_giurisprudenza_anonymize import ner_engine
from dp_it_giurisprudenza_anonymize import policy as policy_mod
from dp_it_giurisprudenza_anonymize.spans import trim_span

_TOKEN_RE = re.compile(r"\[[A-Z_]+(?:_\d+)?\]")


def anonymize_text(text, meta=None, config=None):
    """Redact PII from `text` per `config` (default policy) and return (text_anon, report)."""
    config = config or policy_mod.DEFAULT_CONFIG
    if not text:
        return text, {"entities": [], "absoluteCategory": False, "gazetteerHits": 0,
                      "model": config.model_name}
    absolute = config.is_absolute(meta or {}, text)
    counts = {}
    person_n = 0
    surface_person = {}      # UPPER(surface) -> [PERSONA_n], shared by gazetteer AND ner

    # 1) gazetteer anchor — TWO passes (full names first, then surname components)
    parties = sorted(gazetteer.extract_parties(text), key=len, reverse=True)
    party_tok = {}
    for name in parties:
        key = name.upper()
        if key not in surface_person:
            person_n += 1
            surface_person[key] = f"[{policy_mod.PERSON_TOKEN_PREFIX}_{person_n}]"
        party_tok[name] = surface_person[key]
    gazetteer_hits = 0
    for name in parties:                                                        # PASS 1 full names
        text, k = re.subn(r"\b" + re.escape(name) + r"\b", party_tok[name], text)
        if k:
            gazetteer_hits += k
            counts["PERSON"] = counts.get("PERSON", 0) + k
    comp_tok = {}
    for name in parties:                                                        # longest name wins a
        toks = name.split()                                                     # shared surname
        if len(toks) >= 2:
            comp_tok.setdefault(" ".join(toks[:-1]), party_tok[name])
    for comp in sorted(comp_tok, key=len, reverse=True):                        # PASS 2 surnames
        text, k = re.subn(r"\b" + re.escape(comp) + r"\b", comp_tok[comp], text)
        if k:
            gazetteer_hits += k
            counts["PERSON"] = counts.get("PERSON", 0) + k

    # 2) DeepMount NER on the (gazetteer-anchored) text, policy-mapped, trimmed, applied RTL
    pipe = ner_engine.build_pipeline(config.model_name)
    spans = ner_engine.analyze(pipe, text, config.score_threshold, config.person_labels)
    taken = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    for s in sorted(spans, key=lambda x: x.start, reverse=True):
        tr = trim_span(text, s.start, s.end)
        if tr is None:
            continue
        st, en = tr
        if any(not (en <= a or st >= b) for a, b in taken):
            continue
        surface = text[st:en]
        if _TOKEN_RE.fullmatch(surface):
            continue                                   # already a redaction token (idempotence)
        tok = config.token_for(s.label, absolute)
        if tok is None:
            continue                                   # KEEP in clear
        if tok == "":                                  # person sentinel
            key = surface.upper()
            if key not in surface_person:
                person_n += 1
                surface_person[key] = f"[{policy_mod.PERSON_TOKEN_PREFIX}_{person_n}]"
            tok = surface_person[key]
            counts["PERSON"] = counts.get("PERSON", 0) + 1
        else:
            counts[s.label] = counts.get(s.label, 0) + 1
        taken.append((st, en))
        text = text[:st] + tok + text[en:]

    report = {"entities": [{"type": t, "count": c} for t, c in sorted(counts.items())],
              "absoluteCategory": absolute, "gazetteerHits": gazetteer_hits,
              "model": config.model_name}
    return text, report
