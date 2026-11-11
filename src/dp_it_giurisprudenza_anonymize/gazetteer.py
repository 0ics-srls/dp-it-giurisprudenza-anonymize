"""Deterministic party-name gazetteer from the Cassazione heading — the 100%-reliable anchor on the
known. Cassazione headings read "sul ricorso … proposto da COGNOME NOME … contro NOME/ENTE …";
party names are ALL-CAPS. We take the upper-case runs that follow the party triggers, strip the
connective glue ("QUALE EREDE DI", …), and drop public bodies / companies. expand_components adds the
surname alone so in-text mentions (which often use only the surname) are caught too.
"""
import re

# Public bodies / institutions that are never a physical person (prefix match, upper-case).
ENTI = (
    "AGENZIA", "MINISTERO", "COMUNE", "REGIONE", "PROVINCIA", "CITTA", "CITTÀ", "INPS", "INAIL",
    "EQUITALIA", "PROCURATORE", "PROCURA", "PREFETTURA", "QUESTURA", "REPUBBLICA", "STATO",
    "COMMISSIONE", "BANCA", "ISTITUTO", "AZIENDA", "UNIVERSITA", "UNIVERSITÀ", "CAMERA", "CONSIGLIO",
    "AUTORITA", "AUTORITÀ", "FALLIMENTO", "CONDOMINIO", "SOCIETA", "SOCIETÀ", "ENTE",
    "CORTE", "TRIBUNALE", "GIUDICE", "SEZIONE",   # courts in the party window are never a person
)

# Company forms (ragioni sociali): S.P.A. / S.R.L. / S.N.C. / S.A.S. and spacing variants.
COMPANY = re.compile(r"\bS\.?\s?[PRN]\.?\s?[ALC]\.?\b|\bS\.?A\.?S\.?\b", re.IGNORECASE)

# Party-introducing triggers; a window after each holds the party name(s).
TRIGGER = re.compile(r"proposto\s+da|nei\s+confronti\s+di|\bcontro\b", re.IGNORECASE)

# An upper-case run right after a birth/residence clause is a PLACE, not a party
# ("proposto da ZAGARIA GUIDO nato a SAN CIPRIANO D'AVERSA il ..." — the city stays in clear).
PLACE_BEFORE = re.compile(r"(?:nat[oa]\s+(?:a|in)|residente\s+(?:a|in)|domiciliat[oa]\s+(?:a|in))\s*$",
                          re.IGNORECASE)

# An upper-case run (accents, apostrophes, hyphen, ampersand). A '.' is allowed ONLY as an initial
# separator (immediately followed by an upper-case letter, e.g. "S.P.A."), never a sentence period
# before a space — otherwise "LUIGI. Il ..." would bleed the next sentence's capital into the name.
_WORD = r"[A-ZÀ-Ý'](?:[A-ZÀ-Ý'&-]|\.(?=[A-ZÀ-Ý]))*"
UPPER_RUN = re.compile(_WORD + r"(?:\s+" + _WORD + r")*")

# Connective glue that can lead an upper-case run (COGNOME NOME, QUALE EREDE DI COGNOME NOME, …).
LEAD_STOP = {
    "QUALE", "EREDE", "EREDI", "DI", "E", "ED", "IN", "NEI", "NEL", "CONFRONTI", "PROPRIO",
    "NONCHE", "NONCHÉ", "ANCHE", "NELLA", "QUALITA", "QUALITÀ", "PERSONA", "CONTRO",
}

# Procedural role words that are not names.
ROLE = {
    "RICORRENTE", "RICORRENTI", "CONTRORICORRENTE", "CONTRORICORRENTI", "RESISTENTE",
    "INTIMATO", "INTIMATA", "INTIMATI", "RICORSO",
}

WINDOW = 400


def extract_parties(text):
    """Return the set of PERSON party names (ALL-CAPS, e.g. "DE LUCA SEBASTIANA") from the heading."""
    parties = set()
    triggers = list(TRIGGER.finditer(text or ""))
    for i, tr in enumerate(triggers):
        start = tr.end()
        end = triggers[i + 1].start() if i + 1 < len(triggers) else min(len(text), start + WINDOW)
        window = text[start:end]
        for m in UPPER_RUN.finditer(window):
            if PLACE_BEFORE.search(window[:m.start()]):
                continue
            cand = _clean(m.group())
            if cand and _is_person(cand):
                parties.add(cand)
    return parties


def expand_components(names):
    """Add the surname alone (all tokens but the given name) so in-text surname mentions match."""
    out = set(names)
    for n in names:
        toks = n.split()
        if len(toks) >= 2:
            out.add(" ".join(toks[:-1]))
    return out


def _clean(run):
    toks = run.split()
    while toks and toks[0] in LEAD_STOP:
        toks.pop(0)
    while toks and toks[-1] in LEAD_STOP:
        toks.pop()
    return " ".join(toks).strip()


def _is_person(cand):
    toks = cand.split()
    if len(toks) < 2:
        return False                          # a full party is COGNOME NOME (≥2 tokens)
    if len(toks) > 5:
        return False                          # long all-caps phrase → institution, not a person
    if all(t in ROLE for t in toks):
        return False
    if any(cand.startswith(e) for e in ENTI):
        return False
    if COMPANY.search(cand):
        return False
    return True
