"""RED tests for the rewritten anonymize_text — gazetteer anchor + DeepMount NER + policy mapping +
span-trim veridicità. Model-level: these run in the Docker gate (real DeepMount) and skip locally when
transformers is absent.
"""
import re

import pytest

pytest.importorskip("transformers")

from dp_it_giurisprudenza_anonymize import anonymize
from dp_it_giurisprudenza_anonymize.policy import DEFAULT_MODEL, AnonConfig

_PERSONA = re.compile(r"\[PERSONA_\d+\]")


def test_court_stays_in_clear():
    text = ("sul ricorso proposto da ROSSI MARIO avverso la sentenza della "
            "CORTE DI APPELLO DI MILANO")
    out, _ = anonymize.anonymize_text(text)
    assert "CORTE DI APPELLO DI MILANO" in out
    assert "ROSSI" not in out


def test_section_heading_stays_in_clear():
    text = "RITENUTO IN FATTO 1. Con ordinanza del 7 dicembre 2023 il Tribunale respingeva."
    out, _ = anonymize.anonymize_text(text)
    assert "RITENUTO IN FATTO" in out


def test_address_becomes_indirizzo_not_person():
    text = "elettivamente domiciliata in VIA XX SETTEMBRE 98 presso lo studio"
    out, _ = anonymize.anonymize_text(text)
    assert "VIA XX SETTEMBRE 98" not in out
    assert "[INDIRIZZO]" in out
    assert not _PERSONA.search(out)


def test_shared_surname_leaves_no_component_in_clear():
    text = "sul ricorso proposto da: MOCARELLI PIERO e MOCARELLI GIOVANNI avverso la sentenza"
    out, _ = anonymize.anonymize_text(text)
    assert "MOCARELLI" not in out
    assert "PIERO" not in out
    assert "GIOVANNI" not in out
    assert len(set(_PERSONA.findall(out))) == 2      # two distinct parties, two tokens


def test_same_person_same_token():
    text = "sul ricorso proposto da ROSSI MARIO avverso la sentenza. Il ROSSI proponeva reclamo."
    out, _ = anonymize.anonymize_text(text)
    assert "ROSSI" not in out
    tokens = _PERSONA.findall(out)
    assert len(tokens) >= 2
    assert len(set(tokens)) == 1                     # same person → same [PERSONA_n]


def test_fiscal_code_becomes_cf():
    text = "il contribuente, codice fiscale MBRGNR76H20F839F, proponeva ricorso"
    out, _ = anonymize.anonymize_text(text)
    assert "MBRGNR76H20F839F" not in out
    assert "[CF]" in out


def test_bank_stays_in_clear():
    text = "sul ricorso proposto da ROSSI MARIO contro BANCO BPM S.P.A."
    out, _ = anonymize.anonymize_text(text)
    assert "BANCO BPM" in out


def test_veridicita_no_entity_text_byte_identical():
    text = "Il ricorso è infondato e va rigettato con condanna alle spese."
    out, _ = anonymize.anonymize_text(text)
    assert out == text


def test_veridicita_trailing_period_kept():
    text = "come riferito dal dott. Campese."
    out, _ = anonymize.anonymize_text(text)
    assert "Campese" not in out
    assert out.endswith(".")


def test_report_shape():
    text = "sul ricorso proposto da ROSSI MARIO avverso la sentenza"
    _, report = anonymize.anonymize_text(text)
    assert report["model"] == DEFAULT_MODEL
    assert report["absoluteCategory"] is False
    assert isinstance(report["gazetteerHits"], int) and report["gazetteerHits"] >= 1
    assert isinstance(report["entities"], list)


def test_family_doc_overblocks_city():
    text = "Il minore risiede a Napoli con la madre."
    out, report = anonymize.anonymize_text(text, meta={"materia": "diritto di famiglia"})
    assert report["absoluteCategory"] is True
    assert "Napoli" not in out
    assert "[LUOGO]" in out


def test_config_dropping_indirizzo_leaves_address_in_clear():
    text = "elettivamente domiciliata in VIA XX SETTEMBRE 98 presso lo studio"
    cfg = AnonConfig.from_params({"anonConfig": {"typeToken": {"INDIRIZZO": None}}})
    out, _ = anonymize.anonymize_text(text, config=cfg)
    assert "VIA XX SETTEMBRE 98" in out


def test_real_zagaria_birthdate_fully_redacted():
    # verbatim from snpen 2024 (snpen2024117953O): DeepMount fragments "14/02/1965" — the date-family
    # merge must redact the WHOLE birthdate, no "/02/1965" residue.
    text = ("sul ricorso n. 29431/2023 R.G. proposto da ZAGARIA GUIDO nato a "
            "SAN CIPRIANO D'AVERSA il 14/02/1965 avverso l'ordinanza del 09/06/2023 della "
            "CORTE DI APPELLO di NAPOLI")
    out, _ = anonymize.anonymize_text(text)
    assert "14/02/1965" not in out
    assert "/02/1965" not in out
    assert "[DATA_NASCITA]" in out


def test_real_excerpt_end_to_end():
    # verbatim from snpen 2024 (snpen2024120255S): person + birthdate redacted, court and city kept
    text = ("sul ricorso proposto da: RAGNIOLO FRANCESCO, nato a Napoli il 05/09/1983 "
            "avverso l'ordinanza del 02/11/2023 della CORTE DI APPELLO DI ROMA")
    out, report = anonymize.anonymize_text(text)
    assert "RAGNIOLO" not in out and "FRANCESCO" not in out
    assert "05/09/1983" not in out and "[DATA_NASCITA]" in out
    assert "CORTE DI APPELLO DI ROMA" in out
    assert "Napoli" in out                            # city kept in clear (non-absolute doc)
    assert report["absoluteCategory"] is False
