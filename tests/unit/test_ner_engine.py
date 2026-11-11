"""RED tests for ner_engine.py — DeepMount domain NER wrapper. Model-level tests run in the Docker
gate (real DeepMount00/Italian_NER_XXL, baked offline); they skip locally when transformers is absent.
Merge/threshold/person_labels logic is tested with a stub pipe and runs everywhere.
"""
import pytest

from dp_it_giurisprudenza_anonymize import ner_engine
from dp_it_giurisprudenza_anonymize.ner_engine import Span
from dp_it_giurisprudenza_anonymize.policy import DEFAULT_MODEL


# --- real-model tests (Docker gate; skipped where transformers is absent) --------------------

@pytest.fixture(scope="module")
def pipe():
    pytest.importorskip("transformers")
    return ner_engine.build_pipeline(DEFAULT_MODEL)


def test_court_name_gets_no_person_label(pipe):
    text = "avverso la sentenza della CORTE DI APPELLO DI MILANO"
    court_lo, court_hi = text.index("CORTE"), len(text)
    for s in ner_engine.analyze(pipe, text):
        if s.label == "PERSON":
            assert s.end <= court_lo or s.start >= court_hi, \
                f"person span over the court name: {text[s.start:s.end]!r}"


def test_address_yields_indirizzo(pipe):
    text = "domiciliata in VIA XX SETTEMBRE 98"
    spans = ner_engine.analyze(pipe, text)
    hits = [s for s in spans if s.label == "INDIRIZZO"]
    assert hits, f"no INDIRIZZO span: {spans}"
    assert "VIA XX SETTEMBRE" in text[hits[0].start:hits[0].end]


def test_fiscal_code_yields_codice_fiscale(pipe):
    text = "codice fiscale MBRGNR76H20F839F"
    spans = ner_engine.analyze(pipe, text)
    hits = [s for s in spans if s.label == "CODICE_FISCALE"]
    assert hits, f"no CODICE_FISCALE span: {spans}"
    assert "MBRGNR76H20F839F" in text[hits[0].start:hits[0].end]


def test_party_name_yields_merged_person(pipe):
    text = "proposto da ROSSI MARIO"
    spans = ner_engine.analyze(pipe, text)
    persons = [s for s in spans if s.label == "PERSON"]
    assert persons, f"no PERSON span: {spans}"
    covered = " ".join(text[s.start:s.end] for s in persons)
    assert "ROSSI" in covered and "MARIO" in covered


def test_bank_yields_banca_not_person(pipe):
    text = "BANCO BPM S.P.A."
    spans = ner_engine.analyze(pipe, text)
    labels = {s.label for s in spans if "BANCO BPM" in text[s.start:s.end]}
    assert "PERSON" not in labels
    assert "BANCA" in {s.label for s in spans}


def test_offsets_index_the_surface(pipe):
    text = "proposto da FRISINA GIUSEPPE contro BANCO BPM S.P.A., codice fiscale MBRGNR76H20F839F"
    for s in ner_engine.analyze(pipe, text):
        surface = text[s.start:s.end]
        assert surface.strip() == surface and surface, \
            f"span ({s.start},{s.end}) does not index a clean surface: {surface!r}"


# --- real sentence excerpts (verbatim from the 2024 snpen corpus zips) -----------------------

REAL_RAGNIOLO = ("sul ricorso proposto da: RAGNIOLO FRANCESCO, nato a Napoli il 05/09/1983 "
                 "avverso l'ordinanza del 02/11/2023 della CORTE DI APPELLO DI ROMA")

REAL_ZAGARIA = ("sul ricorso n. 29431/2023 R.G. proposto da ZAGARIA GUIDO nato a "
                "SAN CIPRIANO D'AVERSA il 14/02/1965 avverso l'ordinanza del 09/06/2023 della "
                "CORTE DI APPELLO di NAPOLI; udita la relazione svolta dal Presidente FILIPPO CASA")


def test_real_ricorrente_person_and_birthdate(pipe):
    spans = ner_engine.analyze(pipe, REAL_RAGNIOLO)
    persons = [REAL_RAGNIOLO[s.start:s.end] for s in spans if s.label == "PERSON"]
    assert any("RAGNIOLO" in p and "FRANCESCO" in p for p in persons), persons
    dob = [REAL_RAGNIOLO[s.start:s.end] for s in spans if s.label == "DATA_NASCITA"]
    assert "05/09/1983" in " ".join(dob)
    # the birthplace city is LUOGO (kept in clear by policy), never a person
    napoli = [s for s in spans if "Napoli" in REAL_RAGNIOLO[s.start:s.end]]
    assert napoli and all(s.label == "LUOGO" for s in napoli)


def test_real_court_and_city_not_person(pipe):
    for text in (REAL_RAGNIOLO, REAL_ZAGARIA):
        court_lo = text.index("CORTE DI APPELLO")
        court_hi = court_lo + len("CORTE DI APPELLO")
        for s in ner_engine.analyze(pipe, text):
            if s.label == "PERSON":
                assert s.end <= court_lo or s.start >= court_hi, \
                    f"person span over the court: {text[s.start:s.end]!r}"


def test_real_judge_name_merges_whole_words(pipe):
    # "FILIPPO CASA" (the Presidente) must come out as whole-word person span(s) —
    # aggregation_strategy=max fuses sub-word pieces ("FI"+"##LIPPO").
    spans = ner_engine.analyze(pipe, REAL_ZAGARIA)
    persons = [REAL_ZAGARIA[s.start:s.end] for s in spans if s.label == "PERSON"]
    assert any("FILIPPO CASA" in p for p in persons), persons
    assert any("ZAGARIA" in p and "GUIDO" in p for p in persons), persons


# --- pure logic via stub pipe (runs locally, no model) ---------------------------------------

class StubPipe:
    def __init__(self, entities):
        self._entities = entities

    def __call__(self, text):
        return self._entities


def test_hits_below_threshold_excluded():
    text = "ROSSI qualcosa"
    stub = StubPipe([
        {"start": 0, "end": 5, "entity_group": "COGNOME", "score": 0.95},
        {"start": 6, "end": 14, "entity_group": "LUOGO", "score": 0.10},
    ])
    spans = ner_engine.analyze(stub, text, threshold=0.5)
    assert [s.label for s in spans] == ["PERSON"]


def test_adjacent_nome_cognome_merge_into_one_person():
    text = "MARIO ROSSI"
    stub = StubPipe([
        {"start": 0, "end": 5, "entity_group": "NOME", "score": 0.99},
        {"start": 6, "end": 11, "entity_group": "COGNOME", "score": 0.97},
    ])
    spans = ner_engine.analyze(stub, text)
    assert len(spans) == 1
    s = spans[0]
    assert (s.start, s.end, s.label) == (0, 11, "PERSON")
    assert s.score == pytest.approx(0.97)          # min of the merged pair


def test_fragmented_birthdate_merges_into_one_data_nascita_span():
    # real corpus case (snpen2024117953O): '14' + '/02/' + '1965' contiguous fragments
    text = "nato il 14/02/1965 a Napoli"
    stub = StubPipe([
        {"start": 8, "end": 10, "entity_group": "DATA_NASCITA", "score": 0.61},
        {"start": 10, "end": 14, "entity_group": "DATA_MORTE", "score": 0.39},
        {"start": 14, "end": 18, "entity_group": "DATA_NASCITA", "score": 0.39},
    ])
    spans = ner_engine.analyze(stub, text, threshold=0.5)
    assert len(spans) == 1
    s = spans[0]
    assert (text[s.start:s.end], s.label) == ("14/02/1965", "DATA_NASCITA")


def test_merged_date_passes_threshold_via_peak():
    text = "14/02/1965"
    stub = StubPipe([
        {"start": 0, "end": 2, "entity_group": "DATA_NASCITA", "score": 0.61},
        {"start": 2, "end": 6, "entity_group": "DATA_MORTE", "score": 0.39},
        {"start": 6, "end": 10, "entity_group": "DATA_NASCITA", "score": 0.39},
    ])
    spans = ner_engine.analyze(stub, text, threshold=0.5)
    assert [text[s.start:s.end] for s in spans] == ["14/02/1965"]   # nothing dropped, nothing leaked


def test_data_never_merges_with_date_family():
    text = "14 02/11/2023"
    stub = StubPipe([
        {"start": 0, "end": 2, "entity_group": "DATA_NASCITA", "score": 0.9},
        {"start": 3, "end": 13, "entity_group": "DATA", "score": 0.9},
    ])
    spans = ner_engine.analyze(stub, text)
    assert [(s.label, text[s.start:s.end]) for s in spans] == \
        [("DATA_NASCITA", "14"), ("DATA", "02/11/2023")]


def test_date_family_spans_split_by_word_do_not_merge():
    text = "14 circa 1965"
    stub = StubPipe([
        {"start": 0, "end": 2, "entity_group": "DATA_NASCITA", "score": 0.9},
        {"start": 9, "end": 13, "entity_group": "DATA_NASCITA", "score": 0.9},
    ])
    spans = ner_engine.analyze(stub, text)
    assert len(spans) == 2


def test_single_below_threshold_span_still_excluded():
    text = "1965"
    stub = StubPipe([{"start": 0, "end": 4, "entity_group": "DATA_NASCITA", "score": 0.3}])
    assert ner_engine.analyze(stub, text, threshold=0.5) == []


def test_person_labels_override_leaves_label_raw():
    text = "avv. VERDI LUIGI"
    stub = StubPipe([{"start": 5, "end": 16, "entity_group": "AVV_NOTAIO", "score": 0.9}])
    spans = ner_engine.analyze(stub, text, person_labels=frozenset({"NOME", "COGNOME"}))
    assert [s.label for s in spans] == ["AVV_NOTAIO"]
    assert isinstance(spans[0], Span)
