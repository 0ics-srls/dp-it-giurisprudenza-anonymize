"""Block 11 RED — gazetteer: extract PERSON party names from the Cassazione heading (100%-reliable
anchor on the known), excluding public bodies / companies, and expand components (surname) for
in-text matches."""
from dp_it_giurisprudenza_anonymize import gazetteer


def test_extract_parties_persons_excludes_enti():
    text = (
        "SVOLGIMENTO DEL PROCESSO. sul ricorso 12345/2020 proposto da "
        "DE LUCA SEBASTIANA, QUALE EREDE DI DE LUCA ALFIO, elettivamente domiciliata in Roma "
        "contro AGENZIA DELLE ENTRATE, in persona del legale rappresentante pro tempore, "
        "- controricorrente -")

    parties = gazetteer.extract_parties(text)

    assert "DE LUCA SEBASTIANA" in parties
    assert "DE LUCA ALFIO" in parties
    assert "AGENZIA DELLE ENTRATE" not in parties


def test_enti_and_companies_excluded():
    text = (
        "proposto da ROSSI MARIO, elettivamente domiciliato, "
        "contro ACME COSTRUZIONI S.P.A. e contro MINISTERO DELL'ECONOMIA E DELLE FINANZE")

    parties = gazetteer.extract_parties(text)

    assert "ROSSI MARIO" in parties
    assert not any("S.P.A" in p for p in parties)
    assert not any("MINISTERO" in p for p in parties)


def test_expand_components_adds_surname():
    expanded = gazetteer.expand_components({"DE LUCA SEBASTIANA"})

    assert "DE LUCA SEBASTIANA" in expanded
    assert "DE LUCA" in expanded   # surname alone, for in-text mentions


def test_no_recognizable_parties_yields_empty_set():
    assert gazetteer.extract_parties("Testo senza intestazione riconoscibile, tutto minuscolo.") == set()


def test_accented_and_apostrophe_names_handled():
    text = "proposto da D'AMICO GIUSEPPE contro COMUNE DI NAPOLI"

    parties = gazetteer.extract_parties(text)

    assert "D'AMICO GIUSEPPE" in parties
    assert not any("COMUNE" in p for p in parties)


def test_birthplace_city_after_nato_a_is_not_a_party():
    # real heading (snpen 2024): the city after "nato a" is a PLACE, only the party is captured
    text = ("sul ricorso proposto da ZAGARIA GUIDO nato a SAN CIPRIANO D'AVERSA il 14/02/1965 "
            "avverso l'ordinanza della CORTE DI APPELLO di NAPOLI")
    parties = gazetteer.extract_parties(text)
    assert parties == {"ZAGARIA GUIDO"}
