"""The CGT PII guardrail (port of CgtPiiDetector) and its stratified sample report (port of CgtAnonSampleVerifier).

The MEF source anonymizes the parties itself ("M. F.", "BCC"): the detector verifies that promise. It finds fiscal codes, VAT numbers and titled
full party names, and does not raise on what is lawful (judges in clear, anonymous initials, amounts and docket numbers).
"""
from __future__ import annotations

from pathlib import Path

from dp_it_giurisprudenza_anonymize.guardrail import Finding, SampleDoc, finding_counts, sample, scan, verify


def _real_text() -> str:
    return (Path(__file__).resolve().parent.parent / "fixtures" / "cgt-7355-textraw.txt").read_text(encoding="utf-8")


def test_planted_fiscal_code_is_found() -> None:
    findings = scan("Il contribuente RSSMRA80A01H501U proponeva ricorso.")

    assert [(f.kind, f.match) for f in findings] == [("cf", "RSSMRA80A01H501U")]


def test_planted_vat_number_is_found() -> None:
    findings = scan("La societa' con P.IVA 01234567890 impugnava l'avviso.")

    assert [(f.kind, f.match) for f in findings] == [("piva", "01234567890")]


def test_titled_party_name_is_found() -> None:
    findings = scan("Ricorso proposto dal sig. Mario Rossi avverso l'accertamento.")

    assert any(f.kind == "nome-parte" and "Mario Rossi" in f.match for f in findings)


def test_dott_mario_rossi_is_found_and_other_titles_too() -> None:
    assert any(f.kind == "nome-parte" for f in scan("assistito dal dott. Mario Rossi"))
    assert any(f.kind == "nome-parte" for f in scan("difeso dall'avv. Giulia Bianchi"))
    assert any(f.kind == "nome-parte" for f in scan("la sig.ra Anna Verdi"))


def test_the_real_cgt_7355_text_is_clean() -> None:
    # parties already anonymous ("M. F.", "BCC"), the judge in clear (lawful), amounts in Italian format (3.574,60 is no VAT number)
    assert scan(_real_text()) == []


def test_judge_names_and_anonymous_initials_are_not_flagged() -> None:
    assert scan("Relatore dott. Rossi. M. F. ha presentato appello contro la BCC.") == []
    assert scan("Presidente Mario Rossi, giudice Anna Bianchi") == []  # no title in front of the full name


def test_amounts_and_docket_numbers_are_not_vat_numbers() -> None:
    assert scan("sentenza n. 4269/12/23, importo di euro 3.574,60, ricorso [001130/2024/GIUDIZIO INTERMEDIO]") == []


def test_ade_protocol_with_11_digits_is_not_a_vat_number() -> None:
    assert scan("la dichiarazione Modello Unico Sc (prot. n. 20105822095 - 0000001 del 27/9/2016) ai fini Ires") == []
    assert scan("la dichiarazione (PROT.N.20105822095) ai fini Ires") == []


def test_document_code_framed_by_letters_is_not_a_vat_number() -> None:
    assert scan("impugnava gli Avvisi nn. M16160007632U e M16170007633U per debenza") == []


def test_name_already_anonymized_by_the_source_is_not_flagged() -> None:
    assert scan("a mezzo del Sig. Vxxxx Mxxxx all'epoca legale rappresentante") == []


def test_eleven_digits_at_the_end_of_a_sentence_are_a_vat_number_but_decimals_are_not() -> None:
    assert [f.kind for f in scan("partita iva 01234567890, come noto")] == ["piva"]
    assert scan("importo 12345678901,50 euro") == []
    assert scan("somma 1,12345678901") == []


def test_finding_counts_never_carry_the_matched_text() -> None:
    counts = finding_counts([Finding("cf", "RSSMRA80A01H501U"), Finding("cf", "BNCGLI85B41F205Z"), Finding("piva", "01234567890")])

    assert counts == {"cf": 2, "piva": 1}


def _universe() -> list[SampleDoc]:
    docs: list[SampleDoc] = []
    for anno in (2022, 2023, 2024, 2025):
        for grado in (1, 2):
            for luogo in ("Campania", "Lombardia", "Sicilia"):
                docs.extend(
                    SampleDoc(f"{anno}-{grado}-{luogo}-{i}", anno, grado, luogo, "Testo pulito: il contribuente M. F. contro l'Agenzia delle Entrate.")
                    for i in range(5)
                )
    return docs


def test_sample_is_stratified_by_year_degree_and_place() -> None:
    chosen = sample(_universe(), target=40)

    assert 30 <= len(chosen) <= 50
    assert {d.anno for d in chosen} == {2022, 2023, 2024, 2025}
    assert {d.grado for d in chosen} == {1, 2}
    assert len({d.luogo for d in chosen}) > 1


def test_sample_is_deterministic_and_small_universes_are_returned_whole() -> None:
    assert sample(_universe(), 40) == sample(_universe(), 40)
    small = _universe()[:5]
    assert sample(small, 40) == small


def test_clean_sample_gives_a_clean_report() -> None:
    report = verify(sample(_universe(), 40))

    assert report.clean is True
    assert report.docs_examined >= 30
    assert report.docs_with_pii == 0


def test_dirty_document_is_counted_per_type_and_per_document() -> None:
    docs = _universe()
    docs.append(SampleDoc("sporco-1", 2024, 2, "Lazio", "Il sig. Mario Rossi, CF RSSMRA80A01H501U, P.IVA 01234567890, ricorreva."))

    report = verify(docs)

    assert report.clean is False
    assert report.docs_with_pii == 1
    assert report.per_type == {"cf": 1, "piva": 1, "nome-parte": 1}
    assert [guid for guid, _ in report.dirty_docs] == ["sporco-1"]


def test_report_records_the_outcome_without_the_text() -> None:
    report = verify(sample(_universe(), 40))

    redaction = report.to_redaction_report()

    assert redaction["sourceAnonymized"] is True
    assert redaction["sampleDocs"] == report.docs_examined
    assert redaction["sampleDocsWithPii"] == 0
    assert "contribuente" not in str(redaction)
    markdown = report.to_markdown()
    assert "PULITO" in markdown
    assert "| cf | 0 |" in markdown
