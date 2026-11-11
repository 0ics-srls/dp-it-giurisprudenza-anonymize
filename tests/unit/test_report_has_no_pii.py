"""A redaction report holds counts and categories only, never a redacted string (or the text a guardrail matched)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dp_it_giurisprudenza_anonymize.dispatch import AnonDeps, run_anonymize, safe_report

NAME = "Mario Rossi"
CF = "RSSMRA80A01H501U"


def leaky_engine(text: str, _meta: dict[str, Any] | None = None, _config: Any = None) -> tuple[str, dict[str, Any]]:
    """An engine whose report tries to carry the redacted text in every field it can."""
    report = {
        "entities": [{"type": "PERSON", "count": 2, "text": NAME, "surface": NAME}],
        "absoluteCategory": True,
        "gazetteerHits": 3,
        "model": "fake",
        "spans": [{"start": 0, "end": 11, "text": NAME}],
        "debug": f"found {NAME}",
    }
    return text.replace(NAME, "[PERSONA_1]"), report


def test_safe_report_keeps_only_the_whitelisted_counts_and_categories() -> None:
    report = safe_report(leaky_engine(f"{NAME} ricorre", None, None)[1])

    assert report == {"entities": [{"type": "PERSON", "count": 2}], "absoluteCategory": True, "gazetteerHits": 3, "model": "fake"}
    assert NAME not in json.dumps(report)


def test_safe_report_survives_garbage() -> None:
    assert safe_report({}) == {"entities": [], "absoluteCategory": False, "gazetteerHits": 0, "model": ""}
    assert safe_report({"entities": "nonsense", "gazetteerHits": "many"})["entities"] == []
    assert safe_report({"entities": [{"type": 5, "count": "x"}, "text"]})["entities"] == []


def test_cassazione_plan_written_with_a_leaky_engine_carries_no_name_in_any_report(tmp_path: Path) -> None:
    sentence = {
        "source_id": "s1", "kind": "snciv", "anno": 2025, "text_raw_original": f"ricorso di {NAME}", "ocr_dis": f"rigetta {NAME}",
        "source_url": "u", "materia": None, "sezione": None, "ecli": None,
    }
    plan = {"version": 1, "kind": "cassazione", "itemId": "cass:x", "counts": {}, "skipped": {}, "pronunce": [], "massime": [], "norme": [], "citations": [],
            "cassazione": [sentence], "cgt": [], "cgt_skipped": []}
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "cass_x.plan.json").write_text(json.dumps(plan), encoding="utf-8")
    item = {"key": "cass:x|v1", "meta": {"kind": "cassazione", "itemId": "cass:x", "version": "1"}}

    run_anonymize({"items": [item], "dirIn": str(tmp_path / "in"), "dirOut": str(tmp_path / "out")}, AnonDeps(engine_loader=lambda: leaky_engine), lambda *_: None)

    out = json.loads((tmp_path / "out" / "cass_x.plan.json").read_text(encoding="utf-8"))
    record = out["cassazione"][0]
    assert NAME not in json.dumps(record["redaction_report"])
    assert NAME not in record["text_anon"]
    assert NAME not in record["ocr_dis_anon"]
    assert record["redaction_report"]["text_anon"]["entities"] == [{"type": "PERSON", "count": 2}]


def test_cgt_quarantine_and_reports_carry_no_matched_text(tmp_path: Path) -> None:
    record = {
        "guid": "{B}", "tipo": "Sentenza", "data": "2024-12-31", "grado": 2, "luogo": "Campania", "anno": 2024, "estremi": "e", "source_url": "u",
        "text_raw_original": f"Il contribuente {CF} e il sig. {NAME}.",
    }
    plan = {"version": 1, "kind": "cgt", "itemId": "cgt:x", "counts": {}, "skipped": {}, "pronunce": [], "massime": [], "norme": [], "citations": [],
            "cassazione": [], "cgt": [record], "cgt_skipped": []}
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "cgt_x.plan.json").write_text(json.dumps(plan), encoding="utf-8")
    item = {"key": "cgt:x|v1", "meta": {"kind": "cgt", "itemId": "cgt:x", "version": "1"}}

    result = run_anonymize({"items": [item], "dirIn": str(tmp_path / "in"), "dirOut": str(tmp_path / "out")}, AnonDeps(engine_loader=None), lambda *_: None)

    written = (tmp_path / "out" / "cgt_x.plan.json").read_text(encoding="utf-8")
    assert CF not in written
    assert NAME not in written
    assert CF not in json.dumps(result)
    assert NAME not in json.dumps(result)
