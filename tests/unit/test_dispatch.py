"""Dispatch per meta.kind: cc passes through (no model), cgt goes through the PII guardrail (flagged documents are quarantined), cassazione calls the engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dp_it_giurisprudenza_anonymize import dispatch
from dp_it_giurisprudenza_anonymize.dispatch import AnonDeps, AnonError, run_anonymize

CF = "RSSMRA80A01H501U"
CLEAN_TEXT = "M. F. ha presentato appello contro la sentenza della Corte di giustizia tributaria. SVOLGIMENTO DEL PROCESSO. Il ricorso e' infondato."


def plan(kind: str, **arrays: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": 1, "kind": kind, "itemId": f"{kind}:x", "counts": {}, "skipped": {},
        "pronunce": [], "massime": [], "norme": [], "citations": [], "cassazione": [], "cgt": [], "cgt_skipped": [],
    }
    base.update(arrays)
    return base


def sentence(source_id: str, text: str, materia: str | None = "tributario", dis: str | None = "P.Q.M. rigetta") -> dict[str, Any]:
    return {
        "source_id": source_id, "kind": "snciv", "anno": 2025, "num_dec": "1", "sezione": "T", "materia": materia, "ecli": None,
        "text_raw_original": text, "ocr_dis": dis, "source_url": "https://www.italgiure.giustizia.it/sncass/#/ricerca?id=" + source_id,
    }


def cgt_record(guid: str, text: str) -> dict[str, Any]:
    return {
        "guid": guid, "tipo": "Sentenza", "data": "2024-12-31", "grado": 2, "luogo": "Campania", "sezione": "2", "numero": "7355", "anno": 2024,
        "estremi": "Sentenza del 31/12/2024 n. 7355", "text_raw_original": text, "source_url": "https://def.finanze.it/x?id=" + guid,
    }


def item(kind: str, item_id: str) -> dict[str, Any]:
    return {"key": f"{item_id}|v1", "meta": {"kind": kind, "itemId": item_id, "version": "1"}}


def write_plan(directory: Path, item_id: str, content: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (item_id.replace(":", "_") + ".plan.json")).write_text(json.dumps(content), encoding="utf-8")


def out_plan(tmp_path: Path, item_id: str) -> dict[str, Any]:
    return json.loads((tmp_path / "out" / (item_id.replace(":", "_") + ".plan.json")).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def run(tmp_path: Path, deps: AnonDeps, *items: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return run_anonymize({"items": list(items), "dirIn": str(tmp_path / "in"), "dirOut": str(tmp_path / "out"), **extra}, deps, lambda *_: None)


def never_called(*_args: Any, **_kwargs: Any) -> Any:
    msg = "the engine must not be used for this kind"
    raise AssertionError(msg)


def fake_engine(text: str, meta: dict[str, Any] | None = None, config: Any = None) -> tuple[str, dict[str, Any]]:
    seen.append((text, meta, config))
    return text.replace("ROSSI MARIO", "[PERSONA_1]"), {"entities": [{"type": "PERSON", "count": 1}], "absoluteCategory": False, "gazetteerHits": 1, "model": "fake"}


seen: list[tuple[str, dict[str, Any] | None, Any]] = []


def test_cc_passes_through_unchanged_without_loading_the_model(tmp_path: Path) -> None:
    content = plan("cc", pronunce=[{"ecli": "ECLI:IT:COST:2026:95", "anno": 2026, "numero": 95, "tipologia": "O", "data_deposito": "2026-05-04", "dispositivo": "x", "text_raw": "x"}])
    write_plan(tmp_path / "in", "cc:pronunce_2001_oggi", content)

    result = run(tmp_path, AnonDeps(engine_loader=never_called), item("cc", "cc:pronunce_2001_oggi"))

    assert out_plan(tmp_path, "cc:pronunce_2001_oggi") == content
    assert result["metrics"]["passed_through"] == 1
    assert result["metrics"]["anonymized"] == 0
    assert "anonymize" not in dispatch.__dict__  # the dispatch module does not import the engine


def test_cgt_flagged_document_goes_to_quarantine_and_is_not_passed_on(tmp_path: Path) -> None:
    clean, dirty = cgt_record("{AAAA}", CLEAN_TEXT), cgt_record("{BBBB}", f"Il contribuente {CF} proponeva ricorso. Il sig. Mario Rossi.")
    write_plan(tmp_path / "in", "cgt:2024-12-31:2024-12-31", plan("cgt", cgt=[clean, dirty]))

    result = run(tmp_path, AnonDeps(engine_loader=never_called), item("cgt", "cgt:2024-12-31:2024-12-31"))

    out = out_plan(tmp_path, "cgt:2024-12-31:2024-12-31")
    assert [r["guid"] for r in out["cgt"]] == ["{AAAA}"]
    assert out["cgt"][0]["text_anon"] == CLEAN_TEXT
    assert out["cgt"][0]["text_raw_original"] == CLEAN_TEXT
    assert out["cgt"][0]["redaction_report"] == {"sourceAnonymized": True, "findings": []}
    assert out["quarantine"] == [{"id": "{BBBB}", "reason": "pii_guardrail", "findings": {"cf": 1, "nome-parte": 1}}]
    assert CF not in json.dumps(out)  # the flagged text is nowhere in what travels on
    assert result["metrics"]["quarantined"] == 1
    assert result["metrics"]["anonymized"] == 1
    assert result["quarantined"] == [{"key": "cgt:2024-12-31:2024-12-31|v1", "id": "{BBBB}", "reason": "pii_guardrail"}]


def test_cgt_summary_counts_the_scan(tmp_path: Path) -> None:
    write_plan(tmp_path / "in", "cgt:a", plan("cgt", cgt=[cgt_record("{A}", CLEAN_TEXT), cgt_record("{B}", CLEAN_TEXT)]))

    run(tmp_path, AnonDeps(engine_loader=never_called), item("cgt", "cgt:a"))

    assert out_plan(tmp_path, "cgt:a")["anon_summary"] == {"docs": 2, "flagged": 0, "perType": {}}


def test_cassazione_calls_the_engine_with_meta_and_fills_the_anon_fields(tmp_path: Path) -> None:
    seen.clear()
    write_plan(tmp_path / "in", "cass:snciv:2025:01", plan("cassazione", cassazione=[sentence("s1", "sul ricorso proposto da ROSSI MARIO contro X", dis="P.Q.M. ROSSI MARIO"), sentence("s2", "testo senza dispositivo", dis=None)]))

    result = run(tmp_path, AnonDeps(engine_loader=lambda: fake_engine), item("cassazione", "cass:snciv:2025:01"))

    out = out_plan(tmp_path, "cass:snciv:2025:01")
    first, second = out["cassazione"]
    assert first["text_anon"] == "sul ricorso proposto da [PERSONA_1] contro X"
    assert first["ocr_dis_anon"] == "P.Q.M. [PERSONA_1]"
    assert first["text_raw_original"] == "sul ricorso proposto da ROSSI MARIO contro X"  # retained, never served
    assert first["redaction_report"]["text_anon"] == {"entities": [{"type": "PERSON", "count": 1}], "absoluteCategory": False, "gazetteerHits": 1, "model": "fake"}
    assert "ocr_dis_anon" in first["redaction_report"]
    assert second["ocr_dis_anon"] is None
    assert "ocr_dis_anon" not in second["redaction_report"]
    assert seen[0][1] == {"materia": "tributario", "szdec": "T"}
    assert result["metrics"]["anonymized"] == 2
    assert out["counts"]["cassazione"] == 2


def test_anon_config_override_reaches_the_engine(tmp_path: Path) -> None:
    seen.clear()
    write_plan(tmp_path / "in", "cass:a", plan("cassazione", cassazione=[sentence("s1", "testo")]))

    run(tmp_path, AnonDeps(engine_loader=lambda: fake_engine), item("cassazione", "cass:a"), anonConfig={"scoreThreshold": 0.9, "absoluteKw": ["xyz"]})

    config = seen[0][2]
    assert config.score_threshold == pytest.approx(0.9)
    assert config.absolute_kw == ("xyz",)


def test_empty_text_after_trim_is_not_sent_to_the_engine(tmp_path: Path) -> None:
    seen.clear()
    write_plan(tmp_path / "in", "cass:b", plan("cassazione", cassazione=[sentence("s1", "ok", dis="   ")]))

    run(tmp_path, AnonDeps(engine_loader=lambda: fake_engine), item("cassazione", "cass:b"))

    assert [call[0] for call in seen] == ["ok"]
    assert out_plan(tmp_path, "cass:b")["cassazione"][0]["ocr_dis_anon"] is None


def test_one_failing_item_does_not_stop_the_others(tmp_path: Path) -> None:
    write_plan(tmp_path / "in", "cc:ok", plan("cc"))

    with pytest.raises(AnonError) as error:
        run(tmp_path, AnonDeps(engine_loader=never_called), item("cc", "cc:missing"), item("cc", "cc:ok"))

    assert error.value.code == "system_error"
    assert "cc:missing|v1" in str(error.value)
    assert (tmp_path / "out" / "cc_ok.plan.json").exists()


def test_plan_of_another_kind_is_a_failed_item(tmp_path: Path) -> None:
    write_plan(tmp_path / "in", "cc:x", plan("cgt"))

    with pytest.raises(AnonError) as error:
        run(tmp_path, AnonDeps(engine_loader=never_called), item("cc", "cc:x"))

    assert error.value.code == "system_error"


def test_invalid_plan_is_a_failed_item(tmp_path: Path) -> None:
    write_plan(tmp_path / "in", "cc:x", {"kind": "cc"})

    with pytest.raises(AnonError) as error:
        run(tmp_path, AnonDeps(engine_loader=never_called), item("cc", "cc:x"))

    assert error.value.code == "system_error"


def test_engine_failure_is_a_failed_item(tmp_path: Path) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        msg = "model exploded"
        raise RuntimeError(msg)

    write_plan(tmp_path / "in", "cass:z", plan("cassazione", cassazione=[sentence("s1", "testo")]))

    with pytest.raises(AnonError) as error:
        run(tmp_path, AnonDeps(engine_loader=lambda: boom), item("cassazione", "cass:z"))

    assert error.value.code == "system_error"
    assert "model exploded" in str(error.value)


@pytest.mark.parametrize(
    "params",
    [
        {"items": [], "dirIn": "a", "dirOut": "b"},
        {"items": [{"key": "k", "meta": {"kind": "xyz", "itemId": "i"}}], "dirIn": "a", "dirOut": "b"},
        {"items": [item("cc", "cc:x")], "dirOut": "b"},
        {"items": [item("cc", "cc:x")], "dirIn": "a"},
        {"items": [item("cc", "cc:x")], "dirIn": "a", "dirOut": "b", "workers": 0},
        {"items": [item("cc", "cc:x")], "dirIn": "a", "dirOut": "b", "workers": "four"},
        {"items": [item("cc", "cc:x")], "dirIn": "a", "dirOut": "b", "anonConfig": "strict"},
    ],
)
def test_invalid_params_are_config_errors(params: dict[str, Any]) -> None:
    with pytest.raises(AnonError) as error:
        run_anonymize(params, AnonDeps(engine_loader=never_called), lambda *_: None)

    assert error.value.code == "config_error"


def test_workers_param_is_accepted(tmp_path: Path) -> None:
    write_plan(tmp_path / "in", "cc:w", plan("cc"))

    result = run(tmp_path, AnonDeps(engine_loader=never_called), item("cc", "cc:w"), workers=4, anonConfig=None)

    assert result["metrics"]["passed_through"] == 1
