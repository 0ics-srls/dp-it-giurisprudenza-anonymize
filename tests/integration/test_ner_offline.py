"""The real DeepMount NER, offline, through the whole dispatch. Runs only inside Dockerfile.test, where the model is baked into the image.

No module-level path arithmetic (the test container collects this file from a different layout): everything is resolved lazily in the test body.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

NAMES = ("ROSSI MARIO", "BIANCHI ANNA")
HEADING = (
    "REPUBBLICA ITALIANA IN NOME DEL POPOLO ITALIANO LA CORTE SUPREMA DI CASSAZIONE. Sul ricorso 12345-2020 proposto da ROSSI MARIO, "
    "elettivamente domiciliato in Roma, contro BIANCHI ANNA; - controricorrente - avverso la sentenza n. 100/2019 della Corte di appello di Milano. "
    "SVOLGIMENTO DEL PROCESSO. ROSSI MARIO impugnava l'atto; BIANCHI ANNA resisteva. La Corte rigetta il ricorso."
)


def test_two_party_names_are_redacted_through_the_real_engine(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from dp_it_giurisprudenza_anonymize.dispatch import AnonDeps, run_anonymize  # noqa: PLC0415 - after the importorskip

    record = {
        "source_id": "snciv2025000001O", "kind": "snciv", "anno": 2025, "num_dec": "1", "sezione": "T", "materia": "tributario", "ecli": None,
        "text_raw_original": HEADING, "ocr_dis": "P.Q.M. rigetta il ricorso di ROSSI MARIO.", "source_url": "u",
    }
    plan = {"version": 1, "kind": "cassazione", "itemId": "cass:snciv:2025:01", "counts": {}, "skipped": {}, "pronunce": [], "massime": [], "norme": [],
            "citations": [], "cassazione": [record], "cgt": [], "cgt_skipped": []}
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "cass_snciv_2025_01.plan.json").write_text(json.dumps(plan), encoding="utf-8")
    item = {"key": "cass:snciv:2025:01|v1", "meta": {"kind": "cassazione", "itemId": "cass:snciv:2025:01", "version": "1"}}

    run_anonymize({"items": [item], "dirIn": str(tmp_path / "in"), "dirOut": str(tmp_path / "out")}, AnonDeps(), lambda *_: None)

    out = json.loads((tmp_path / "out" / "cass_snciv_2025_01.plan.json").read_text(encoding="utf-8"))["cassazione"][0]
    for name in NAMES:
        assert name not in out["text_anon"]
        for part in name.split():
            assert part not in out["text_anon"], f"{part} leaked into text_anon"
    assert "[PERSONA_" in out["text_anon"]
    assert "ROSSI" not in out["ocr_dis_anon"]
    assert "CORTE" in out["text_anon"].upper()  # the court stays in clear
    assert out["text_raw_original"] == HEADING  # retained for the posture column
    assert out["redaction_report"]["text_anon"]["model"] == "DeepMount00/Italian_NER_XXL"
    assert "ROSSI" not in json.dumps(out["redaction_report"])


def test_two_workers_give_the_same_result_as_one(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from dp_it_giurisprudenza_anonymize.dispatch import AnonDeps, run_anonymize  # noqa: PLC0415 - after the importorskip

    records = [
        {"source_id": f"s{n}", "kind": "snciv", "anno": 2025, "materia": "tributario", "sezione": "T", "ecli": None, "text_raw_original": HEADING,
         "ocr_dis": None, "source_url": "u"}
        for n in range(3)
    ]
    plan = {"version": 1, "kind": "cassazione", "itemId": "cass:w", "counts": {}, "skipped": {}, "pronunce": [], "massime": [], "norme": [], "citations": [],
            "cassazione": records, "cgt": [], "cgt_skipped": []}
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "cass_w.plan.json").write_text(json.dumps(plan), encoding="utf-8")
    item = {"key": "cass:w|v1", "meta": {"kind": "cassazione", "itemId": "cass:w", "version": "1"}}

    texts = []
    for workers in (1, 2):
        out_dir = tmp_path / f"out{workers}"
        run_anonymize({"items": [item], "dirIn": str(tmp_path / "in"), "dirOut": str(out_dir), "workers": workers}, AnonDeps(), lambda *_: None)
        texts.append([r["text_anon"] for r in json.loads((out_dir / "cass_w.plan.json").read_text(encoding="utf-8"))["cassazione"]])

    assert texts[0] == texts[1]
