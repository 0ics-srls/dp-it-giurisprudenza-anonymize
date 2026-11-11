"""The ``execute`` action: per item, ``<dirIn>/<stem>.plan.json`` becomes ``<dirOut>/<stem>.plan.json`` with the anonymized text fields (P79-11).

``meta.kind`` decides what happens to the text:

* ``cc``: open data of a public court, copied unchanged; the NER model is never loaded;
* ``cassazione``: the full NER + gazetteer + policy engine (``anonymize.anonymize_text``), ``text_anon`` and ``ocr_dis_anon`` filled, the report holding
  counts and categories only;
* ``cgt``: the source anonymizes the parties itself; the PII guardrail verifies it. A flagged document goes to ``quarantine[]`` and is NOT passed on.

The raw text stays in the plan (``text_raw_original``) for the retained, never-served column. A failing item does not stop the others.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import re
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dp_it_giurisprudenza_anonymize import guardrail
from dp_it_giurisprudenza_anonymize.policy import AnonConfig

CONFIG_ERROR = "config_error"
SYSTEM_ERROR = "system_error"
KINDS = ("cc", "cassazione", "cgt")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_ARRAYS = ("pronunce", "massime", "norme", "citations", "cassazione", "cgt", "cgt_skipped")

Engine = Callable[[str, "dict[str, Any] | None", Any], "tuple[str, dict[str, Any]]"]
EngineLoader = Callable[[], Engine]


class AnonError(Exception):
    """A run-level failure carrying one of the protocol error codes."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AnonDeps:
    """The ports of a run. ``engine_loader`` returns the text engine; None loads the real NER engine on first use (cc and cgt never use it)."""

    engine_loader: EngineLoader | None = None


def load_real_engine() -> Engine:
    from dp_it_giurisprudenza_anonymize import anonymize  # noqa: PLC0415 - the model stack is heavy and cc/cgt items never need it

    return anonymize.anonymize_text


def _config(override: dict[str, Any] | None) -> AnonConfig:
    return AnonConfig.from_params({"anonConfig": override})  # type: ignore[no-untyped-call,no-any-return]


def _clean_entity(entity: Any) -> dict[str, Any] | None:
    if isinstance(entity, dict) and isinstance(entity.get("type"), str) and isinstance(entity.get("count"), int) and not isinstance(entity["count"], bool):
        return {"type": entity["type"], "count": entity["count"]}
    return None


def safe_report(report: dict[str, Any]) -> dict[str, Any]:
    """Whitelist of an engine report: entity types with counts, the absolute-category flag, the gazetteer hit count and the model name. Nothing else
    survives, so no redacted string can reach a plan through a report."""
    raw = report.get("entities")
    entities = [clean for clean in map(_clean_entity, raw) if clean] if isinstance(raw, list) else []
    hits = report.get("gazetteerHits")
    model = report.get("model")
    return {
        "entities": entities,
        "absoluteCategory": report.get("absoluteCategory") is True,
        "gazetteerHits": hits if isinstance(hits, int) and not isinstance(hits, bool) else 0,
        "model": model if isinstance(model, str) else "",
    }


def safe_name(item_id: str) -> str:
    """File stem of an item id (identical to the grab and convert classes)."""
    return _UNSAFE.sub("_", item_id.replace(" ", "-"))


def _validate_item(entry: Any) -> None:
    meta = entry.get("meta") if isinstance(entry, dict) else None
    if not isinstance(entry, dict) or not entry.get("key") or not isinstance(meta, dict) or meta.get("kind") not in KINDS:
        raise AnonError(CONFIG_ERROR, f"item needs a key and meta.kind one of {list(KINDS)}: {entry!r}")
    if not isinstance(meta.get("itemId"), str) or not meta["itemId"]:
        raise AnonError(CONFIG_ERROR, f"item {entry['key']}: meta.itemId is required")


def _validate(params: dict[str, Any]) -> tuple[list[dict[str, Any]], Path, Path, AnonConfig, int]:
    items = params.get("items")
    if not isinstance(items, list) or not items:
        raise AnonError(CONFIG_ERROR, "items must be a non-empty array")
    for entry in items:
        _validate_item(entry)
    for name in ("dirIn", "dirOut"):
        if not isinstance(params.get(name), str) or not params[name]:
            raise AnonError(CONFIG_ERROR, f"{name} is required")
    override = params.get("anonConfig")
    if override is not None and not isinstance(override, dict):
        raise AnonError(CONFIG_ERROR, "anonConfig must be an object")
    workers = params.get("workers", 1)
    if workers is None:
        workers = 1
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise AnonError(CONFIG_ERROR, "workers must be an integer >= 1")
    return items, Path(params["dirIn"]), Path(params["dirOut"]), _config(override), workers


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_bytes(data)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _read_plan(path: Path, kind: str) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("version") != 1 or any(not isinstance(plan.get(name), list) for name in _ARRAYS):
        msg = f"{path.name} is not a plan file v1"
        raise ValueError(msg)
    if plan.get("kind") != kind:
        msg = f"{path.name} holds a {plan.get('kind')!r} plan, the item says {kind!r}"
        raise ValueError(msg)
    return plan


# --- cassazione -------------------------------------------------------------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(override: dict[str, Any] | None) -> None:
    _WORKER["engine"] = load_real_engine()
    _WORKER["config"] = _config(override)


def _worker_record(record: dict[str, Any]) -> dict[str, Any]:
    return _anonymize_record(record, _WORKER["engine"], _WORKER["config"])


def _anonymize_record(record: dict[str, Any], engine: Engine, config: AnonConfig) -> dict[str, Any]:
    meta = {"materia": record.get("materia"), "szdec": record.get("sezione")}
    text_anon, report = engine(record["text_raw_original"], meta, config)
    reports = {"text_anon": safe_report(report)}
    dispositivo = record.get("ocr_dis")
    ocr_dis_anon = None
    if isinstance(dispositivo, str) and dispositivo.strip():
        ocr_dis_anon, dis_report = engine(dispositivo, meta, config)
        reports["ocr_dis_anon"] = safe_report(dis_report)
    return {**record, "text_anon": text_anon, "ocr_dis_anon": ocr_dis_anon, "redaction_report": reports}


def _anonymize_cassazione(plan: dict[str, Any], deps: AnonDeps, config: AnonConfig, override: dict[str, Any] | None, workers: int) -> int:
    records: list[dict[str, Any]] = plan["cassazione"]
    if deps.engine_loader is None and workers > 1 and len(records) > 1:
        context = multiprocessing.get_context("spawn")  # fork after threads exist is a deadlock hazard; spawn children load the model once
        with ProcessPoolExecutor(max_workers=min(workers, len(records)), mp_context=context, initializer=_worker_init, initargs=(override,)) as pool:
            plan["cassazione"] = list(pool.map(_worker_record, records, chunksize=1))
    else:
        engine = (deps.engine_loader or load_real_engine)()
        plan["cassazione"] = [_anonymize_record(record, engine, config) for record in records]
    return len(records)


# --- cgt ---------------------------------------------------------------------------------------------------------------------------------------


def _guard_cgt(plan: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    for record in plan["cgt"]:
        findings = guardrail.scan(record["text_raw_original"])
        if findings:
            counts = guardrail.finding_counts(findings)
            quarantine.append({"id": record["guid"], "reason": "pii_guardrail", "findings": counts})
            for kind, count in counts.items():
                totals[kind] = totals.get(kind, 0) + count
            continue
        kept.append({**record, "text_anon": record["text_raw_original"], "redaction_report": {"sourceAnonymized": True, "findings": []}})
    scanned = len(plan["cgt"])
    plan["cgt"] = kept
    plan["quarantine"] = quarantine
    plan["anon_summary"] = {"docs": scanned, "flagged": len(quarantine), "perType": totals}
    return len(kept), quarantine


# --- driver ------------------------------------------------------------------------------------------------------------------------------------


def _process_item(  # noqa: PLR0913, PLR0917
    entry: dict[str, Any], dir_in: Path, dir_out: Path, deps: AnonDeps, config: AnonConfig, params: dict[str, Any], workers: int,
) -> dict[str, Any]:
    meta = entry["meta"]
    stem = safe_name(meta["itemId"])
    source = dir_in / f"{stem}.plan.json"
    plan = _read_plan(source, meta["kind"])
    outcome: dict[str, Any] = {"anonymized": 0, "passed_through": 0, "quarantined": []}
    if meta["kind"] == "cc":
        _write_atomic(dir_out / source.name, source.read_bytes())
        outcome["passed_through"] = 1
    else:
        if meta["kind"] == "cassazione":
            outcome["anonymized"] = _anonymize_cassazione(plan, deps, config, params.get("anonConfig"), workers)
        else:
            outcome["anonymized"], quarantined = _guard_cgt(plan)
            outcome["quarantined"] = [{"key": entry["key"], "id": q["id"], "reason": q["reason"]} for q in quarantined]
        plan["counts"] = {name: len(plan[name]) for name in _ARRAYS}
        _write_atomic(dir_out / source.name, json.dumps(plan, ensure_ascii=False).encode("utf-8"))
    outcome["file"] = {"key": entry["key"], "plan": str(dir_out / source.name)}
    return outcome


def run_anonymize(params: dict[str, Any], deps: AnonDeps, on_progress: Callable[[int, str], None]) -> dict[str, Any]:
    """Processes every item; failed items are collected and raised as one system_error after the good ones were written."""
    items, dir_in, dir_out, config, workers = _validate(params)
    processed: list[dict[str, str]] = []
    quarantined: list[dict[str, Any]] = []
    metrics = {"items": 0, "anonymized": 0, "passed_through": 0, "quarantined": 0}
    failures: list[str] = []
    for index, entry in enumerate(items):
        try:
            outcome = _process_item(entry, dir_in, dir_out, deps, config, params, workers)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            failures.append(f"{entry['key']} ({error})")
        else:
            processed.append(outcome["file"])
            quarantined.extend(outcome["quarantined"])
            metrics["items"] += 1
            metrics["anonymized"] += outcome["anonymized"]
            metrics["passed_through"] += outcome["passed_through"]
            metrics["quarantined"] += len(outcome["quarantined"])
        on_progress(int((index + 1) * 100 / len(items)), f"{index + 1}/{len(items)} {entry['key']}")
    if failures:
        msg = f"{len(failures)} of {len(items)} items failed: {'; '.join(failures)}"
        raise AnonError(SYSTEM_ERROR, msg)
    return {"dirOut": str(dir_out), "processed": processed, "quarantined": quarantined, "metrics": metrics}


def default_workers() -> int:
    """Informational: the host core count (the default of ``workers`` is 1 so a run stays inside the memory limit: every worker loads the model)."""
    return os.cpu_count() or 1
