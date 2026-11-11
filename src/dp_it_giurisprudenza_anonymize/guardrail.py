"""PII guardrail for CGT text (port of the legacy CgtPiiDetector and CgtAnonSampleVerifier).

The MEF source anonymizes the parties itself ("M. F.", "BCC"). This module VERIFIES that promise: it finds fiscal codes, VAT numbers and titled
full party names, and stays quiet on what is lawful (judges in clear, anonymous initials, amounts, docket numbers, AdE protocol numbers). A document
with a finding is quarantined by the dispatch. A finding keeps the matched string only in memory; everything that is written holds counts per type.
"""
from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

KINDS = ("cf", "piva", "nome-parte")
_CODICE_FISCALE = re.compile(r"\b[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]\b")
# 11 ISOLATED digits: no digits, dots or slashes around; a comma counts only as a DECIMAL separator, i.e. between digits. Letters around them
# make a document code ("Avviso n. M16160007632U"), not a VAT number.
_PARTITA_IVA = re.compile(r"(?<![A-Za-z\d./])(?<!\d,)(?P<piva>\d{11})(?![A-Za-z\d./])(?!,\d)", re.IGNORECASE)
# an AdE protocol number has 11 digits too: the context "prot. n." declares it non-PII (a variable-width lookbehind in the legacy regex)
_PROTOCOL_BEFORE = re.compile(r"prot\.\s{0,3}n\.\s{0,3}$", re.IGNORECASE)
# a party name is suspect when it comes with a TITLE and the full name ("sig. Mario Rossi"): the form the source should have reduced to initials
_NOME_PARTE = re.compile(r"\b(?:[Ss]ig\.(?:ra)?|[Dd]ott\.(?:ssa)?|[Aa]vv\.|[Ii]ng\.)\s+[A-Z][a-zà-ù]+\s+[A-Z][a-zà-ù]+")


@dataclass(frozen=True)
class Finding:
    """A suspected PII: its kind (cf, piva, nome-parte) and the match. The match must never be written to a plan, report or log."""

    kind: str
    match: str


def scan(text: str) -> list[Finding]:
    findings = [Finding("cf", m.group()) for m in _CODICE_FISCALE.finditer(text)]
    for m in _PARTITA_IVA.finditer(text):
        if _PROTOCOL_BEFORE.search(text[: m.start()]):
            continue
        findings.append(Finding("piva", m.group("piva")))
    findings.extend(Finding("nome-parte", m.group()) for m in _NOME_PARTE.finditer(text) if "xx" not in m.group().lower())
    return findings


def finding_counts(findings: Iterable[Finding]) -> dict[str, int]:
    """Counts per kind, in the fixed kind order; this is what a report may carry."""
    counted = Counter(f.kind for f in findings)
    return {kind: counted[kind] for kind in KINDS if counted[kind]}


@dataclass(frozen=True)
class SampleDoc:
    """A CGT document to sample: the stratification axes (year, degree, place) and its text."""

    guid: str
    anno: int
    grado: int
    luogo: str
    text: str


def sample(docs: list[SampleDoc], target: int) -> list[SampleDoc]:
    """Stratified, deterministic sample: round robin over the strata (anno, grado, luogo) so no stratum stays uncovered."""
    if target < 1:
        msg = "target must be >= 1"
        raise ValueError(msg)
    if len(docs) <= target:
        return docs
    ordered = sorted(docs, key=lambda d: (d.anno, d.grado, d.luogo))
    strata = [deque(group) for _, group in groupby(ordered, key=lambda d: (d.anno, d.grado, d.luogo))]
    chosen: list[SampleDoc] = []
    while len(chosen) < target and any(strata):
        for stratum in strata:
            if len(chosen) >= target:
                break
            if stratum:
                chosen.append(stratum.popleft())
    return chosen


@dataclass(frozen=True)
class SampleReport:
    """The outcome of the sample check; ``clean`` is the datum the gate decides on."""

    docs_examined: int
    docs_with_pii: int
    per_type: dict[str, int]
    dirty_docs: list[tuple[str, list[Finding]]]

    @property
    def clean(self) -> bool:
        return self.docs_with_pii == 0

    def to_redaction_report(self) -> dict[str, object]:
        """The outcome of the verification (never the text): sourceAnonymized true = the source anonymizes and the sample confirmed it."""
        return {"sourceAnonymized": self.clean, "sampleDocs": self.docs_examined, "sampleDocsWithPii": self.docs_with_pii}

    def to_markdown(self) -> str:
        lines = [
            "# Verifica anonimizzazione alla fonte (campione CGT)",
            "",
            f"Esito: **{'PULITO' if self.clean else 'SPORCO'}** - {self.docs_examined} documenti esaminati, {self.docs_with_pii} con PII sospette.",
            "",
            "| Tipo PII | Trovate |",
            "|----------|---------|",
            *(f"| {kind} | {self.per_type.get(kind, 0)} |" for kind in KINDS),
            *(f"- doc {guid}: {', '.join(sorted({f.kind for f in findings}))}" for guid, findings in self.dirty_docs),
        ]
        return "\n".join(lines) + "\n"


def verify(docs: Iterable[SampleDoc]) -> SampleReport:
    per_type: Counter[str] = Counter()
    dirty: list[tuple[str, list[Finding]]] = []
    examined = 0
    for doc in docs:
        examined += 1
        findings = scan(doc.text)
        if not findings:
            continue
        dirty.append((doc.guid, findings))
        per_type.update(f.kind for f in findings)
    return SampleReport(examined, len(dirty), {kind: per_type[kind] for kind in KINDS if per_type[kind]}, dirty)
