"""DeepMount domain NER wrapper. build_pipeline loads the HF token-classification model once (baked
into the image, HF_HUB_OFFLINE=1 at runtime); analyze returns Span(start, end, label, score) built on
the pipeline's CHARACTER offsets — never the sub-word "word" text (DeepMount fragments names into
"FR"+"##ISINA"; the offsets are correct). Adjacent person-family spans (NOME then COGNOME, whitespace
gap only) merge into ONE "PERSON" span; the family set is the person_labels parameter so a config
override changes what is merged/relabeled. The threshold applies PER SPAN for non-person labels but to
the PEAK component score for a merged person: DeepMount often scores the given name low ("ROSSI" 0.80 +
"MARIO" 0.39) and dropping the weak component would orphan-leak half a name — over-block posture for
persons. transformers is imported lazily: local collection works without the model, model tests run in
the Docker gate.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    score: float


_PIPE_CACHE = {}

_PERSON_FAMILY = frozenset({"NOME", "COGNOME", "AVV_NOTAIO"})

_DATE_FAMILY = frozenset({"DATA_NASCITA", "DATA_MORTE"})


def build_pipeline(model_name):
    from transformers import pipeline
    if model_name not in _PIPE_CACHE:
        # "max" fuses sub-word pieces into whole words ("FI"+"##LIPPO" → "FILIPPO"); "simple"
        # leaves fragments that would redact half a name. Probed on real 2024 sentences.
        _PIPE_CACHE[model_name] = pipeline("token-classification", model=model_name,
                                           aggregation_strategy="max")
    return _PIPE_CACHE[model_name]


def analyze(pipe, text, threshold=0.5, person_labels=_PERSON_FAMILY):
    if not text:
        return []
    raw = pipe(text)
    spans = [Span(int(e["start"]), int(e["end"]), e["entity_group"], float(e["score"]))
             for e in raw if e.get("start") is not None]
    spans.sort(key=lambda s: s.start)
    merged = _merge_persons(spans, text, person_labels)
    merged = _merge_dates(merged, text)
    # peak filter for every span: a fragment's low score must not orphan-leak part of a merged
    # name/date; for a non-merged span peak == score, so singles behave exactly as before.
    return [span for span, peak in merged if peak >= threshold]


def _merge_persons(spans, text, person_labels):
    """Merge adjacent person-family spans into PERSON; yield (span, peak-component-score) pairs."""
    out = []
    for s in spans:
        if (out and s.label in person_labels and out[-1][0].label == "PERSON"
                and text[out[-1][0].end:s.start].strip() == ""):
            prev, peak = out[-1]
            out[-1] = (Span(prev.start, s.end, "PERSON", min(prev.score, s.score)), max(peak, s.score))
        else:
            label = "PERSON" if s.label in person_labels else s.label
            out.append((Span(s.start, s.end, label, s.score), s.score))
    return out


def _merge_dates(pairs, text):
    """Merge adjacent date-family (span, peak) pairs — DeepMount fragments a birthdate into
    '14' + '/02/' + '1965' with mixed labels; merged label = the peak component's label."""
    out = []
    for span, peak in pairs:
        if (out and span.label in _DATE_FAMILY and out[-1][0].label in _DATE_FAMILY
                and text[out[-1][0].end:span.start].strip() == ""):
            prev, prev_peak = out[-1]
            label = prev.label if prev_peak >= peak else span.label
            out[-1] = (Span(prev.start, span.end, label, min(prev.score, span.score)),
                       max(prev_peak, peak))
        else:
            out.append((span, peak))
    return out
