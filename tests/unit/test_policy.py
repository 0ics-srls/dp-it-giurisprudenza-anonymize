"""RED tests for policy.py — the configurable redaction posture (AnonConfig).
Pure logic, no model import: runs locally with `python -m pytest tests/test_policy.py -q`.
"""
import pytest

from dp_it_giurisprudenza_anonymize.policy import AnonConfig, DEFAULT_CONFIG, DEFAULT_MODEL


# --- token_for: person labels ---------------------------------------------------------------

@pytest.mark.parametrize("label", ["NOME", "COGNOME", "AVV_NOTAIO"])
def test_token_for_person_label_returns_person_sentinel(label):
    assert DEFAULT_CONFIG.token_for(label) == ""


def test_token_for_merged_person_label_always_person_sentinel():
    # "PERSON" is the merged label the NER engine emits; it is a person even when not listed
    # in the configurable person_labels set.
    assert "PERSON" not in DEFAULT_CONFIG.person_labels
    assert DEFAULT_CONFIG.token_for("PERSON") == ""


# --- token_for: typed identifier labels -----------------------------------------------------

def test_token_for_typed_labels():
    assert DEFAULT_CONFIG.token_for("CODICE_FISCALE") == "[CF]"
    assert DEFAULT_CONFIG.token_for("INDIRIZZO") == "[INDIRIZZO]"
    assert DEFAULT_CONFIG.token_for("EMAIL") == "[EMAIL]"


# --- token_for: KEPT labels stay in clear ---------------------------------------------------

@pytest.mark.parametrize("label", ["TRIBUNALE", "LEGGE", "BANCA", "LUOGO", "N_SENTENZA", "IMPORTO"])
def test_token_for_kept_label_returns_none(label):
    assert DEFAULT_CONFIG.token_for(label) is None


# --- token_for: absolute-category over-block ------------------------------------------------

def test_token_for_luogo_absolute_only():
    assert DEFAULT_CONFIG.token_for("LUOGO", absolute=True) == "[LUOGO]"
    assert DEFAULT_CONFIG.token_for("LUOGO", absolute=False) is None


def test_token_for_stato_civile_absolute_only():
    assert DEFAULT_CONFIG.token_for("STATO_CIVILE", absolute=True) == "[STATO_CIVILE]"
    assert DEFAULT_CONFIG.token_for("STATO_CIVILE", absolute=False) is None


# --- is_absolute: category detection --------------------------------------------------------

def test_is_absolute_family_materia():
    assert DEFAULT_CONFIG.is_absolute({"materia": "diritto di famiglia"}, "testo neutro") is True


def test_is_absolute_neutral_materia_and_text():
    assert DEFAULT_CONFIG.is_absolute({"materia": "tributario"}, "avviso di accertamento IVA") is False


def test_is_absolute_from_text_keyword():
    assert DEFAULT_CONFIG.is_absolute({}, "si discute della adozione del minore in oggetto") is True


# --- from_params: defaults and overrides ----------------------------------------------------

def test_from_params_no_anon_config_uses_defaults():
    cfg = AnonConfig.from_params({})
    assert "NOME" in cfg.person_labels
    assert cfg.model_name == DEFAULT_MODEL


def test_from_params_type_token_drop_indirizzo():
    cfg = AnonConfig.from_params({"anonConfig": {"typeToken": {"INDIRIZZO": None}}})
    assert cfg.token_for("INDIRIZZO") is None


def test_from_params_score_threshold_override():
    cfg = AnonConfig.from_params({"anonConfig": {"scoreThreshold": 0.9}})
    assert cfg.score_threshold == 0.9


def test_from_params_absolute_kw_replaces_tuple():
    cfg = AnonConfig.from_params({"anonConfig": {"absoluteKw": ["parola magica"]}})
    assert cfg.absolute_kw == ("parola magica",)
    assert cfg.is_absolute({}, "contiene la parola magica qui") is True
    # a default keyword no longer matches — the tuple is REPLACED, not merged
    assert cfg.is_absolute({}, "adozione del minore") is False


def test_from_params_deep_merge_keeps_unlisted_defaults():
    cfg = AnonConfig.from_params({"anonConfig": {"typeToken": {"EMAIL": "[MAIL]"}}})
    assert cfg.token_for("EMAIL") == "[MAIL]"          # overridden
    assert cfg.token_for("CODICE_FISCALE") == "[CF]"   # unlisted default survives
    assert cfg.token_for("IBAN") == "[IBAN]"
