"""Configurable redaction posture (DATA, not code). The default posture redacts ONLY what identifies
a physical/legal person (names, birth data, CF, contacts, addresses, financial/device identifiers,
cadastral); everything else — courts, laws, cities, banks, sentence numbers, amounts — stays in clear.
Absolute-category documents (minors / family / personal status) additionally over-block LUOGO and
STATO_CIVILE. An `anonConfig` key in the execute params deep-merges over DEFAULT_CONFIG so an operator
changes the posture without touching code.
"""
from dataclasses import dataclass, field, replace

DEFAULT_MODEL = "DeepMount00/Italian_NER_XXL"
DEFAULT_THRESHOLD = 0.5
PERSON_TOKEN_PREFIX = "PERSONA"

DEFAULT_PERSON_LABELS = frozenset({"NOME", "COGNOME", "AVV_NOTAIO"})
DEFAULT_TYPE_TOKEN = {
    "DATA_NASCITA": "[DATA_NASCITA]", "ETA": "[ETA]", "CODICE_FISCALE": "[CF]", "P_IVA": "[PIVA]",
    "EMAIL": "[EMAIL]", "NUMERO_TELEFONO": "[TELEFONO]", "INDIRIZZO": "[INDIRIZZO]",
    "CODICE_POSTALE": "[CAP]", "MAPPALE": "[MAPPALE]", "IBAN": "[IBAN]", "BIC": "[BIC]",
    "NUMERO_CONTO": "[CONTO]", "NUMERO_CARTA": "[CARTA]", "CVV": "[CVV]",
    "NUM_ASSEGNO_BANCARIO": "[ASSEGNO]", "TARGA_VEICOLO": "[TARGA]", "IMEI": "[IMEI]",
    "MAC": "[MAC]", "IP": "[IP]", "IPV6_1": "[IP]", "PASSWORD": "[PASSWORD]", "PIN": "[PIN]",
}
DEFAULT_ABSOLUTE_EXTRA = {"LUOGO": "[LUOGO]", "STATO_CIVILE": "[STATO_CIVILE]"}
DEFAULT_ABSOLUTE_MATERIA = ("famiglia", "minori", "minore", "persone", "stato civile",
                            "filiazione", "adozione", "separazione", "divorzio")
DEFAULT_ABSOLUTE_KW = ("minore", "minori", "adozione", "filiazione", "affidamento", "separazione",
                       "divorzio", "stato di figlio", "interdizione", "amministrazione di sostegno",
                       "interesse del minore")


@dataclass(frozen=True)
class AnonConfig:
    model_name: str = DEFAULT_MODEL
    score_threshold: float = DEFAULT_THRESHOLD
    person_labels: frozenset = DEFAULT_PERSON_LABELS
    type_token: dict = field(default_factory=lambda: dict(DEFAULT_TYPE_TOKEN))
    absolute_extra: dict = field(default_factory=lambda: dict(DEFAULT_ABSOLUTE_EXTRA))
    absolute_materia: tuple = DEFAULT_ABSOLUTE_MATERIA
    absolute_kw: tuple = DEFAULT_ABSOLUTE_KW

    def token_for(self, label, absolute=False):
        if label == "PERSON" or label in self.person_labels:
            return ""                                   # person sentinel → caller assigns [PERSONA_n]
                                                        # "PERSON" = merged NER label, always a person
        if label in self.type_token:
            return self.type_token[label]
        if absolute and label in self.absolute_extra:
            return self.absolute_extra[label]
        return None                                     # KEEP in clear

    def is_absolute(self, meta, text):
        meta = meta or {}
        materia = (meta.get("materia") or "").lower()
        szdec = (meta.get("szdec") or "").lower()
        if any(k in materia for k in self.absolute_materia) or any(k in szdec for k in self.absolute_materia):
            return True
        blob = (text or "").lower()
        return any(k in blob for k in self.absolute_kw)

    @staticmethod
    def from_params(params):
        over = (params or {}).get("anonConfig") or {}
        base = AnonConfig()
        changes = {}
        if "model" in over: changes["model_name"] = over["model"]
        if "scoreThreshold" in over: changes["score_threshold"] = float(over["scoreThreshold"])
        if "personLabels" in over: changes["person_labels"] = frozenset(over["personLabels"])
        if "typeToken" in over:                          # deep-merge over defaults
            merged = dict(DEFAULT_TYPE_TOKEN); merged.update(over["typeToken"]); changes["type_token"] = merged
        if "absoluteExtra" in over:
            merged = dict(DEFAULT_ABSOLUTE_EXTRA); merged.update(over["absoluteExtra"]); changes["absolute_extra"] = merged
        if "absoluteMateria" in over: changes["absolute_materia"] = tuple(over["absoluteMateria"])
        if "absoluteKw" in over: changes["absolute_kw"] = tuple(over["absoluteKw"])
        return replace(base, **changes)


DEFAULT_CONFIG = AnonConfig()
