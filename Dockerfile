# dp-it-giurisprudenza-anonymize production image (Python). proc-protocol stdin loop on main.py.
# The domain NER model (DeepMount00/Italian_NER_XXL, Apache-2.0) is PRE-DOWNLOADED into the image at build time, so the container never
# fetches anything at runtime (offline, air-gap safe). See NOTICE.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the transformer model into the HF cache so it is present offline at runtime.
RUN python -c "from transformers import AutoTokenizer, AutoModelForTokenClassification; \
    m='DeepMount00/Italian_NER_XXL'; AutoTokenizer.from_pretrained(m); \
    AutoModelForTokenClassification.from_pretrained(m)"

COPY *.py ./
COPY src ./src

# Force offline at runtime: a network fetch would be a bug, not a fallback.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/app/src

ENTRYPOINT ["python", "main.py"]
