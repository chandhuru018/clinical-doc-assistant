from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

_MODEL_NAME = "d4data/biomedical-ner-all"

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForTokenClassification.from_pretrained(_MODEL_NAME)

_ner_pipeline = pipeline(
    "token-classification",
    model=_model,
    tokenizer=_tokenizer,
    aggregation_strategy="simple",
)

def extract_medical_entities(text: str) -> list[dict]:
    """Returns a list of {entity_group, word, score} for medical entities found in text."""
    results = _ner_pipeline(text)
    cleaned = [
        {
            "entity": r["entity_group"],
            "text": r["word"],
            "confidence": round(float(r["score"]), 2),
        }
        for r in results
    ]
    return cleaned