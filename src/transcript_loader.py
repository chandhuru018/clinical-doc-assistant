import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Loads MTS-Dialog CSV and returns dialogue + reference note columns."""
    df = pd.read_csv(csv_path)
    return df

def clean_transcript(text: str) -> str:
    """Basic cleanup: strip whitespace, normalize spacing."""
    text = text.strip()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return " ".join(sentences)