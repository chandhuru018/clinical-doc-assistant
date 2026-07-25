import time
import pandas as pd
from rouge_score import rouge_scorer
from src.transcript_loader import clean_transcript
from src.soap_generator import generate_soap_note

DATA_PATH = "data/MTS-Dialog/Main-Dataset/MTS-Dialog-TrainingSet.csv"
NUM_SAMPLES = 15  # keep small to respect free API rate limits

def run_evaluation():
    df = pd.read_csv(DATA_PATH)
    sample_df = df.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    results = []

    for i, row in sample_df.iterrows():
        dialogue = row["dialogue"]
        reference_note = row["section_text"]
        section = row["section_header"]

        print(f"[{i+1}/{NUM_SAMPLES}] Generating note for section: {section}...")

        try:
            cleaned = clean_transcript(dialogue)
            generated_note = generate_soap_note(cleaned)

            scores = scorer.score(reference_note, generated_note)

            results.append({
                "id": row["ID"],
                "section_header": section,
                "rouge1_f1": round(scores["rouge1"].fmeasure, 3),
                "rougeL_f1": round(scores["rougeL"].fmeasure, 3),
            })
        except Exception as e:
            print(f"  Error on row {i}: {e}")

        time.sleep(2)  # avoid hitting free-tier rate limits

    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)

    print("\n=== Evaluation Summary ===")
    print(f"Samples evaluated: {len(results_df)}")
    print(f"Average ROUGE-1 F1: {results_df['rouge1_f1'].mean():.3f}")
    print(f"Average ROUGE-L F1: {results_df['rougeL_f1'].mean():.3f}")
    print("\nFull results saved to evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()