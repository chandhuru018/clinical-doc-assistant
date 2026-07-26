# 🩺 AI Clinical Documentation Assistant

An AI-powered tool that converts doctor-patient conversation transcripts (typed or audio) into structured, editable SOAP notes — with multilingual support and an offline-capable fine-tuned model for low-connectivity settings.

**🔗 Live demo:** [clinical-doc-assistant.streamlit.app]** https://clinical-doc-assistant.streamlit.app

> ⚠️ This tool generates AI drafts only. All output must be reviewed by a qualified clinician before being added to an official patient record.

---

## What it does

1. **Input** — type/paste a conversation transcript, or upload an audio recording (mp3/wav/m4a)
2. **Transcription** — audio is transcribed via Groq's Whisper API; non-English conversations can be auto-translated to English in the same step
3. **SOAP generation** — the transcript is converted into a structured 4-part clinical note (Subjective / Objective / Assessment / Plan)
4. **Entity extraction** — a biomedical NER model (ClinicalBERT) pulls out symptoms, body structures, durations, and other clinical entities for quick reference
5. **Review & edit** — every section is editable before saving
6. **Export** — save to session history or download as a formatted Word document

---

## Architecture: two deployment modes

This project intentionally supports **two different ways of running**, because they serve different needs:

| | **Cloud (deployed)** | **Local / offline** |
|---|---|---|
| Model | Groq API — Llama 3.3 70B | Custom fine-tuned Llama 3.2 3B (QLoRA), quantized to GGUF (Q4_K_M) |
| Requires internet | Yes | No |
| Speed | Seconds | Minutes (CPU-only) |
| Where it runs | Streamlit Community Cloud | Doctor's own machine |
| Use case | General access, demo | Low-connectivity clinics with no reliable internet |

The local model can't be deployed to the public website — Streamlit Community Cloud's free tier guarantees only 1GB RAM, and the quantized model alone is ~1.9GB. Instead, it's designed to be run entirely on a clinician's own computer as a separate offline tool. See [Running locally / offline](#running-locally--offline) below.

---

## Tech stack

- **UI:** Streamlit
- **Cloud LLM:** Groq API (Llama 3.3 70B for generation, Whisper large-v3 for transcription/translation)
- **Local LLM:** Fine-tuned Llama 3.2 3B, converted to GGUF, run via `llama-cpp-python`
- **Fine-tuning:** QLoRA (via Unsloth) on the [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) dataset (~9,600 examples), trained on a free Colab T4 GPU
- **NER:** `d4data/biomedical-ner-all` (ClinicalBERT) via  Transformers
- **NLP preprocessing:** spaCy
- **Document export:** python-docx
- **Evaluation:** ROUGE (via `rouge_score`)

---

## Setup

### Cloud version (Groq only)

```bash
git clone https://github.com/chandhuru018/clinical-doc-assistant.git
cd clinical-doc-assistant
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run:
```bash
streamlit run app.py
```

### Running locally / offline

The fine-tuned local model is not included in this repo (it's ~1.9GB — too large for GitHub, and excluded via `.gitignore`). To use the offline mode:

1. Follow the fine-tuning process described below to produce `merged_model_q4.gguf`, or obtain a copy separately
2. Place it in `models/merged_model_q4.gguf`
3. Install the local-inference dependency (not included in the default `requirements.txt`, since it's not needed for cloud deployment):
   ```bash
   pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   ```
4. Run the app as normal — the sidebar will now show a "Local (fine-tuned, offline)" model option

**Hardware note:** local generation was developed and tested on a CPU-only laptop (Intel i3, 8GB RAM). Expect generation times of 1–3 minutes per note, and close other memory-heavy applications before running for best performance.

---

## Fine-tuning process

- Base model: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- Method: QLoRA (r=16, lora_alpha=16), 1 epoch, 600 steps
- Data: MTS-Dialog Main Dataset + all Augmented Data splits (9,608 examples), formatted as conversation → single SOAP section
- Training loss: 2.49 → ~1.05–1.15 (stable, no divergence)
- Post-training: merged adapter into base weights, converted to GGUF (f16), quantized to Q4_K_M (~1.9GB final size)
- Evaluated on 15 held-out MTS-Dialog samples: ROUGE-1 F1 = 0.301, ROUGE-L F1 = 0.201

**Note on the local model's output format:** because it was fine-tuned per-section (matching MTS-Dialog's structure), it doesn't natively produce a full 4-part note in one generation like the Groq model does. The app works around this by calling the local model once per SOAP section and assembling the results — see `src/local_model.py`.

---

## Known limitations

Documented honestly, based on real testing during development:

- **Hallucination:** both models, especially the smaller fine-tuned one, can occasionally invent minor details not present in the transcript (e.g., inferring a patient's age/gender when unstated, or adding a follow-up plan that wasn't discussed). All output requires clinician review — this is enforced in the UI but not automatically preventable.
- **Translation is general-purpose, not medical:** the multilingual pipeline uses Groq Whisper's general translation endpoint, not a medically fine-tuned translator. Clinical terms, drug names, or regional shorthand could be mistranslated. Recommend review by a speaker of the original language for high-stakes cases.
- **Local model speed/quality tradeoff:** the offline model is meaningfully slower and less detailed than the cloud model, reflecting the difference between a 3B and 70B parameter model. It exists for connectivity-limited use cases, not as a like-for-like replacement.
- **Evaluation caveat:** ROUGE scores are naturally lower than a section-matched benchmark because the local model's fine-tuning target (single sections) differs from the app's full 4-part note structure.

---

## Project structure

```
clinical-doc-assistant/
├── app.py                    # Main Streamlit app
├── src/
│   ├── transcript_loader.py  # spaCy-based transcript cleaning
│   ├── soap_generator.py     # Groq-based SOAP generation
│   ├── local_model.py        # Local GGUF model wrapper (offline mode)
│   ├── soap_parser.py        # Parses SOAP sections from model output
│   ├── audio_transcriber.py  # Groq Whisper transcription/translation
│   ├── entity_extractor.py   # ClinicalBERT NER
│   ├── note_exporter.py      # Word document export
│   └── evaluate.py           # ROUGE evaluation script
├── models/                   # Local GGUF model (not tracked in git)
├── data/                     # MTS-Dialog dataset (not tracked in git — see below)
├── .streamlit/config.toml    # UI theme
└── requirements.txt
```

To obtain the training dataset:
```bash
git clone https://github.com/abachaa/MTS-Dialog.git data/MTS-Dialog
```

