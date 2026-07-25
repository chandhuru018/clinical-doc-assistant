"""
Local GGUF model wrapper using llama-cpp-python.
Mirrors the interface of soap_generator.generate_soap_note() so app.py
can swap between the cloud model (Groq) and this local model easily.

NOTE: unlike the Groq model, this fine-tuned model was trained to generate
ONE SOAP section at a time (per MTS-Dialog's per-section format), not a full
4-part note in one shot. So we call it 4 times, once per section, and
assemble the result into the same structure app.py/soap_parser.py expect.
"""

from llama_cpp import Llama
import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "merged_model_q4.gguf"
)

SECTION_PROMPT_TEMPLATE = """You are a clinical documentation assistant. Read the following doctor-patient conversation transcript and write ONLY the {section_name} section of a SOAP note.

{section_instruction}

Only include information explicitly present in the transcript. Do not invent clinical details. Do not include other SOAP sections, only {section_name}.

Transcript:
\"\"\"
{transcript}
\"\"\"

{section_name}:
"""

SECTIONS = [
    ("SUBJECTIVE", "Describe the patient's reported symptoms, history, and concerns, in their own words where possible."),
    ("OBJECTIVE", "Describe observable findings, vitals, and exam results mentioned in the conversation."),
    ("ASSESSMENT", "Give the clinical impression or diagnosis discussed."),
    ("PLAN", "Describe the treatment plan, medications, and follow-up instructions."),
]

_llm = None  # loaded lazily, once, and reused


def load_model():
    """Load the local GGUF model into memory. Takes ~30-90s on CPU the first time."""
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=4,
            verbose=False,
        )
    return _llm


def _generate_section(llm, section_name: str, section_instruction: str, transcript: str, max_tokens: int) -> str:
    prompt = SECTION_PROMPT_TEMPLATE.format(
        section_name=section_name,
        section_instruction=section_instruction,
        transcript=transcript,
    )
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
        repeat_penalty=1.3,
    )
    return response["choices"][0]["message"]["content"].strip()


def generate_soap_note_local(transcript: str, max_tokens_per_section: int = 200) -> str:
    """Local equivalent of soap_generator.generate_soap_note(), using the
    fine-tuned model. Calls the model once per section (matching how it was
    fine-tuned) and assembles a full note in the same header format the
    Groq version produces, so soap_parser.py can parse either one."""
    llm = load_model()

    parts = []
    for section_name, section_instruction in SECTIONS:
        text = _generate_section(llm, section_name, section_instruction, transcript, max_tokens_per_section)
        parts.append(f"{section_name}:\n{text}")

    return "\n\n".join(parts)