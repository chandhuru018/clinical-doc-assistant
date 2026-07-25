import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SOAP_PROMPT_TEMPLATE = """You are a clinical documentation assistant. Convert the following doctor-patient conversation transcript into a structured SOAP note.

Follow this exact format:

SUBJECTIVE:
(Patient's reported symptoms, history, and concerns)

OBJECTIVE:
(Observable findings, vitals, exam results mentioned in the conversation)

ASSESSMENT:
(Clinical impression / diagnosis discussed)

PLAN:
(Treatment plan, medications, follow-up instructions)

Only include information explicitly present in the transcript. Do not invent clinical details.

Transcript:
\"\"\"
{transcript}
\"\"\"

SOAP Note:
"""

def generate_soap_note(transcript: str, model: str = "llama-3.3-70b-versatile") -> str:
    prompt = SOAP_PROMPT_TEMPLATE.format(transcript=transcript)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content