import streamlit as st
import tempfile
import pandas as pd
from datetime import date
from src.transcript_loader import clean_transcript
from src.soap_generator import generate_soap_note
from src.local_model import generate_soap_note_local
from src.soap_parser import parse_soap_sections
from src.audio_transcriber import transcribe_audio
from src.entity_extractor import extract_medical_entities
from src.note_exporter import build_docx

st.set_page_config(page_title="Clinical Documentation Assistant", layout="wide", page_icon="🩺")

st.markdown("""
<style>
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="select"] div,
    div[data-baseweb="input"] input {
        color: #0f172a !important;
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-size: 15px !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.15) !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] label {
        font-weight: 600 !important;
        color: #1e293b !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    h1 {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    h2, h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px;
        margin-top: 24px !important;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    .stButton button, .stDownloadButton button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

if "note_history" not in st.session_state:
    st.session_state.note_history = []

with st.sidebar:
    st.header("Patient & Visit Info")
    patient_name = st.text_input("Patient Name")
    patient_dob = st.date_input(
        "Date of Birth",
        value=date(1990, 1, 1),
        min_value=date(1920, 1, 1),
        max_value=date.today(),
    )
    visit_date = st.date_input(
        "Visit Date",
        value=date.today(),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )
    provider_name = st.text_input("Provider Name")
    visit_type = st.selectbox("Visit Type", ["General", "Follow-up", "New Patient", "Urgent Care"])

    st.divider()
    st.header("Model")
    model_choice = st.radio(
        "Note generation model",
        ["Cloud (Groq — Llama 3.3 70B)", "Local (fine-tuned, offline)"],
        help="Local model runs fully offline on your CPU. Slower, and may occasionally invent minor details not in the transcript — always review before saving.",
    )



    st.divider()
    st.header("Note History (this session)")
    if st.session_state.note_history:
        for note in reversed(st.session_state.note_history):
            with st.expander(f"{note['patient_name']} — {note['visit_date']}"):
                st.caption("Subjective (preview)")
                st.write(note["soap"]["SUBJECTIVE"][:150] + "...")
    else:
        st.caption("No notes saved yet.")

st.title("🩺 AI Clinical Documentation Assistant")
st.caption("Converts doctor-patient conversation transcripts into structured, editable SOAP notes")
st.warning("⚠️ AI-generated draft. Review and edit before adding to a patient's official record.")

if st.button("🔄 Clear and Start New Note"):
    st.session_state.clear()
    st.rerun()

input_mode = st.radio(
    "How would you like to provide the conversation?",
    ["Type/Paste Text", "Upload Audio Recording"],
    horizontal=True,
)

transcript_input = ""

if input_mode == "Type/Paste Text":
    transcript_input = st.text_area(
        "Paste the doctor-patient conversation transcript:",
        height=200,
        placeholder="Doctor: What brings you in today?\nPatient: I've had a headache for three days..."
    )
else:
    conversation_language = st.radio(
        "Conversation language",
        ["English", "Other language (auto-translate to English)"],
        horizontal=True,
        help="If the doctor/patient spoke in a regional language, choose 'Other' — Whisper will translate the conversation to English automatically before generating the note.",
    )
    audio_file = st.file_uploader("Upload a recording (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + audio_file.name.split(".")[-1]) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        needs_translation = conversation_language.startswith("Other")
        with st.spinner("Translating audio to English..." if needs_translation else "Transcribing audio..."):
            transcript_input = transcribe_audio(tmp_path, translate_to_english=needs_translation)
        transcript_input = st.text_area("Transcribed conversation (edit if needed):", value=transcript_input, height=180)

if st.button("Generate SOAP Note", type="primary", use_container_width=True):
    if not transcript_input.strip():
        st.warning("Please provide a transcript first (type it or upload audio).")
        if "generated" in st.session_state:
            del st.session_state["generated"]
    else:
        with st.spinner("Generating clinical note..." + (" (local model, this may take a minute)" if "Local" in model_choice else "")):
            cleaned = clean_transcript(transcript_input)
            if "Local" in model_choice:
                raw_note = generate_soap_note_local(cleaned)
            else:
                raw_note = generate_soap_note(cleaned)
            parsed = parse_soap_sections(raw_note)
        with st.spinner("Extracting medical entities..."):
            entities = extract_medical_entities(cleaned)

        st.session_state.generated = {
            "raw_note": raw_note,
            "parsed": parsed,
            "entities": entities,
        }
        st.success("SOAP note generated — review and edit below.")

if "generated" in st.session_state:
    gen = st.session_state.generated

    st.subheader("Review & Edit Note")
    col1, col2 = st.columns(2)
    with col1:
        subjective = st.text_area("Subjective", value=gen["parsed"].get("SUBJECTIVE", ""), height=160)
        objective = st.text_area("Objective", value=gen["parsed"].get("OBJECTIVE", ""), height=160)
    with col2:
        assessment = st.text_area("Assessment", value=gen["parsed"].get("ASSESSMENT", ""), height=160)
        plan = st.text_area("Plan", value=gen["parsed"].get("PLAN", ""), height=160)

    st.subheader("🔍 Extracted Medical Entities")
    if gen["entities"]:
        st.dataframe(pd.DataFrame(gen["entities"]), use_container_width=True)
    else:
        st.caption("No entities detected.")

    final_soap = {
        "SUBJECTIVE": subjective,
        "OBJECTIVE": objective,
        "ASSESSMENT": assessment,
        "PLAN": plan,
    }

    colA, colB = st.columns(2)
    with colA:
        if st.button("💾 Save Note to Session History", use_container_width=True):
            st.session_state.note_history.append({
                "patient_name": patient_name or "Unnamed",
                "visit_date": str(visit_date),
                "soap": final_soap,
            })
            st.success("Saved to session history (see sidebar).")

    with colB:
        docx_buffer = build_docx(patient_name, patient_dob, visit_date, provider_name, visit_type, final_soap)
        st.download_button(
            "⬇️ Download as Word Document",
            data=docx_buffer,
            file_name=f"SOAP_note_{patient_name or 'patient'}_{visit_date}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

    with st.expander("Raw model output"):
        st.text(gen["raw_note"])