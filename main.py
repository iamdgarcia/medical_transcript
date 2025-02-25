import streamlit as st
from st_audiorec import st_audiorec
import whisper
import tempfile
import os
from openai import OpenAI
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

st.set_page_config(page_title="Audio Recorder with Transcription")
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''', unsafe_allow_html=True)
st.markdown('''<style>.stAudio {height: 45px;}</style>''', unsafe_allow_html=True)
st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)
st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)
st.markdown("""
    <style>
        .summary-box {
            font-size: 18px;
            background-color: #f8f9fa;
            text-color: black;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
        }
    </style>
""", unsafe_allow_html=True)
def transcribe_audio(file_path):
    """Transcribes the recorded audio using Whisper Medium model for Spanish."""
    if not file_path:
        return "No audio recorded."
    model = whisper.load_model(os.environ.get("WHISPER_MODEL","base"), device="cpu")
    result = model.transcribe(file_path, language="es",fp16=False)
    return result["text"]

def summarize_text_with_llm(text):
    """Generates a summary using an LLM (GPT-4) with OpenAI's latest API format."""
    prompt = (
        "Extrae la información más importante de la siguiente conversación entre un doctor y un paciente y genera un resumen estructurado. "
        "El resumen debe incluir lo siguiente:\n\n"
        "1. **Introducción breve**: Describe el motivo de la consulta o tema principal discutido.\n"
        "2. **Síntomas o problemas discutidos**: Explica los síntomas mencionados por el paciente, su duración, frecuencia e intensidad.\n"
        "3. **Diagnóstico o evaluación médica**: Indica el diagnóstico (provisional o final) y las observaciones clave del doctor.\n"
        "4. **Tratamiento o recomendaciones**: Menciona los medicamentos prescritos (nombre, dosis, duración), pruebas solicitadas y recomendaciones generales.\n"
        "5. **Próximos pasos**: Indica la fecha de la próxima consulta y cualquier documento o examen requerido.\n\n"
        "Elimina comentarios irrelevantes o repeticiones y organiza el resumen en secciones bien definidas.\n\n"
        f"Conversación:\n{text}"
    )
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Eres un asistente que genera resúmenes médicos estructurados a partir de conversaciones."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-4o",
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

st.title("🩺 Audio Recorder with Medical Transcription and Summary")
st.write("**Record a conversation between a doctor and a patient, get an automatic transcription, and receive a structured summary.**")

# Record audio
wav_audio_data = st_audiorec()
if wav_audio_data:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_wav.write(wav_audio_data)
        audio_file_path = temp_wav.name
    
    st.success("✅ Recording complete!")
    st.audio(wav_audio_data, format='audio/wav')
    
    st.info("Transcribing audio in Spanish...")
    transcription = transcribe_audio(audio_file_path)
    st.text_area("Transcription", transcription, height=200)
    
    st.info("Generating structured medical summary...")
    summary = summarize_text_with_llm(transcription)
    st.markdown("### 📋 Summary")
    st.markdown(summary, unsafe_allow_html=True)
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.download_button("📥 Download Summary", summary, file_name="medical_summary.txt", mime="text/plain")
    with col2:
        st.download_button("Download Audio", wav_audio_data, file_name="recorded_audio.wav", mime="audio/wav")
    
