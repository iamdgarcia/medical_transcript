import streamlit as st
import sounddevice as sd
import numpy as np
import threading
import queue
import scipy.io.wavfile as wav
import io
import whisper

st.title("Audio Recorder App")

recording_queue = queue.Queue()
recording = False
samplerate = 44100  # Sample rate

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    recording_queue.put(indata.copy())

def start_recording():
    global recording
    recording = True
    recording_queue.queue.clear()
    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
        while recording:
            sd.sleep(100)  # Keep the thread alive while recording

def stop_recording():
    global recording
    recording = False

def transcribe_audio(file_path):
    """Transcribe the recorded audio using Whisper."""
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

def summarize_text(text):
    """Generate a summary of the transcribed text."""
    summary = text[:200] + "..." if len(text) > 200 else text  # Simple truncation as a placeholder
    return summary

def process_audio(audio_data, samplerate):
    """Process the recorded audio."""
    # Example: Compute audio duration
    duration = len(audio_data) / samplerate
    st.success(f"Audio recorded! Duration: {duration:.2f} seconds")
    
    # Save to WAV file (optional)
    wav_filename = "recorded_audio.wav"
    wav.write(wav_filename, samplerate, audio_data)
    
    # Transcribe the audio
    transcription = transcribe_audio(wav_filename)
    st.text_area("Transcription", transcription, height=200)
    
    # Summarize the transcription
    summary = summarize_text(transcription)
    st.text_area("Summary", summary, height=100)
    
    # Provide download link
    with open(wav_filename, "rb") as f:
        st.download_button("Download Audio", f, file_name=wav_filename, mime="audio/wav")

if "recording_thread" not in st.session_state:
    st.session_state.recording_thread = None

if st.button("Start Recording"):
    if not st.session_state.recording_thread or not st.session_state.recording_thread.is_alive():
        st.session_state.recording_thread = threading.Thread(target=start_recording)
        st.session_state.recording_thread.start()
        st.info("Recording started...")

if st.button("Stop Recording"):
    stop_recording()
    st.info("Processing audio...")
    
    # Collect recorded audio
    audio_frames = []
    while not recording_queue.empty():
        audio_frames.append(recording_queue.get())
    
    if audio_frames:
        audio_data = np.concatenate(audio_frames, axis=0)
        process_audio(audio_data, samplerate)
    else:
        st.error("No audio recorded!")
