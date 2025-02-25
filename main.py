import streamlit as st
import pyaudio
import wave
import whisper
import threading

st.title("Audio Recorder with Transcription and Summary")

recording = False
frames = []
audio = pyaudio.PyAudio()


def start_recording(samplerate=44100, channels=1, chunk=1024, format=pyaudio.paInt16):
    """Starts recording audio using PyAudio in a separate thread."""
    global recording, frames
    recording = True
    frames = []
    stream = audio.open(format=format, channels=channels, rate=samplerate, input=True, frames_per_buffer=chunk)
    
    while recording:
        data = stream.read(chunk)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()

def stop_recording(output_filename="recorded_audio.wav", samplerate=44100):
    """Stops recording and saves the audio file."""
    global recording, frames
    recording = False
    
    wf = wave.open(output_filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(samplerate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return output_filename

def transcribe_audio(file_path):
    """Transcribes the recorded audio using Whisper."""
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

def summarize_text(text):
    """Generates a simple summary of the transcribed text."""
    summary = text[:200] + "..." if len(text) > 200 else text
    return summary

if st.button("Start Recording"):
    st.session_state.recording_thread = threading.Thread(target=start_recording)
    st.session_state.recording_thread.start()
    st.info("Recording started...")

if st.button("Stop Recording"):
    if "recording_thread" in st.session_state:
        stop_recording()
        st.success("Recording stopped!")
    
        st.info("Transcribing audio...")
        transcription = transcribe_audio("recorded_audio.wav")
        st.text_area("Transcription", transcription, height=200)
    
        st.info("Generating summary...")
        summary = summarize_text(transcription)
        st.text_area("Summary", summary, height=100)
    
        with open("recorded_audio.wav", "rb") as f:
            st.download_button("Download Audio", f, file_name="recorded_audio.wav", mime="audio/wav")
