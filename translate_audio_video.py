import streamlit as st
import re
import os
import asyncio
import nest_asyncio
import openai
import edge_tts
import whisper
from pydub.utils import which
from pydub import AudioSegment
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import tempfile
from datetime import timedelta

# --- FFmpeg Configuration ---
os.environ["PATH"] = r"C:\ffmpeg\bin" + ";" + os.environ["PATH"]
ffmpeg_path = which("ffmpeg")

# --- Streamlit Config ---
st.set_page_config(page_title="Video Translator Pro", layout="wide")
nest_asyncio.apply()

# --- Core Functions ---
def parse_transcript(transcript: str):
    segments = []
    pattern = r'^\s*(\d{1,2}):(\d{2}):\s*(.*)$'
    for line in transcript.splitlines():
        match = re.match(pattern, line)
        if match:
            minutes, seconds, text = match.groups()
            start_time = int(minutes) * 60 + int(seconds)
            segments.append((start_time, text.strip()))
    return segments

def improve_translation(text: str, api_key: str) -> str:
    openai.api_key = api_key
    prompt = f"Translate to French while keeping technical terms:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message["content"].strip()

async def generate_tts(text: str, output_file: str, voice: str, rate: str):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_file)

def process_audio_segment(text: str, output_file: str, voice: str, rate: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(generate_tts(text, output_file, voice, rate))

def extract_audio(video_path: str, progress_bar):
    progress_bar.progress(10, text="Extracting audio from video...")
    video = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", "_audio.wav")
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    return audio_path

def transcribe_audio(audio_path: str, progress_bar):
    progress_bar.progress(25, text="Transcribing audio content...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    
    formatted_transcript = ""
    for segment in result["segments"]:
        start_time = segment['start']
        text = segment['text'].strip()
        formatted_transcript += f"{timedelta(seconds=start_time)}: {text}\n"
    return formatted_transcript

def create_synchronized_audio(segments, output_path, voice, rate, use_llm, api_key, progress_bar):
    audio_segments = []
    total_segments = len(segments)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, (start_time, text) in enumerate(segments):
            try:
                progress = 40 + int((i/total_segments)*40)
                progress_bar.progress(progress, 
                    text=f"Processing segment {i+1}/{total_segments}..."
                )
                
                translated = improve_translation(text, api_key) if use_llm else text
                segment_path = os.path.join(temp_dir, f"segment_{i}.mp3")
                process_audio_segment(translated, segment_path, voice, rate)
                segment_audio = AudioSegment.from_mp3(segment_path)
                
                if i < len(segments)-1:
                    next_start = segments[i+1][0]
                    target_duration = (next_start - start_time) * 1000
                else:
                    target_duration = len(segment_audio)
                
                if len(segment_audio) < target_duration:
                    segment_audio += AudioSegment.silent(duration=target_duration - len(segment_audio))
                else:
                    segment_audio = segment_audio[:target_duration]
                
                audio_segments.append(segment_audio)
            except Exception as e:
                st.error(f"Error in segment {i}: {str(e)}")
                raise
        
        progress_bar.progress(85, text="Finalizing audio track...")
        final_audio = sum(audio_segments)
        final_audio.export(output_path, format="mp3")

def merge_media(video_path, audio_path, output_path, progress_bar):
    progress_bar.progress(90, text="Merging audio with video...")
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    final = video.set_audio(audio)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=4)

# --- Streamlit UI ---
def main():
    st.title("Professional Video Translator")
    
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload Video", 
                      type=["mp4", "mov", "avi"],
                      accept_multiple_files=False)
        
        target_lang = st.selectbox("Target Voice", [
            "fr-CA-SylvieNeural", "en-US-AriaNeural",
            "es-ES-AlvaroNeural", "de-DE-KillianNeural"
        ])
        speech_rate = st.select_slider("Speech Speed", [
            "-30%", "-20%", "-10%", "0%", "+10%", "+20%"
        ], "-10%")
        use_ai = st.checkbox("Enable AI Translation")
        api_key = st.text_input("OpenAI API Key", type="password") if use_ai else ""
    
    if uploaded_file:
        progress_bar = st.progress(0, text="Initializing...")
        
        with tempfile.TemporaryDirectory() as work_dir:
            try:
                # Save uploaded file
                video_path = os.path.join(work_dir, "source.mp4")
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process audio
                audio_path = extract_audio(video_path, progress_bar)
                transcript = transcribe_audio(audio_path, progress_bar)
                segments = parse_transcript(transcript)
                
                # Transcript editing
                st.subheader("Generated Transcript")
                edited_transcript = st.text_area("Edit transcript:", 
                                               value=transcript, 
                                               height=300)
                
                if st.button("Generate Translated Video"):
                    try:
                        segments = parse_transcript(edited_transcript)
                        
                        # Audio processing
                        final_audio = os.path.join(work_dir, "final_audio.mp3")
                        create_synchronized_audio(
                            segments=segments,
                            output_path=final_audio,
                            voice=target_lang,
                            rate=speech_rate,
                            use_llm=use_ai,
                            api_key=api_key,
                            progress_bar=progress_bar
                        )
                        
                        # Video merging
                        output_path = os.path.join(work_dir, "output.mp4")
                        merge_media(video_path, final_audio, output_path, progress_bar)
                        
                        # Final output
                        progress_bar.progress(100, text="Processing complete!")
                        st.success("Translation finished!")
                        
                        st.video(output_path)
                        with open(output_path, "rb") as f:
                            st.download_button(
                                "Download Video",
                                f.read(),
                                file_name=f"translated_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        progress_bar.empty()
                
            except Exception as e:
                st.error(f"Initialization failed: {str(e)}")
                progress_bar.empty()

if __name__ == "__main__":
    main()