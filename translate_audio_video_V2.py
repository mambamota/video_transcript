import os
import re
import asyncio
import nest_asyncio
import openai
import edge_tts
import whisper
import streamlit as st
from pydub import AudioSegment
from pydub.utils import which
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from tempfile import NamedTemporaryFile
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
# Set FFmpeg path FOR WINDOWS (do not remove this line)
#os.environ["PATH"] = r"C:\ffmpeg\bin" + ";" + os.environ["PATH"]
#ffmpeg_path = which("ffmpeg")

# --- Configuration ---
# Set FFmpeg path for Ubuntu server (do not remove this line)
#os.environ["PATH"] = "/root/mmb/ffmpeg/bin:" + os.environ["PATH"]
ffmpeg_path = which("ffmpeg")

from shutil import which

# No need to modify PATH if ffmpeg is installed system-wide
ffmpeg_path = which("ffmpeg")

if not ffmpeg_path:
    raise RuntimeError("ffmpeg not found. Please install it using: sudo apt install ffmpeg")

print(f"✅ ffmpeg found at: {ffmpeg_path}")

# Temporary directory for segment audio files
SEGMENTS_DIR = "segments_temp"
os.makedirs(SEGMENTS_DIR, exist_ok=True)

# Default parameters
VOICE_CHOICES = [
    "fr-CA-SylvieNeural", 
    "fr-FR-DeniseNeural", 
    "fr-CA-CHantalNeural"
]
DEFAULT_VOICE = VOICE_CHOICES[0]
DEFAULT_RATE = "-10%"                 # Default speaking rate
OUTPUT_VIDEO = "translated_video.mp4"
FINAL_AUDIO_FILE = "final_voice.mp3"

# --- Utility Functions ---


def improve_translation(text: str, user_api_key: str) -> str:
    """
    Use OpenAI's LLM to translate/improve the text into natural French.
    The API key used will be the one provided by the user (or fallback to .env).
    """
    # Use the provided API key if available; otherwise, rely on the one already set (e.g., via .env)
    openai.api_key = user_api_key if user_api_key else os.getenv("OPENAI_API_KEY")
    
    prompt = (
        "Please translate the following text into natural, fluent French, "
        "improving the wording for clarity:\n\n" + text
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Use a supported model name
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    translated_text = response.choices[0].message["content"].strip()
    return translated_text

def translate_text(text: str, use_llm: bool, user_api_key: str = None) -> str:
    """
    Translates the given text into French.
    Uses OpenAI LLM if enabled and a key is provided; otherwise uses fallback with deep-translator.
    """
    if use_llm:
        try:
            return improve_translation(text, user_api_key)
        except Exception as e:
            st.warning(f"LLM translation failed, falling back to GoogleTranslator. Error: {e}")
    
    try:
        return GoogleTranslator(source='auto', target='fr').translate(text)
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return text  # Fallback to original text

async def generate_segment_audio(text: str, output_file: str, voice: str, rate: str):
    """
    Asynchronously generate TTS audio for the given text segment using Edge-TTS.
    """
    communicator = edge_tts.Communicate(text, voice, rate=rate)
    await communicator.save(output_file)

def run_generate_audio_for_segment(text: str, output_file: str, voice: str, rate: str):
    """
    Run the asynchronous TTS generation for a segment.
    """
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_segment_audio(text, output_file, voice, rate))


def merge_audio_with_video(video_path: str, audio_path: str, output_video: str = OUTPUT_VIDEO):
    """
    Merge the final audio with the original video.
    """
    st.info("Merging the translated audio with the video...")
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    final_video = video_clip.without_audio().set_audio(audio_clip)
    final_video.write_videofile(output_video, codec="libx264", audio_codec="aac")
    st.success("Final video saved!")
    return output_video


def generate_transcript(video_path: str) -> str:
    """
    Use Whisper to generate a transcript from the video.
    Returns a transcript string with each line formatted as "mm:ss - mm:ss: text" 
    if end time is available, otherwise "mm:ss: text".
    """
    st.info("Generating transcript using Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    transcript_lines = []
    for segment in result["segments"]:
        start_minutes = int(segment["start"] // 60)
        start_seconds = int(segment["start"] % 60)
        clean_text = segment["text"].strip().replace("\n", " ")
        # Check if end time is available; Whisper's result usually includes "end"
        if "end" in segment and segment["end"] is not None:
            end_minutes = int(segment["end"] // 60)
            end_seconds = int(segment["end"] % 60)
            transcript_lines.append(
                f"{start_minutes:01d}:{start_seconds:02d} - {end_minutes:01d}:{end_seconds:02d}: {clean_text}"
            )
        else:
            transcript_lines.append(
                f"{start_minutes:01d}:{start_seconds:02d}: {clean_text}"
            )
    transcript_text = "\n".join(transcript_lines)
    st.success("Transcript generated!")
    return transcript_text


def parse_transcript(transcript: str):
    """
    Parse a transcript with timestamps.
    
    It first tries to parse the extended format:
      "mm:ss - mm:ss: text"
    If that fails, it falls back to the simple format:
      "mm:ss: text"
      
    Returns a list of segments:
      - When end time is available: (start_time, end_time, text)
      - Otherwise: (start_time, None, text)
    """
    segments = []
    # Extended pattern: start and end time present
    pattern_extended = r'^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2}):\s*(.*)$'
    # Simple pattern: only start time provided
    pattern_simple = r'^\s*(\d{1,2}):(\d{2}):\s*(.*)$'
    
    for line in transcript.splitlines():
        match = re.match(pattern_extended, line)
        if match:
            start_min, start_sec, end_min, end_sec, text = match.groups()
            start_time = int(start_min) * 60 + int(start_sec)
            end_time = int(end_min) * 60 + int(end_sec)
            segments.append((start_time, end_time, text.strip()))
        else:
            match = re.match(pattern_simple, line)
            if match:
                start_min, start_sec, text = match.groups()
                start_time = int(start_min) * 60 + int(start_sec)
                segments.append((start_time, None, text.strip()))
    return segments


def create_synchronized_audio(segments, use_llm: bool, voice: str, rate: str, user_api_key: str = None, progress_callback=None):
    """
    Generate audio for each transcript segment, adjust duration, and concatenate them 
    into one final audio file.
    
    Improvement:
      - Uses the end_time (if available) to set the intended duration.
      - Otherwise, uses the next segment's start time (or falls back to the generated audio length).
    """
    audio_segments = []
    num_segments = len(segments)
    
    for i, seg in enumerate(segments):
        # Unpack segment tuple. seg can be (start, end, text)
        if len(seg) == 3:
            start_time, end_time, text = seg
        else:
            start_time, text = seg
            end_time = None
        
        # Optionally improve translation via LLM (or fallback to deep-translator)
        translated_text = translate_text(text, use_llm, user_api_key)
        
        if not translated_text or len(translated_text.strip()) == 0:
            raise ValueError(f"Segment {i} (start: {start_time}s) has empty translated text!")
        
        segment_filename = os.path.join(SEGMENTS_DIR, f"segment_{i}.mp3")
        run_generate_audio_for_segment(translated_text, segment_filename, voice, rate)
        segment_audio = AudioSegment.from_file(segment_filename)
        
        # Determine intended duration in milliseconds:
        if end_time is not None:
            intended_duration_ms = (end_time - start_time) * 1000
        elif i < num_segments - 1:
            next_start = segments[i+1][0]  # Use start time of next segment
            intended_duration_ms = (next_start - start_time) * 1000
        else:
            intended_duration_ms = len(segment_audio)
        
        # Adjust segment audio length:
        if len(segment_audio) < intended_duration_ms:
            silence = AudioSegment.silent(duration=intended_duration_ms - len(segment_audio))
            segment_audio = segment_audio + silence
        else:
            segment_audio = segment_audio[:intended_duration_ms]
        
        # Add leading silence for the very first segment if needed
        if i == 0 and start_time > 0:
            leading_silence = AudioSegment.silent(duration=start_time * 1000)
            audio_segments.append(leading_silence)
        
        audio_segments.append(segment_audio)
        if progress_callback:
            progress_callback((i + 1) / num_segments * 50)  # Update progress (first 50%)
    
    final_audio = sum(audio_segments)
    final_audio.export(FINAL_AUDIO_FILE, format="mp3")
    st.success("Final synchronized audio generated!")
    return FINAL_AUDIO_FILE



# --- Streamlit Interface ---
st.title("Video Translator & Voice Over Generator")

# Sidebar options for user input
st.sidebar.header("Settings")
voice = st.sidebar.selectbox("Target Audio Voice", options=VOICE_CHOICES, index=0)
rate = st.sidebar.text_input("Target Audio Rate (e.g., -10%)", value=DEFAULT_RATE)
use_llm = st.sidebar.checkbox("Improve translation with LLM", value=False)
user_api_key = None
if use_llm:
    user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if not user_api_key:
        st.sidebar.warning("LLM translation is enabled. Please provide your OpenAI API Key.")

# Video uploader (max file size set in Streamlit configuration if needed)
uploaded_file = st.file_uploader("Upload your video file (Max 5GB)", type=["mp4"])

if uploaded_file is not None:
    # Save uploaded video temporarily
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_file_path = tmp_file.name
    st.video(video_file_path, format="video/mp4")
    
    if st.button("Process Video"):
        progress_bar = st.progress(0)
        try:
            # Generate transcript from video using Whisper (transcript not shown to user)
            transcript_text = generate_transcript(video_file_path)
            progress_bar.progress(10)
            
            segments = parse_transcript(transcript_text)
            if not segments:
                st.error("No transcript segments were found. Please check the transcript format.")
            else:
                # Create synchronized audio from transcript segments
                final_audio_file = create_synchronized_audio(
                    segments, use_llm, voice, rate,
                    user_api_key=user_api_key,
                    progress_callback=lambda val: progress_bar.progress(10 + int(val))
                )
                progress_bar.progress(60)
                
                # Merge the generated audio with the original video
                final_video_path = merge_audio_with_video(video_file_path, final_audio_file)
                progress_bar.progress(90)
                
                # Display the final video in the app
                st.video(final_video_path, format="video/mp4")
                
                # Provide a download button for the final video
                with open(final_video_path, "rb") as vid_file:
                    st.download_button(
                        label="Download Translated Video",
                        data=vid_file,
                        file_name=OUTPUT_VIDEO,
                        mime="video/mp4"
                    )
                progress_bar.progress(100)
                st.success("Process completed successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
