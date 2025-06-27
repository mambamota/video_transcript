import os
import re
import asyncio
import nest_asyncio
import edge_tts
import whisper
import streamlit as st
from shutil import which
from pydub import AudioSegment
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from tempfile import NamedTemporaryFile
from deep_translator import GoogleTranslator
import ollama
from datetime import datetime
import glob
import shutil
import gc
import openai
import time


# --- Configuration ---
ffmpeg_path = which("ffmpeg")
if not ffmpeg_path:
    raise RuntimeError("ffmpeg not found. Please install ffmpeg first.")
print(f"✅ ffmpeg found at: {ffmpeg_path}")

SEGMENTS_DIR = "segments_temp"
os.makedirs(SEGMENTS_DIR, exist_ok=True)

VOICE_CHOICES = ["fr-CA-SylvieNeural", "fr-FR-DeniseNeural", "fr-CA-CHantalNeural"]
DEFAULT_VOICE = VOICE_CHOICES[0]
DEFAULT_RATE = "-10%"
OUTPUT_VIDEO = "translated_video.mp4"
FINAL_AUDIO_FILE = "final_voice.mp3"
            # At the top with other constants
MIN_SPEED_CHANGE = 0.01  # 1% minimum adjustment threshold

# --- Debugging Functions ---
def create_translation_log(debug_entries: list) -> str:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"translation_debug_{timestamp}.md"
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("# Translation Debug Log\n\n")
            for entry in debug_entries:
                f.write(entry + "\n---\n")
        return log_file_path
    except Exception as e:
        st.error(f"Failed to create debug log: {e}")
        return None

# --- Core Functions ---
def chunk_text(text: str, max_length: int = 1000) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def translate_text(text: str) -> str:
    try:
        # openai_api_key is expected to be defined in the UI
        if openai_api_key:
            st.info("Using OpenAI for translation...")
            return translate_with_openai(text, openai_api_key, target_language="fr")
        else:
            st.info("Using Google Translator for translation...")
            chunks = chunk_text(text, max_length=512)
            translated_chunks = []
            for chunk in chunks:
                clean_chunk = chunk.strip()
                if not clean_chunk:
                    continue
                try:
                    translated = GoogleTranslator(source='auto', target='fr').translate(clean_chunk)
                    if not translated.strip():
                        raise ValueError("Empty translation")
                    translated_chunks.append(translated)
                except Exception as e:
                    st.warning(f"Translation failed for chunk: {clean_chunk}. Using original text. Error: {e}")
                    translated_chunks.append(clean_chunk)
            return "\n".join(translated_chunks)
    except Exception as e:
        st.error(f"Translation process failed: {e}")
        return text


def clean_translation(text: str) -> str:
    text = text.strip()

    # # 1. If BOTH « ... » are present, keep only the content inside
    # if '«' in text and '»' in text:
    #     match = re.search(r'«\s*(.*?)\s*»', text, flags=re.DOTALL)
    #     if match:
    #         return match.group(1).strip()

    # 2. Remove known intro phrases (case-insensitive)
    intro_phrases = [
        r"^voici\s+(une\s+)?traduction(\s+possible)?\s*[:\-–]*",
        r"^traduction\s*[:\-–]*",
        r"^la\s+phrase\s+traduite\s+est\s*[:\-–]*",
        r"^version\s+traduite\s*[:\-–]*",
        r"^on\s+peut\s+traduire\s+cela\s+par\s*[:\-–]*",
        r"^translate\s+the\s+following\s+text.*?:",  # english junk
        r"^text\s*[:\-–]*",                          # Text:
    ]
    for pattern in intro_phrases:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # # 3. Remove any remaining French or English quotation marks
    # text = re.sub(r'^[«“"\']+\s*', '', text)
    # text = re.sub(r'\s*[»”"\']+$', '', text)

    # 4. Remove line breaks and redundant spacing
    text = text.replace("\n", " ").replace("\r", "")
    text = re.sub(r'\s+', ' ', text).strip()

    return text



def convert_time(time_str: str) -> int:
    # Convert time string "MM:SS" to total seconds.
    m, s = map(int, time_str.split(':'))
    return m * 60 + s

def parse_transcript(transcript: str):
    st.write(f"testing new version of teh function")
    sentence_groups = []
    base_segments = []
    
    # Process each line: We use a regex to extract the timestamp and text.
    for line in transcript.splitlines():
        # Remove potential prefix like "Texte :" and extra whitespace.
        line = re.sub(r'^Texte\s*:\s*', '', line).strip()
        match = re.search(r'(\d+:\d+)\s*-\s*(\d+:\d+):\s*(.+)$', line)
        if match:
            start = convert_time(match.group(1))
            end = convert_time(match.group(2))
            text = match.group(3).strip()
            base_segments.append((start, end, text))
        else:
            st.warning(f"Line skipped due to incorrect format: {line}")

    if not base_segments:
        st.error("No valid timestamped segments found in the transcript.")
        st.stop()

    # Modified grouping logic
    current_group = []
    sentence_end_pattern = re.compile(r'[.!?…](?:\s|$)')

    for seg_start, seg_end, text in base_segments:
        current_group.append((seg_start, seg_end, text))
        
        # ACTUAL MERGE LOGIC: Only flush group when sentence ends
        if sentence_end_pattern.search(text):
            full_text = ' '.join(t for _, _, t in current_group).strip()
            group_start = current_group[0][0]
            group_end = current_group[-1][1]
            sentence_groups.append((group_start, group_end, full_text))
            current_group = []

    # Handle remaining segments after loop
    if current_group:
        full_text = ' '.join(t for _, _, t in current_group).strip()
        group_start = current_group[0][0]
        group_end = current_group[-1][1]
        sentence_groups.append((group_start, group_end, full_text))

    if not sentence_groups:
        st.error("No valid sentence groups found in the transcript.")
        st.stop()

    return sentence_groups


def convert_time(time_str: str) -> int:
    m, s = map(int, time_str.split(':'))
    return m * 60 + s

def convert_seconds_to_time(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m:02}:{s:02}"

async def generate_segment_audio(text: str, output_file: str, voice: str, rate: str):
    if not re.match(r"^[+-]\d+%$", rate):
        rate = "-10%"
        st.warning(f"Invalid rate format. Using default: {rate}")
    communicator = edge_tts.Communicate(text, voice, rate=rate)
    await communicator.save(output_file)

def run_generate_audio_for_segmentOLD(text: str, output_file: str, voice: str, rate: str):
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_segment_audio(text, output_file, voice, rate))

def generate_transcript(video_path: str) -> str:
    st.info("Generating transcript using Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    transcript_lines = []
    for segment in result["segments"]:
        start_min = int(segment["start"] // 60)
        start_sec = int(segment["start"] % 60)
        end_min = int(segment["end"] // 60)
        end_sec = int(segment["end"] % 60)
        text = segment["text"].strip().replace("\n", " ")
        transcript_lines.append(f"{start_min:01d}:{start_sec:02d} - {end_min:01d}:{end_sec:02d}: {text}")
    return "\n".join(transcript_lines)



def create_synchronized_audioOLD(sentence_groups, original_segments, translated_segments, voice: str, rate: str, progress_callback=None):
    """
    Generate synchronized audio for each sentence group using pre-translated text.
    Each sentence_group is a tuple (start_sec, end_sec, full_sentence).
    Debug entries include original sentence, translated text, target and actual durations.
    """
    from pydub import AudioSegment
    import shutil

    audio_segments = []
    debug_entries = []  # To store debug info for each group
    total_sentences = len(sentence_groups)

    # Clean previous segments folder if exists
    if os.path.exists(SEGMENTS_DIR):
        shutil.rmtree(SEGMENTS_DIR)
    os.makedirs(SEGMENTS_DIR, exist_ok=True)

    # Assume total_video_duration is from the end time of last sentence_group
    total_video_duration = sentence_groups[-1][1] * 1000  # in milliseconds

    # Track cumulative excess duration
    cumulative_excess = 0
    previous_speed = 1.0  #➕ Track speed between segments
    speech_context = []   #➕ For future context-aware processing

    for idx, (start, end, _) in enumerate(sentence_groups):
        segment_file = os.path.join(SEGMENTS_DIR, f"sentence_{idx}.mp3")

        # Get the original and translated text for this segment
        original = original_segments[idx]
        translated = translated_segments[idx]

        # Debugging logs
        st.write(f"Original: {original}")
        st.write(f"Translated: {translated}")

        # Ensure translation is not empty
        if not translated.strip():
            st.warning(f"Translation missing or invalid for sentence {idx+1}. Using original text.")
            translated = original

        # Adjust speaking rate dynamically for long segments
        length_ratio = len(translated) / len(original) if len(original) > 0 else 1
        if length_ratio > 1.2:  # If translation is 20% longer
            adjustment = int(rate[:-1]) - int((length_ratio - 1) * 10)
            adjusted_rate = f"{adjustment}%"
            st.warning(f"Adjusting speaking rate to {adjusted_rate} for segment {idx+1}.")
            run_generate_audio_for_segment(translated, segment_file, voice, adjusted_rate)
        else:
            # Generate audio for the translated text with the default rate
            run_generate_audio_for_segment(translated, segment_file, voice, rate)

        # Check if the audio file was generated successfully
        if not os.path.exists(segment_file) or os.path.getsize(segment_file) == 0:
            raise FileNotFoundError(f"Audio generation failed for sentence {idx+1}")

        segment_audio = AudioSegment.from_file(segment_file)

        # Calculate target duration from transcript timing (in ms)
        target_duration_ms = (end - start) * 1000
        current_duration = len(segment_audio)

        # Allow minor mismatches without adjustment
        tolerance_ms = 200
        if abs(current_duration - target_duration_ms) <= tolerance_ms:
            pass
        else:
            if current_duration < target_duration_ms:
                # Add silence to match the target duration
                silence_duration = target_duration_ms - current_duration - cumulative_excess
                silence_duration = max(0, silence_duration)  # Ensure no negative duration
                silence = AudioSegment.silent(duration=silence_duration)
                segment_audio += silence
                cumulative_excess = 0  # Reset excess after compensation
            elif current_duration > target_duration_ms:
                # Trim the audio to match the target duration
                segment_audio = segment_audio[:target_duration_ms]
                cumulative_excess += current_duration - target_duration_ms

        audio_segments.append(segment_audio)

        # Save debug info for this segment
        debug_entries.append(
            f"Segment {idx+1} (start: {start}s, end: {end}s):\n"
            f"**Original:** {original}\n"
            f"**Translated:** {translated}\n"
            f"**Target duration:** {target_duration_ms/1000:.2f}s, "
            f"**Audio duration:** {current_duration/1000:.2f}s, "
            f"**Cumulative excess:** {cumulative_excess/1000:.2f}s"
        )

        if progress_callback:
            progress = (idx + 1) / total_sentences * 80
            progress_callback(progress)

    # Combine all audio segments
    final_audio = sum(audio_segments)
    final_duration = len(final_audio)

    # Allow a small tolerance for mismatches (e.g., ±500ms)
    tolerance_ms = 500

    if final_duration < total_video_duration - tolerance_ms:
        # Add silence to match the total video duration
        silence = AudioSegment.silent(duration=total_video_duration - final_duration)
        final_audio += silence
    elif final_duration > total_video_duration + tolerance_ms:
        # Redistribute excess duration across all segments
        excess_duration = final_duration - total_video_duration
        st.warning(
            f"Final audio exceeds total video duration by "
            f"{excess_duration / 1000:.2f}s. Redistributing excess duration."
        )
        adjustment_ratio = excess_duration / len(audio_segments)
        adjusted_segments = []
        for segment in audio_segments:
            adjusted_duration = len(segment) - adjustment_ratio
            adjusted_segments.append(segment[:max(0, int(adjusted_duration))])
        final_audio = sum(adjusted_segments)

    # Export the final audio file
    final_audio.export(FINAL_AUDIO_FILE, format="mp3")

    # Write debug log file
    debug_log_path = create_translation_log(debug_entries)
    if not debug_log_path:
        st.warning("Debug log file could not be created.")

    # Clean up temporary segments folder
    if os.path.exists(SEGMENTS_DIR):
        shutil.rmtree(SEGMENTS_DIR)

    st.success("Final synchronized audio generated!")
    return FINAL_AUDIO_FILE, debug_log_path


def save_transcript(transcript_text: str, filename: str = "transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    st.info(f"Transcript saved to {filename}")

def merge_audio_with_video(video_path: str, audio_path: str):
    try:
        st.info("Merging audio with video...")
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        output_video_path = OUTPUT_VIDEO
        video.set_audio(audio).write_videofile(output_video_path, codec="libx264", audio_codec="aac")
        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
            raise RuntimeError("Merged video file is missing or invalid.")
        return output_video_path
    except Exception as e:
        st.error(f"Failed to merge audio with video: {e}")
        raise

def translate_with_openai(text: str, api_key: str, target_language: str = "fr") -> str:
    try:
        openai.api_key = api_key
        prompt = f"""You are a professional translator specializing in ERP Cloud Fusion systems.
        Translate the following text into {target_language}, ensuring that technical terms 
        and user interface elements are accurately translated in the context of ERP Cloud Fusion.

        Only return the translated sentence without introductory phrases. 
        Do not add anything beyond the translation itself.

        Text: {text}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"OpenAI Translation failed: {e}")
        return text

def validate_transcript_format(transcript: str):
    for line in transcript.splitlines():
        line = re.sub(r'^Texte\s*:\s*', '', line)
        if not re.match(r'^\d+:\d+\s*-\s*\d+:\d+:\s*.+$', line):
            st.warning(f"Invalid transcript line format: {line}")

def translate_with_ollama(text: str, model: str = "7shi/llama-translate:8b-q4_K_M", output_file: str = "ollama_response.txt") -> str:
    try:
        response = ollama.generate(
            model=model,
            prompt = (
                f"Translate the following text into French, ensuring technical terms and UI elements "
                f"are accurately translated in the context of ERP Cloud Fusion.\n\n"
                f"Only return the translated sentence with no extra formatting or commentary.\n\n"
                f"Text: {text}"
            )
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Ollama Response:\n")
            f.write(str(response))
        if "response" not in response:
            st.error("Unexpected Ollama response format.")
            return text
        translated_text = response["response"].strip()
        if not translated_text:
            st.error("Ollama returned an empty or invalid response.")
            return text
        return translated_text
    except Exception as e:
        st.error(f"Ollama Translation failed: {e}")
        return text

import numpy as np
from pydub import AudioSegment
import os
import shutil
import subprocess

# +++ NEW HELPER FUNCTIONS +++

def adjust_speed_with_psola(input_file: str, output_file: str, speed_factor: float) -> None:
    """High-quality speed adjustment with pitch preservation"""
    try:
        # Use temporary file for processing
        temp_file = output_file.replace(".mp3", "_temp.mp3")
        
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', input_file,
            '-af', f'rubberband=pitch=1.0:tempo={speed_factor:.2f}',
            temp_file
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Replace original file after successful processing
        os.replace(temp_file, output_file)
        
    except subprocess.CalledProcessError as e:
        # Clean up temporary file if exists
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise RuntimeError(f"PSOLA adjustment failed: {e.stderr.decode()}") from e







def calculate_speed_factor(target_duration: float, 
                          actual_duration: float, 
                          previous_speed: float = 1.0) -> float:
    """Calculate smoothed speed factor with constraints"""
    if actual_duration <= 0:
        return 1.0  # Fallback to prevent division by zero
    
    raw_speed = target_duration / actual_duration
    MAX_CHANGE = 0.2  # Maximum 20% change between consecutive segments
    
    # Apply bounded rate change
    bounded_speed = previous_speed * (1 + np.clip(
        (raw_speed/previous_speed - 1),
        -MAX_CHANGE, 
        MAX_CHANGE
    ))
    
    # Maintain natural sounding speed limits
    return np.clip(bounded_speed, 0.85, 1.15)  # 15% slower to 15% faster

# +++ UPDATED CORE FUNCTIONS +++
def create_synchronized_audio(sentence_groups, original_segments, translated_segments, 
                             voice: str, rate: str, progress_callback=None):
    """
    Enhanced version with psychoacoustic speed adjustment and smooth transitions
    Returns: (audio_file_path, debug_log_path)
    """
    from pydub import AudioSegment
    
    audio_segments = []
    debug_entries = []
    total_video_duration = sentence_groups[-1][1] * 1000  # ms
    cumulative_excess = 0
    previous_speed = 1.0  # Track speed between segments

    # Clean working directory
    if os.path.exists(SEGMENTS_DIR):
        shutil.rmtree(SEGMENTS_DIR)
    os.makedirs(SEGMENTS_DIR, exist_ok=True)

    for idx, (start, end, _) in enumerate(sentence_groups):
        segment_file = os.path.join(SEGMENTS_DIR, f"seg_{idx}.mp3")
        original = original_segments[idx]
        translated = translated_segments[idx]

        # --- Audio Generation Phase ---
        try:
            # Generate at natural speed first (no rate adjustment)
            run_generate_audio_for_segment(translated, segment_file, voice)
            
            # Load and measure generated audio
            segment_audio = AudioSegment.from_file(segment_file)
            actual_duration = len(segment_audio) / 1000  # seconds
            target_duration = end - start



            # Inside the segment processing loop:
            # --- Dynamic Speed Adjustment ---
            speed_factor = 1.0  # Default no adjustment
            need_adjustment = False

            # Check if adjustment is needed
            if abs(actual_duration - target_duration) > 0.1:
                speed_factor = calculate_speed_factor(
                    target_duration,
                    actual_duration,
                    previous_speed
                )
                
                # Validate meaningful adjustment
                if abs(speed_factor - 1.0) > MIN_SPEED_CHANGE:
                    need_adjustment = True

            if need_adjustment:
                # Apply PSOLA adjustment
                temp_file = segment_file.replace(".mp3", "_adjusted.mp3")
                try:
                    adjust_speed_with_psola(segment_file, temp_file, speed_factor)
                    os.replace(temp_file, segment_file)
                    previous_speed = speed_factor
                    segment_audio = AudioSegment.from_file(segment_file)
                    new_duration = len(segment_audio) / 1000
                    debug_msg = f"Adjusted: {speed_factor:.2f}x | Target: {target_duration:.2f}s → Actual: {new_duration:.2f}s"
                except Exception as e:
                    st.warning(f"Speed adjustment failed: {str(e)}")
                    debug_msg = f"Adjustment failed | Target: {target_duration:.2f}s | Original: {actual_duration:.2f}s"
            else:
                debug_msg = f"No adjustment needed | Target: {target_duration:.2f}s | Actual: {actual_duration:.2f}s"

            debug_entries.append(f"Segment {idx+1}: {debug_msg}")

            # --- Duration Matching (ALWAYS APPLIED) ---
            current_duration = len(segment_audio)
            target_ms = (end - start) * 1000


            # Soft matching with cumulative error distribution
            duration_diff = current_duration - target_ms
            if abs(duration_diff) > 50:  # 50ms tolerance
                cumulative_excess += duration_diff
                if cumulative_excess > 0:
                    # Trim future silence
                    segment_audio = segment_audio[:target_ms]
                else:
                    # Add minimal silence
                    segment_audio += AudioSegment.silent(
                        duration=-cumulative_excess,
                        frame_rate=segment_audio.frame_rate
                    )

            audio_segments.append(segment_audio)

            if progress_callback:
                progress = (idx + 1) / len(sentence_groups) * 90
                progress_callback(progress)

        except Exception as e:
            st.error(f"Failed processing segment {idx+1}: {str(e)}")
            raise

    # --- Final Assembly ---
    try:
        final_audio = sum(audio_segments)
        final_duration = len(final_audio)
        
        # Final duration alignment (max 100ms tolerance)
        if abs(final_duration - total_video_duration) > 100:
            silence_needed = total_video_duration - final_duration
            if silence_needed > 0:
                final_audio += AudioSegment.silent(silence_needed)
            else:
                final_audio = final_audio[:total_video_duration]

        final_audio.export(FINAL_AUDIO_FILE, format="mp3")
    except Exception as e:
        st.error(f"Final assembly failed: {str(e)}")
        raise

    # Cleanup and return
    debug_log_path = create_translation_log(debug_entries)
    if os.path.exists(SEGMENTS_DIR):
        shutil.rmtree(SEGMENTS_DIR)
        
    return FINAL_AUDIO_FILE, debug_log_path

def run_generate_audio_for_segment(text: str, output_file: str, voice: str):
    """Generate base audio at natural speaking pace"""
    nest_asyncio.apply()
    
    async def _generate():
        try:
            # Generate at natural speed (-0% adjustment)
            comm = edge_tts.Communicate(text, voice)
            await comm.save(output_file)
            
            # Validate output
            if os.path.getsize(output_file) < 1024:  # 1KB sanity check
                raise ValueError("Generated audio too small")
        except Exception as e:
            st.error(f"Audio generation failed: {str(e)}")
            raise

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_generate())

# --- Streamlit UI --- 

st.title("Video Translator & Voice Generator")
st.sidebar.header("Settings")

# Constants and settings (should match your full code)
VOICE_CHOICES = ["fr-CA-SylvieNeural", "fr-FR-DeniseNeural", "fr-CA-CHantalNeural"]
DEFAULT_VOICE = VOICE_CHOICES[0]
DEFAULT_RATE = "-10%"
FINAL_AUDIO_FILE = "final_voice.mp3"
OUTPUT_VIDEO = "translated_video.mp4"

# Initialize session state variables if not already present
if "original_transcript" not in st.session_state:
    st.session_state["original_transcript"] = ""
if "transcript_saved" not in st.session_state:
    st.session_state["transcript_saved"] = False
if "translated_transcript" not in st.session_state:
    st.session_state["translated_transcript"] = ""

# Sidebar settings
voice = st.sidebar.selectbox("Voice", VOICE_CHOICES, index=0)
rate = st.sidebar.text_input("Speaking Rate", DEFAULT_RATE)
translation_model = st.sidebar.selectbox("Select Translation Model",
                                         options=["Ollama (Local)", "OpenAI (Cloud)"],
                                         index=0)
if translation_model == "OpenAI (Cloud)":
    openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
else:
    openai_api_key = None
if translation_model == "Ollama (Local)":
    st.sidebar.info("Translations will be performed using the Ollama local model.")
elif translation_model == "OpenAI (Cloud)":
    st.sidebar.info("Translations will be performed using OpenAI's cloud-based model.")

# --- Video Upload & Preview ---
uploaded_file = st.file_uploader("Upload Video", type=["mp4"], accept_multiple_files=False)

if uploaded_file:
    try:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            video_path = tmp_file.name
            os.chmod(video_path, 0o644)
        with VideoFileClip(video_path) as preview_clip:
            if preview_clip.duration > 600:
                st.warning("Preview disabled for videos longer than 10 minutes")
            else:
                st.video(video_path, format="video/mp4")
    except Exception as e:
        st.error(f"Video handling failed: {e}")
        st.stop()
else:
    st.info("Please upload a video to get started.")

# --- Transcript Generation (Cached) ---
@st.cache_data(show_spinner=True)
def cached_generate_transcript(video_path):
    # Replace with a call to your heavy generate_transcript(video_path)
    time.sleep(2)  # simulate processing delay
    simulated_transcript = generate_transcript(video_path)
    return simulated_transcript

# --- Step 1: Process & Save Transcript ---
if not st.session_state["transcript_saved"]:
    st.subheader("1. Generate and Save Transcript")
    if st.button("Process Video"):
        try:
            st.info("Generating transcript...")
            transcript = cached_generate_transcript(video_path)
            st.session_state["original_transcript"] = transcript
            st.success("Transcript generated!")
        except Exception as e:
            st.error(f"Error during transcript generation: {e}")

    if st.session_state["original_transcript"]:
        with st.form(key="transcript_form", clear_on_submit=False):
            edited_transcript = st.text_area("Review & Edit Original Transcript (Optional):",
                                             value=st.session_state["original_transcript"],
                                             height=300)
            submitted = st.form_submit_button("Save Transcript")
            if submitted:
                st.session_state["original_transcript"] = edited_transcript
                st.session_state["transcript_saved"] = True
                st.success("Transcript saved successfully!")
                st.info("You can now proceed to translation and audio generation.")
else:
    st.info("Transcript already saved! Proceed to the next steps.")

# --- Step 1 (continued): Parse & Translate Transcript ---
# --- Step 2: Parse & Translate Transcript or Use User-Provided Transcript ---
st.subheader("2. Parse and Translate Transcript or Use Your Own Translated Transcript")

# Option for the user to provide their own translated transcript
use_own_translated_transcript = st.checkbox("I have my own translated transcript")


if use_own_translated_transcript:
    # Allow the user to paste their translated transcript
    st.info("Paste your translated transcript below:")
    user_translated_transcript = st.text_area(
        "Paste Translated Transcript (with timestamps):",
        height=300,
        placeholder="Example:\n00:00 - 00:05: Votre texte traduit ici.\n00:05 - 00:10: Votre texte traduit ici.",
        key="user_translated_input"  # Add unique key
    )

    # Save the user-provided transcript
    if st.button("Save Translated Transcript", key="save_custom_transcript"):
        if not user_translated_transcript.strip():
            st.error("Translated transcript cannot be empty!")
        else:
            try:
                # Directly store the raw input first
                st.session_state.translated_transcript = user_translated_transcript
                
                # Validate format by parsing
                parsed_groups = parse_transcript(user_translated_transcript)
                
                # Rebuild with consistent formatting
                formatted_transcript = "\n".join(
                    f"{convert_seconds_to_time(start)} - {convert_seconds_to_time(end)}: {text.strip()}"
                    for start, end, text in parsed_groups
                )
                
                # Update with formatted version
                st.session_state.translated_transcript = formatted_transcript
                st.session_state.translated_segments = [text for _, _, text in parsed_groups]
                
                st.success("Translated transcript saved successfully!")
                st.experimental_rerun()  # Force UI update
                
            except Exception as e:
                st.error(f"Error parsing transcript: {str(e)}")
                st.session_state.translated_transcript = ""
else:
    # Proceed with the app's translation process
    try:
        st.info("Parsing transcript...")
        # Get sentence groups with original timestamps and text.
        sentence_groups = parse_transcript(st.session_state["original_transcript"])
        st.info("Translating transcript segment-by-segment...")

        # Process each segment individually to preserve full sentence meaning and timing.
        translated_segments = []
        for idx, (start, end, original_text) in enumerate(sentence_groups):
            st.info(f"Translating segment {idx+1} ({convert_seconds_to_time(start)} to {convert_seconds_to_time(end)})")
            
            # Choose the translation method based on user selection.
            if translation_model == "Ollama (Local)":
                translated_text = clean_translation(translate_with_ollama(original_text))
            elif translation_model == "OpenAI (Cloud)":
                if not openai_api_key:
                    st.error("OpenAI API key is required for translation with OpenAI.")
                    st.stop()
                translated_text = clean_translation(translate_with_openai(original_text, openai_api_key, target_language="fr"))

            else:
                # Fallback if nothing is specified (should not occur)
                translated_text = original_text

            # Ensure that we have a non-empty translation.
            if not translated_text.strip():
                st.warning(f"Translation failed for segment {idx+1}. Using original text.")
                translated_text = original_text

            translated_segments.append(translated_text)

        # Rebuild final translated transcript with original time codes.
        translated_transcript = "\n".join(
            f"{convert_seconds_to_time(start)} - {convert_seconds_to_time(end)}: {trans_seg}"
            for (start, end, _), trans_seg in zip(sentence_groups, translated_segments)
        )

        # Update session state.
        st.session_state["translated_transcript"] = translated_transcript

        # Allow user to review and optionally edit the translated transcript.
        with st.form(key="translated_transcript_form", clear_on_submit=False):
            edited_translated_transcript = st.text_area("Review & Edit Translated Transcript (Optional):",
                                                         value=st.session_state["translated_transcript"],
                                                         height=300)
            submitted_translated = st.form_submit_button("Save Translated Transcript")
            if submitted_translated:
                st.session_state["translated_transcript"] = edited_translated_transcript
                st.success("Translated transcript saved!")
        st.success("Transcript processing and translation completed! You can now generate audio and merge with video.")

        # Optionally, save the translated transcript to a file.
        translated_transcript_file = "translated_transcript.txt"
        with open(translated_transcript_file, "w", encoding="utf-8") as f:
            f.write(st.session_state["translated_transcript"])
        st.info(f"Translated transcript saved to {translated_transcript_file}")
        
    except Exception as e:
        st.error(f"Error during transcript parsing/translation: {e}")
# --- Step 2: Generate Audio and Merge with Video ---
st.subheader("3. Generate Audio and Merge with Video")
# This button should always be active if the transcript has been processed.
if st.button("Generate Audio and Merge with Video"):
    if not st.session_state.get("translated_transcript"):
        st.error("Please process and save the transcript first before generating audio.")
    else:
        try:
            st.info("Parsing edited translated transcript...")
            sentence_groups = parse_transcript(st.session_state["translated_transcript"])


            # Extract original and translated segments
            # Extract original and translated segments
            original_segments = [line.split(": ", 1)[1] for line in st.session_state["original_transcript"].splitlines()]

            # Extract translated segments and clean them further if 'Texte:' or 'Text:' exists
            translated_segments = []
            for line in st.session_state["translated_transcript"].splitlines():
                if "Texte:" in line:
                    # Extract only the content after 'Texte:'
                    translated_segments.append(line.split("Texte:", 1)[1].strip())
                elif "Text:" in line:
                    # Extract only the content after 'Text:'
                    translated_segments.append(line.split("Text:", 1)[1].strip())
                else:
                    # Use the full line if no 'Texte:' or 'Text:' is found
                    translated_segments.append(line.split(": ", 1)[1].strip())
                    
            st.info("Generating synchronized audio...")
            # Call the function
            audio_file, debug_log_path = create_synchronized_audio(sentence_groups, original_segments, translated_segments, voice, rate)



            
            #audio_file, debug_log_path = create_synchronized_audio(sentence_groups, translated_segments, voice, rate)

            st.info("Merging audio with video...")
            output_video_path = merge_audio_with_video(video_path, audio_file)

            st.session_state["output_audio_path"] = audio_file
            st.session_state["output_video_path"] = output_video_path
            st.session_state["debug_log_path"] = debug_log_path

            st.success("Audio and video processing completed successfully!")
            st.audio(audio_file, format="audio/mp3")
            st.video(output_video_path, format="video/mp4")
        except Exception as e:
            st.error(f"Error during audio/video processing: {e}")