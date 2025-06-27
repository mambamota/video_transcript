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
from tempfile import NamedTemporaryFile


# --- Configuration ---
ffmpeg_path = which("ffmpeg")
if not ffmpeg_path:
    raise RuntimeError("ffmpeg not found. Please install ffmpeg first.")
print(f"✅ ffmpeg found at: {ffmpeg_path}")

SEGMENTS_DIR = "segments_temp"
os.makedirs(SEGMENTS_DIR, exist_ok=True)

VOICE_CHOICES = [
    "fr-CA-SylvieNeural", 
    "fr-FR-DeniseNeural", 
    "fr-CA-CHantalNeural"
]
DEFAULT_VOICE = VOICE_CHOICES[0]
DEFAULT_RATE = "-10%"
OUTPUT_VIDEO = "translated_video.mp4"
FINAL_AUDIO_FILE = "final_voice.mp3"


# --- Debugging Functions ---


from datetime import datetime

def create_translation_log(debug_entries: list) -> str:
    """Create a timestamped debug log file with debug_entries and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"translation_debug_{timestamp}.md"
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("# Translation Debug Log\n\n")
        for entry in debug_entries:
            f.write(entry + "\n---\n")
    return log_file_path


def write_log_entry(log_file, original: str, translated: str, notes: str = ""):
    """Write formatted entry to debug log"""
    entry = f"""
### Original:
{original}

### Translated:
{translated}

{notes}

---
"""
    log_file.write(entry)

# --- Core Functions ---

def chunk_text(text: str, max_length: int = 512) -> list:
    """
    Split the text into chunks that do not exceed max_length,
    preserving complete sentences. If a sentence fragment does not start
    with a capital letter, it is merged with the previous sentence.
    The text is first normalized by joining all lines.
    """
    import re
    # Normalize the text: join lines into a single string with a space
    normalized_text = " ".join(text.splitlines()).strip()
    
    # If the entire text fits in max_length, return it as a single chunk.
    if len(normalized_text) <= max_length:
        return [normalized_text]
    
    # Split text into sentences based on punctuation followed by whitespace.
    sentences = re.split(r'(?<=[.!?])\s+', normalized_text)
    
    # Merge any sentence fragment that does not start with a capital letter
    merged_sentences = []
    for s in sentences:
        if not s:
            continue
        if merged_sentences and not s[0].isupper():
            merged_sentences[-1] += " " + s
        else:
            merged_sentences.append(s)
    
    # Group merged sentences into chunks that do not exceed max_length.
    chunks = []
    current_chunk = ""
    for sentence in merged_sentences:
        if len(current_chunk) + len(sentence) > max_length:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def translate_text(text: str) -> str:
    """
    Translate text using Google Translator only, with improved error handling.
    """
    try:
        chunks = chunk_text(text, max_length=512)
        translated_chunks = []

        for chunk in chunks:
            # Clean chunk text
            clean_chunk = chunk.strip()
            if not clean_chunk:
                continue

            try:
                # Translate with Google
                translated = GoogleTranslator(source='auto', target='fr').translate(clean_chunk)
                if not translated.strip():
                    raise ValueError("Empty translation")
                translated_chunks.append(translated)
            except Exception as e:
                st.warning(f"Translation failed for chunk: {clean_chunk}. Using original text. Error: {e}")
                translated_chunks.append(clean_chunk)  # Fallback to original text

        return " ".join(translated_chunks)

    except Exception as e:
        st.error(f"Translation process failed: {e}")
        return text  # Return original text as fallback

def parse_transcript(transcript: str):
    """
    Groups timestamped text into complete sentences with timing info
    Returns: List of (start_sec, end_sec, full_sentence)
    """
    sentence_groups = []
    current_group = []
    sentence_end_pattern = r'[.!?](?:\s|$)'
    
    # First extract all timestamped segments
    base_segments = []
    for line in transcript.splitlines():
        match = re.search(r'(\d+:\d+)\s*-\s*(\d+:\d+):\s*(.+)$', line)
        if match:
            start = convert_time(match.group(1))
            end = convert_time(match.group(2))
            text = match.group(3).strip()
            base_segments.append((start, end, text))

    # Group into complete sentences
    for seg_start, seg_end, text in base_segments:
        current_group.append((seg_start, seg_end, text))
        
        # Check if text contains sentence ending
        if re.search(sentence_end_pattern, text):
            full_text = ' '.join(t for _, _, t in current_group)
            group_start = current_group[0][0]
            group_end = current_group[-1][1]
            sentence_groups.append((group_start, group_end, full_text))
            current_group = []

    # Add any remaining text
    if current_group:
        full_text = ' '.join(t for _, _, t in current_group)
        group_start = current_group[0][0]
        group_end = current_group[-1][1]
        sentence_groups.append((group_start, group_end, full_text))

    return sentence_groups


def convert_time(time_str: str) -> int:
    """Convert mm:ss to total seconds"""
    m, s = map(int, time_str.split(':'))
    return m * 60 + s



async def generate_segment_audio(text: str, output_file: str, voice: str, rate: str):
    """Generate TTS audio with rate validation"""
    # Validate rate format
    if not re.match(r"^[+-]\d+%$", rate):
        rate = "-10%"
        st.warning(f"Invalid rate format. Using default: {rate}")
    
    communicator = edge_tts.Communicate(text, voice, rate=rate)
    await communicator.save(output_file)


def run_generate_audio_for_segment(text: str, output_file: str, voice: str, rate: str):
    """Run async TTS generation"""
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_segment_audio(text, output_file, voice, rate))

def generate_transcript(video_path: str) -> str:
    """Generate transcript using Whisper"""
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
        transcript_lines.append(
            f"{start_min:01d}:{start_sec:02d} - {end_min:01d}:{end_sec:02d}: {text}"
        )
    return "\n".join(transcript_lines)

import glob


def create_synchronized_audio(sentence_groups, voice: str, rate: str, progress_callback=None):
    """
    Generate synchronized audio for each sentence group and log debug info.
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

    for idx, (start, end, sentence) in enumerate(sentence_groups):
        segment_file = os.path.join(SEGMENTS_DIR, f"sentence_{idx}.mp3")

        if not sentence.strip():
            raise ValueError(f"Empty sentence in group {idx+1}")

        # Translate using the updated translate_text() function
        translated = translate_text(sentence)
        if not translated.strip():
            st.warning(f"Translation failed for sentence {idx+1}. Using original text.")
            translated = sentence  # Fallback to original text

        # Generate audio for the translated text
        run_generate_audio_for_segment(translated, segment_file, voice, rate)
        if not os.path.exists(segment_file) or os.path.getsize(segment_file) == 0:
            raise FileNotFoundError(f"Audio generation failed for sentence {idx+1}")
        segment_audio = AudioSegment.from_file(segment_file)

        # Calculate target duration from transcript timing (in ms)
        target_duration_ms = (end - start) * 1000
        current_duration = len(segment_audio)

        # Allow minor mismatches without truncation
        tolerance_ms = 200
        if abs(current_duration - target_duration_ms) <= tolerance_ms:
            st.warning(
                f"Segment {idx+1} audio duration mismatch within tolerance: "
                f"{(current_duration - target_duration_ms) / 1000:.2f}s. Allowing mismatch."
            )
        else:
            # Apply adjustments or truncation as needed
            if current_duration < target_duration_ms:
                # Add silence, compensating for cumulative excess
                silence_duration = target_duration_ms - current_duration - cumulative_excess
                silence_duration = max(0, silence_duration)  # Ensure no negative duration
                silence = AudioSegment.silent(duration=silence_duration)
                segment_audio += silence
                cumulative_excess = 0  # Reset excess after compensation
            elif current_duration > target_duration_ms:
                # Calculate excess duration
                excess_duration = current_duration - target_duration_ms
                cumulative_excess += excess_duration
                st.warning(
                    f"Segment {idx+1} audio exceeds target duration by "
                    f"{excess_duration / 1000:.2f}s. Redistributing excess duration."
                )

        audio_segments.append(segment_audio)

        # Save debug info for this segment
        debug_entries.append(
            f"Segment {idx+1} (start: {start}s, end: {end}s):\n"
            f"**Original:** {sentence}\n"
            f"**Translated:** {translated}\n"
            f"**Target duration:** {target_duration_ms/1000:.2f}s, "
            f"**Audio duration:** {current_duration/1000:.2f}s"
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
        # Adjust each segment proportionally
        adjustment_ratio = excess_duration / len(audio_segments)
        adjusted_segments = []
        for segment in audio_segments:
            adjusted_duration = len(segment) - adjustment_ratio
            adjusted_segments.append(segment[:max(0, int(adjusted_duration))])
        final_audio = sum(adjusted_segments)

    # Export the final audio file
    final_audio.export(FINAL_AUDIO_FILE, format="mp3")

    # Validate final audio
    if not os.path.exists(FINAL_AUDIO_FILE):
        raise RuntimeError("Final audio file creation failed")
    if abs(len(AudioSegment.from_file(FINAL_AUDIO_FILE)) - total_video_duration) > 100:
        raise ValueError(f"Final audio duration mismatch: {len(final_audio)/1000:.1f}s vs video {total_video_duration/1000:.1f}s")

    # Write debug log file and return its path as additional info (could be used in the UI)
    debug_log_path = create_translation_log(debug_entries)

    # Clean up temporary segments folder
    if os.path.exists(SEGMENTS_DIR):
        shutil.rmtree(SEGMENTS_DIR)

    st.success("Final synchronized audio generated!")
    return FINAL_AUDIO_FILE, debug_log_path



def create_synchronized_audioOLD_METY(sentence_groups, voice: str, rate: str, progress_callback=None):
    """
    Generate synchronized audio for each sentence group and log debug info.
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

    for idx, (start, end, sentence) in enumerate(sentence_groups):
        segment_file = os.path.join(SEGMENTS_DIR, f"sentence_{idx}.mp3")

        if not sentence.strip():
            raise ValueError(f"Empty sentence in group {idx+1}")

        # Translate using the updated translate_text() function
        translated = translate_text(sentence)
        if not translated.strip():
            st.warning(f"Translation failed for sentence {idx+1}. Using original text.")
            translated = sentence  # Fallback to original text

        # Generate audio for the translated text
        run_generate_audio_for_segment(translated, segment_file, voice, rate)
        if not os.path.exists(segment_file) or os.path.getsize(segment_file) == 0:
            raise FileNotFoundError(f"Audio generation failed for sentence {idx+1}")
        segment_audio = AudioSegment.from_file(segment_file)

        # Calculate target duration from transcript timing (in ms)
        target_duration_ms = (end - start) * 1000
        current_duration = len(segment_audio)

        # Adjust segment audio length to match target duration
        if current_duration < target_duration_ms:
            silence = AudioSegment.silent(duration=target_duration_ms - current_duration)
            segment_audio += silence
        else:
            segment_audio = segment_audio[:target_duration_ms]

        audio_segments.append(segment_audio)

        # Save debug info for this segment
        debug_entries.append(
            f"Segment {idx+1} (start: {start}s, end: {end}s):\n"
            f"**Original:** {sentence}\n"
            f"**Translated:** {translated}\n"
            f"**Target duration:** {target_duration_ms/1000:.2f}s, "
            f"**Audio duration:** {len(segment_audio)/1000:.2f}s"
        )

        if progress_callback:
            progress = (idx + 1) / total_sentences * 80
            progress_callback(progress)

    # Combine all audio segments
    final_audio = sum(audio_segments)
    final_duration = len(final_audio)

    # Force exact duration match with total video duration
    if final_duration < total_video_duration:
        silence = AudioSegment.silent(duration=total_video_duration - final_duration)
        final_audio += silence
    elif final_duration > total_video_duration:
        final_audio = final_audio[:total_video_duration]

    final_audio.export(FINAL_AUDIO_FILE, format="mp3")

    # Validate final audio
    if not os.path.exists(FINAL_AUDIO_FILE):
        raise RuntimeError("Final audio file creation failed")
    if abs(len(AudioSegment.from_file(FINAL_AUDIO_FILE)) - total_video_duration) > 100:
        raise ValueError(f"Final audio duration mismatch: {len(final_audio)/1000:.1f}s vs video {total_video_duration/1000:.1f}s")

    # Write debug log file and return its path as additional info (could be used in the UI)
    debug_log_path = create_translation_log(debug_entries)

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
    """Merge audio with video and handle errors robustly."""
    try:
        st.info("Merging audio with video...")
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Set audio to video and write output
        output_video_path = OUTPUT_VIDEO
        video.set_audio(audio).write_videofile(output_video_path, codec="libx264", audio_codec="aac")

        # Validate output file
        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
            raise RuntimeError("Merged video file is missing or invalid.")
        return output_video_path
    except Exception as e:
        st.error(f"Failed to merge audio with video: {e}")
        raise


# --- Streamlit UI ---


st.title("Video Translator & Voice Generator")
st.sidebar.header("Settings")

# --- Constants ---
VOICE_CHOICES = ["fr-CA-SylvieNeural", "fr-FR-DeniseNeural", "fr-CA-CHantalNeural"]
DEFAULT_VOICE = VOICE_CHOICES[0]
DEFAULT_RATE = "-10%"
SEGMENTS_DIR = "segments_temp"
FINAL_AUDIO_FILE = "final_voice.mp3"
OUTPUT_VIDEO = "translated_video.mp4"

# --- UI Elements ---
voice = st.sidebar.selectbox("Voice", VOICE_CHOICES, index=0)
rate = st.sidebar.text_input("Speaking Rate", DEFAULT_RATE)
uploaded_file = st.file_uploader("Upload Video", type=["mp4"], accept_multiple_files=False)

if uploaded_file:
    # --- Temporary File Handling ---
    try:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            video_path = tmp_file.name
            os.chmod(video_path, 0o644)  # Ensure read access
            
        # Safe preview for short videos
        with VideoFileClip(video_path) as preview_clip:
            if preview_clip.duration > 600:  # 10 minutes
                st.warning("Preview disabled for videos longer than 10 minutes")
            else:
                st.video(video_path, format="video/mp4")
                
    except Exception as e:
        st.error(f"Video handling failed: {str(e)}")
        st.stop()

    # ... [existing UI code above remains unchanged]

    if st.button("Process Video"):
        progress_bar = st.progress(0)
        debug_log_path = None
        audio_file = None
        output_file = None

        try:
            progress_bar.progress(0)
            # Clean previous runs (folder and files)
            for path in [SEGMENTS_DIR, FINAL_AUDIO_FILE, OUTPUT_VIDEO]:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            progress_bar.progress(5)

            if not (voice in VOICE_CHOICES and re.match(r"^[+-]\d{1,3}%$", rate)):
                rate = DEFAULT_RATE
                st.warning("Invalid settings, using defaults")

            progress_bar.progress(10)
            transcript = generate_transcript(video_path)
            save_transcript(transcript, "original_transcript.txt")
            progress_bar.progress(20)

            sentence_groups = parse_transcript(transcript)
            if not sentence_groups:
                raise ValueError("No transcript segments were found. Please check the transcript format.")

            total_video_duration = sentence_groups[-1][1]
            progress_bar.progress(20)

            debug_log_path = create_translation_log([])  # Create an empty log initially

            def handle_audio_progress(val):
                progress_bar.progress(20 + int(val * 0.6))

            st.info("Starting audio generation...")
            audio_result = create_synchronized_audio(
                sentence_groups,
                voice,
                rate,
                progress_callback=handle_audio_progress
            )
            if not audio_result or not isinstance(audio_result, tuple) or len(audio_result) != 2:
                raise RuntimeError("Audio generation failed. No valid audio file or debug log returned.")
            
            # Unpack result: audio file and debug log path
            audio_file, debug_log_path = audio_result
            if not audio_file or not os.path.exists(audio_file):
                raise FileNotFoundError("Generated audio file is missing or invalid.")
            st.info("Audio generation completed.")

            # Validate audio duration
            try:
                # Validate input files
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")
                if not os.path.exists(audio_file):
                    raise FileNotFoundError(f"Audio file not found: {audio_file}")
                if os.path.getsize(audio_file) == 0:
                    raise ValueError("Audio file is empty or invalid.")

                # Validate audio duration
                audio_segment = AudioSegment.from_file(audio_file)
                audio_duration = len(audio_segment) / 1000  # Convert to seconds
                if abs(audio_duration - total_video_duration) > 0.5:  # Allow up to 500ms mismatch
                    if audio_duration < total_video_duration:
                        silence = AudioSegment.silent(duration=(total_video_duration - audio_duration) * 1000)
                        audio_segment += silence
                    elif audio_duration > total_video_duration:
                        audio_segment = audio_segment[:total_video_duration * 1000]
                    audio_segment.export(audio_file, format="mp3")

                # Merge audio with video
                output_video_path = merge_audio_with_video(video_path, audio_file)

                # Validate output video
                if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
                    raise RuntimeError("Merged video file is missing or invalid.")

                st.success("Video processing completed successfully!")
                st.video(output_video_path, format="video/mp4")
                with open(output_video_path, "rb") as vid_file:
                    st.download_button(
                        label="Download Translated Video",
                        data=vid_file,
                        file_name=OUTPUT_VIDEO,
                        mime="video/mp4"
                    )
            except Exception as e:
                st.error(f"Processing Failed: {e}")

            progress_bar.progress(100)
            st.success("Process completed successfully!")
            
        except Exception as e:
            st.error(f"Processing Failed: {str(e)}")
            if debug_log_path and os.path.exists(debug_log_path):
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n## ERROR\n{str(e)}\n")
        finally:
            # Cleanup resources (if needed)
            for path in [audio_file, output_file]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        st.warning(f"Cleanup warning: {str(e)}")
            if os.path.exists(SEGMENTS_DIR):
                try:
                    shutil.rmtree(SEGMENTS_DIR)
                except Exception as e:
                    st.warning(f"Directory cleanup failed: {str(e)}")
            
            # Offer debug log for download if available
            if debug_log_path and isinstance(debug_log_path, str) and os.path.exists(debug_log_path):
                with open(debug_log_path, "rb") as f:
                    st.download_button(
                        "Download Debug Report",
                        data=f,
                        file_name=os.path.basename(debug_log_path),
                        mime="text/markdown"
                    )
            
            gc.collect()
            progress_bar.empty()
