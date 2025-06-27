import os
import re
import ffmpeg
import pydub
import pysrt
import time
import asyncio
import edge_tts
import numpy as np
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from faster_whisper import WhisperModel
from shutil import which
from datetime import datetime
import tempfile
import aiohttp
import ssl
import random
from pydub.silence import detect_nonsilent
from typing import List, Dict

# ============== Configuration ==============
FFMPEG_PATH = which("ffmpeg")
INPUT_VIDEO = "to translate/4.2.4_Configuration de la solution_Avr_10_Latest.mp4"
BASE_NAME = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"{BASE_NAME}_run_{TIMESTAMP}"
MODEL_SIZE = "small"
USE_EDGE_TTS = True

# ============== Audio Processing Functions ==============
class AudioProcessor:
    @staticmethod
    def apply_speed_adjustment(raw_audio: AudioSegment, speed_setting: str) -> AudioSegment:
        """Apply speed adjustment with duration compensation"""
        speed_factor = 1 + (int(speed_setting.strip('%')) / 100)
        original_duration = len(raw_audio)
        
        # Speed adjustment with crossfade to avoid clicks
        sped_up = raw_audio.speedup(
            playback_speed=speed_factor,
            chunk_size=150,
            crossfade=25
        )
        
        # Calculate compensation
        new_duration = len(sped_up)
        compensation_ms = original_duration - new_duration
        
        if compensation_ms > 0:
            return sped_up + AudioSegment.silent(duration=compensation_ms)
        return sped_up

    @staticmethod
    def generate_phrase_audio(text: str, voice_speed: str) -> AudioSegment:
        """Generate phrase audio with natural ending detection"""
        async def _generate():
            communicate = edge_tts.Communicate(text)
            return await communicate
        
        raw_audio = asyncio.run(_generate()).audio
        processed = AudioProcessor.apply_speed_adjustment(raw_audio, voice_speed)
        
        # Detect natural speech endings
        non_silent = detect_nonsilent(
            processed, 
            min_silence_len=50,
            silence_thresh=processed.dBFS - 16
        )
        
        if non_silent:
            end_pad = 150  # Minimum ending padding
            new_end = max(non_silent[-1][1] + end_pad, len(processed))
            return processed[:new_end]
        return processed

# ============== Timing Synchronization ==============
class SyncValidator:
    @staticmethod
    def validate_segment_timing(original_duration: float, translated_audio: AudioSegment) -> AudioSegment:
        """Ensure audio duration matches video segment duration"""
        audio_duration = len(translated_audio) / 1000  # Convert ms to seconds
        drift = original_duration - audio_duration
        
        if abs(drift) > 0.5:  # 500ms tolerance
            compensation_ms = int(drift * 1000)
            if compensation_ms > 0:
                return translated_audio + AudioSegment.silent(duration=compensation_ms)
            else:
                return translated_audio[:compensation_ms]
        return translated_audio

    @staticmethod
    def calculate_phrase_timings(phrases: List[str], silences: List[float]) -> List[Dict]:
        """Calculate precise timings for each phrase"""
        timings = []
        current_time = 0.0
        
        for i, phrase in enumerate(phrases):
            # Generate temporary audio to measure duration
            with tempfile.NamedTemporaryFile() as tmp:
                asyncio.run(edge_tts.Communicate(phrase).save(tmp.name))
                audio = AudioSegment.from_file(tmp.name)
                duration = len(audio) / 1000  # Convert ms to seconds
            
            timings.append({
                "start": current_time,
                "end": current_time + duration,
                "phrase": phrase
            })
            
            # Add silence after phrase if not last element
            if i < len(silences):
                current_time += duration + (silences[i] / 1000)
        
        return timings

# ============== Main Processing Pipeline ==============
class TranslationPipeline:
    def __init__(self):
        self.debug_log = []
        
    async def process_segment(self, segment: Dict, output_path: str):
        """Process single video segment with sync validation"""
        # Generate translated audio
        translated_audio = await self._generate_translated_audio(segment)
        
        # Validate timing
        original_duration = segment["end"] - segment["start"]
        validated_audio = SyncValidator.validate_segment_timing(
            original_duration, translated_audio
        )
        
        # Save debug information
        self._log_segment_debug(segment, translated_audio, validated_audio)
        
        # Export final audio
        validated_audio.export(output_path, format="wav")
    
    async def _generate_translated_audio(self, segment: Dict) -> AudioSegment:
        """Generate translated audio with proper timing"""
        combined_audio = AudioSegment.silent(segment["pre_silence"])
        
        for i, phrase in enumerate(segment["phrases"]):
            # Generate phrase audio
            phrase_audio = AudioProcessor.generate_phrase_audio(
                phrase, segment["speed"]
            )
            
            # Add inter-phrase silence
            if i > 0 and i <= len(segment["inter_silences"]):
                combined_audio += AudioSegment.silent(
                    segment["inter_silences"][i-1]
                )
            
            combined_audio += phrase_audio
        
        # Add post-silence
        combined_audio += AudioSegment.silent(segment["post_silence"])
        
        return combined_audio
    
    def _log_segment_debug(self, segment, translated_audio, validated_audio):
      """Log segment debug information."""
      # Duration in seconds
      original_duration = segment["end"] - segment["start"]
      translated_duration = len(translated_audio) / 1000
      validated_duration = len(validated_audio) / 1000

      # Log entry
      log_entry = {
          "segment_start": segment["start"],
          "segment_end": segment["end"],
          "original_duration": original_duration,
          "translated_duration": translated_duration,
          "validated_duration": validated_duration,
          "voice_speed": segment["speed"],
          "pre_silence": segment["pre_silence"],
          "post_silence": segment["post_silence"],
          "inter_silences": segment["inter_silences"],
          "phrases": segment["phrases"],
      }
      self.debug_log.append(log_entry)

# ============== Helper Functions ==============
def sanitize_silences(silences: List[float]) -> List[float]:
    """Ensure silences are within valid range"""
    return [max(0, min(5000, s)) for s in silences]



def parse_review_file(review_path: str) -> List[Dict]:
    """Parse review file with sanity checks"""
    segments = []
    current_segment = {}
    segment_number = 0

    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Segment"):
                segment_number += 1
                # Save previous segment if it exists
                if current_segment:
                    segments.append(current_segment)
                # Start a new segment
                current_segment = {
                    'segment_number': segment_number,
                    'start': 0.0,
                    'end': 0.0,
                    'original': '',
                    'auto_translated': '',
                    'phrases': [],
                    'speed': "+0%",
                    'pre_silence': 0,
                    'post_silence': 100,
                    'inter_silences': [],
                    'decalage': 0
                }
                match = re.match(r"Segment (\d+) \(start: (\d+\.?\d*)s, end: (\d+\.?\d*)s\):", line)
                if match:
                    current_segment['segment_number'] = int(match.group(1))
                    current_segment['start'] = float(match.group(2))
                    current_segment['end'] = float(match.group(3))
            elif line.startswith("**Original:**"):
                current_segment['original'] = line.split('**Original:**')[1].strip()
            elif line.startswith("**Auto Translated:**"):
                current_segment['auto_translated'] = line.split('**Auto Translated:**')[1].strip()
            elif line.startswith("**Final Translation:**"):
                current_segment['phrases'] = split_french_phrases(line.split('**Final Translation:**')[1].strip())
            elif line.startswith("**Voice Speed:**"):
                try:
                    current_segment['speed'] = line.split('**Voice Speed:**')[1].strip()
                except:
                    print(f"Warning: Could not parse speed for segment {segment_number}. Setting to default +0%.")
                    current_segment['speed'] = "+0%"
            elif line.startswith("**Pre-Silence:**"):
                try:
                    current_segment['pre_silence'] = int(line.split('**Pre-Silence:**')[1].strip())
                except ValueError:
                    print(f"Warning: Could not parse pre_silence for segment {segment_number}. Setting to default 0.")
                    current_segment['pre_silence'] = 0
            elif line.startswith("**Post-Silence:**"):
                try:
                    current_segment['post_silence'] = int(line.split('**Post-Silence:**')[1].strip())
                except ValueError:
                    print(f"Warning: Could not parse post_silence for segment {segment_number}. Setting to default 100.")
                    current_segment['post_silence'] = 100
            elif line.startswith("**Inter-Phrase-Silence:**"):
                silences_str = line.split('**Inter-Phrase-Silence:**')[1].strip()
                if silences_str:
                    try:
                        current_segment['inter_silences'] = [int(s.strip()) for s in silences_str.split(',') if s.strip()]
                    except ValueError:
                        print(f"Warning: Could not parse inter_silences for segment {segment_number}. Setting to empty list.")
                        current_segment['inter_silences'] = []
                else:
                    current_segment['inter_silences'] = []
            elif line.startswith("**Décalage (local ms):**"):
                try:
                    current_segment['decalage'] = int(line.split('**Décalage (local ms):**')[1].strip())
                except:
                    print(f"Warning: Could not parse decalage for segment {segment_number}. Setting to default 0.")
                    current_segment['decalage'] = 0
        if current_segment:
            segments.append(current_segment)

    return segments




def split_french_phrases(text: str) -> List[str]:
    """Splits a French text into phrases using common punctuation marks."""
    # Split by periods, question marks, and exclamation points, but keep the delimiters.
    phrases = re.split(r"([.?!])", text)
    # Recombine the delimiters with the preceding text.
    # Here's the fix: convert phrases to an iterator explicitly
    phrases_iter = iter(phrases)
    phrases = [phrase + next(phrases_iter, '') for phrase in phrases]
    # Clean up: remove empty strings and strip whitespace.
    phrases = [p.strip() for p in phrases if p.strip()]
    return phrases

# ============== Main Execution ==============
async def main():
    # Initialize pipeline
    pipeline = TranslationPipeline()

    # Load review configuration
    review_file_path = "translation_review.txt"
    segments = parse_review_file(review_file_path)

    # Create output directory
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Process each segment
    tasks = []
    audio_segments = []
    for idx, segment in enumerate(segments):
        output_path = os.path.join(output_dir, f"segment_{idx+1}.wav")
        task = pipeline.process_segment(segment, output_path)
        tasks.append(task)
        audio_segments.append(output_path)

    # Run all tasks concurrently
    await asyncio.gather(*tasks)
    
    # Final video assembly
    input_video_path = INPUT_VIDEO
    original_video = VideoFileClip(input_video_path)
    
    # Load the audio segments
    audio_clips = [AudioFileClip(audio_path) for audio_path in audio_segments]
    
    # Prepare video segments
    video_segments = []
    current_time = 0
    for i, segment in enumerate(segments):
        start_time = segment["start"]
        end_time = segment["end"]
        
        # Extract the video clip for the segment
        video_clip = original_video.subclip(start_time, end_time)
        
        # Set the audio of the video clip
        video_clip = video_clip.set_audio(audio_clips[i])
        
        video_segments.append(video_clip)
        current_time = end_time

    # Concatenate all video segments
    final_video = concatenate_videoclips(video_segments)
    
    # Set the FPS to the original video's FPS
    final_video = final_video.set_fps(original_video.fps)

    # Write the final video file
    output_video_path = os.path.join(output_dir, "final_translated_video.mp4")
    final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac", temp_audiofile='temp-audio.m4a', remove_temp=True)
    
    print(f"✅ Final translated video created at: {output_video_path}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("Error: It seems you are trying to call asyncio.run from within an already running event loop.")
            print("This can happen in interactive environments like Jupyter notebooks.")
            print("Please try running this script from a regular Python environment.")
        else:
            raise e
