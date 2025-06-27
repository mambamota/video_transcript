import whisper
from transformers import pipeline
from TTS.api import TTS

from pydub import AudioSegment
import numpy as np
# Nouveaux imports (testés et fonctionnels)
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

def translate_video(input_path, output_path):
    # 1. Extract audio
    video = VideoFileClip(input_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)

    # 2. Transcribe with Whisper (large model for accuracy)
    model = whisper.load_model("large-v3", device="cpu")
    result = model.transcribe(audio_path, language="en", fp16=False)  # Force FP32
    segments = result["segments"]

    # 3. Translate with NLLB (state-of-the-art translation)
    translator = pipeline("translation", 
                        model="facebook/nllb-200-3.3B",
                        src_lang="eng_Latn", 
                        tgt_lang="fra_Latn")
    
    # Context-aware translation with sentence grouping
    translated_text = []
    current_group = ""
    for seg in segments:
        if len(current_group) + len(seg['text']) < 500:
            current_group += " " + seg['text']
        else:
            translated_group = translator(current_group)[0]['translation_text']
            translated_text.append(translated_group)
            current_group = seg['text']
    if current_group:
        translated_text.append(translator(current_group)[0]['translation_text'])

    # 4. Generate French speech with natural flow
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
             progress_bar=False, gpu=False)
    
    # Split translation into natural speech chunks
    full_audio = AudioSegment.silent(duration=0)
    for i, text in enumerate(translated_text):
        tts.tts_to_file(text=text, 
                       speaker_wav="fr_speaker_ref.wav",  # Provide reference audio
                       language="fr",
                       file_path=f"temp_{i}.wav")
        
        chunk = AudioSegment.from_wav(f"temp_{i}.wav")
        full_audio += chunk

    # 5. Synchronize with video
    new_audio = AudioFileClip("final_audio.wav")
    final_video = video.set_audio(new_audio.set_duration(video.duration))
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Usage
translate_video("4.2.4_Configuration de la solution_Avr_10_Latest.mp4", "4.2.4_Configuration de la solution_Avr_10_Latest_fr.mp4")
