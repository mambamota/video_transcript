import whisper
from transformers import pipeline
from TTS.api import TTS
from pydub import AudioSegment
import numpy as np
import os
import time
import logging
from datetime import datetime
#from moviepy.editor import VideoFileClip, AudioFileClip  # Import corrigé
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig


# Configuration des chemins
DEBUG_DIR = "/root/mmb/debug_files"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DEBUG_DIR, 'translation_debug.log')),
        logging.StreamHandler()
    ]
)

def translate_video(input_path, output_path, use_small_model=False, skip_existing=True, start_from_step=1):
    start_time = time.time()
    logging.info(f"🚀 Démarrage du processus pour {input_path}")
    
    # === Étape 1: Extraction audio ===
    if start_from_step <= 1:
        try:
            logging.info("1/5 - Extraction de l'audio...")
            video = VideoFileClip(input_path)
            audio_path = os.path.join(DEBUG_DIR, "temp_audio.wav")
            
            if not os.path.exists(audio_path):
                video.audio.write_audiofile(audio_path)
                logging.info(f"✅ Audio extrait → {audio_path} ({video.duration:.2f}s)")
            else:
                logging.info(f"⏩ Audio existant utilisé → {audio_path}")
        except Exception as e:
            logging.error(f"❌ Échec extraction audio: {str(e)}")
            return
    else:
        logging.info("⏩ Étape 1 ignorée (démarrage à partir de l'étape 5)")
        video = VideoFileClip(input_path)
        audio_path = os.path.join(DEBUG_DIR, "temp_audio.wav")

    # === Étape 2: Transcription Whisper ===
    if start_from_step <= 2:
        try:
            transcript_path = os.path.join(DEBUG_DIR, "transcription_raw.txt")
            segments = []
            
            if skip_existing and os.path.exists(transcript_path):
                logging.info("2/5 - Chargement transcription existante...")
                with open(transcript_path, "r") as f:
                    for line in f:
                        parts = line.strip().split("] ")
                        time_part = parts[0][1:].split("-")
                        text_part = " ".join(parts[1:])
                        segments.append({
                            'start': float(time_part[0]),
                            'end': float(time_part[1]),
                            'text': text_part
                        })
                logging.info(f"⏩ Transcription chargée → {transcript_path}")
            else:
                logging.info("2/5 - Chargement modèle Whisper...")
                model_size = "base" if use_small_model else "large-v3"
                model = whisper.load_model(model_size, device="cpu")
                logging.info(f"✅ Modèle Whisper chargé ({model_size})")
                
                logging.info("Démarrage transcription...")
                result = model.transcribe(audio_path, language="en", fp16=False)
                segments = result["segments"]
                
                with open(transcript_path, "w") as f:
                    for seg in segments:
                        f.write(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}\n")
                logging.info(f"✅ Transcription terminée → {transcript_path}")
        except Exception as e:
            logging.error(f"❌ Échec transcription: {str(e)}")
            return
    else:
        logging.info("⏩ Étape 2 ignorée (démarrage à partir de l'étape 5)")

    # === Étape 3: Traduction NLLB ===
    if start_from_step <= 3:
        try:
            translation_path = os.path.join(DEBUG_DIR, "translation_output.txt")
            translated_text = []
            
            if skip_existing and os.path.exists(translation_path):
                logging.info("3/5 - Chargement traduction existante...")
                with open(translation_path, "r") as f:
                    translated_text = [line.strip() for line in f]
                logging.info(f"⏩ Traduction chargée → {translation_path}")
            else:
                logging.info("3/5 - Chargement modèle NLLB...")
                translator = pipeline(
                    "translation",
                    model="facebook/nllb-200-3.3B",
                    src_lang="eng_Latn",
                    tgt_lang="fra_Latn"
                )
                logging.info("✅ Modèle NLLB chargé")
                
                current_group = ""
                total_chars = 0
                
                logging.info("Démarrage traduction...")
                for i, seg in enumerate(segments):
                    seg_text = seg['text'].strip()
                    if len(current_group) + len(seg_text) < 500:
                        current_group += " " + seg_text
                    else:
                        start_t = time.time()
                        translated = translator(current_group)[0]['translation_text']
                        translated_text.append(translated)
                        total_chars += len(current_group)
                        current_group = seg_text
                        logging.info(f"Traduit segment {i} ({len(current_group)}c) → {time.time()-start_t:.1f}s")
                
                if current_group:
                    translated_text.append(translator(current_group)[0]['translation_text'])
                
                with open(translation_path, "w") as f:
                    f.write("\n".join(translated_text))
                logging.info(f"✅ Traduction terminée → {translation_path}")
        except Exception as e:
            logging.error(f"❌ Échec traduction: {str(e)}")
            return
    else:
        logging.info("⏩ Étape 3 ignorée (démarrage à partir de l'étape 5)")

    # === Étape 4: Synthèse vocale TTS ===
    if start_from_step <= 4:
        try:
            add_safe_globals([XttsConfig, XttsAudioConfig])
            from TTS.api import TTS

            logging.info("4/5 - Initialisation TTS...")
            tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=False
            ).to("cpu")
            logging.info("✅ Modèle TTS initialisé")
                
            full_audio = AudioSegment.silent(duration=0)
            logging.info("Démarrage génération audio...")
            
            translation_path = os.path.join(DEBUG_DIR, "translation_output.txt")
            with open(translation_path, "r") as f:
                translated_text = [line.strip() for line in f]
            
            for i, text in enumerate(translated_text):
                start_t = time.time()
                temp_path = os.path.join(DEBUG_DIR, f"temp_{i}.wav")
                tts.tts_to_file(
                    text=text,
                    speaker_wav="fr_speaker_ref.wav",
                    language="fr",
                    file_path=temp_path
                )
                chunk = AudioSegment.from_wav(temp_path)
                full_audio += chunk
                logging.info(f"Généré segment {i} ({len(text)}c) → {time.time()-start_t:.1f}s")
            
            final_audio_path = os.path.join(DEBUG_DIR, "final_audio.wav")
            full_audio.export(final_audio_path, format="wav")
            logging.info(f"✅ Audio final généré → {final_audio_path}")
        except Exception as e:
            logging.error(f"❌ Échec synthèse vocale: {str(e)}")
            return
    else:
        logging.info("⏩ Étape 4 ignorée (démarrage à partir de l'étape 5)")

    # === Étape 5: Synchronisation vidéo ===
    try:
        logging.info("5/5 - Synchronisation audio/vidéo...")
        
        # Charger l'audio généré
        final_audio_path = os.path.join(DEBUG_DIR, "final_audio.wav")
        new_audio = AudioFileClip(final_audio_path)
        
        # Synchronisation avec MoviePy v2.0+
        final_video = video.with_audio(new_audio)
      
        
        # Export avec paramètres optimisés
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            bitrate="5000k",
            threads=4,
            logger='bar' if logging.getLogger().level == logging.INFO else None
        )
        logging.info(f"🎉 Vidéo finale sauvegardée → {output_path}")
    except Exception as e:
        logging.error(f"❌ Échec synchronisation: {str(e)}")
        return

    # === Nettoyage ===
    try:
        for f in os.listdir(DEBUG_DIR):
            if f.startswith("temp_"):
                os.remove(os.path.join(DEBUG_DIR, f))
        logging.info("🧹 Nettoyage fichiers temporaires terminé")
    except Exception as e:
        logging.warning(f"⚠ Erreur nettoyage: {str(e)}")

    logging.info(f"⏱ Temps total: {time.time()-start_time:.1f}s")

# Exécution avec options
input_video = "/root/mmb/4.2.4_Configuration de la solution_Avr_10_Latest.mp4"
output_video = "/root/mmb/4.2.4_Configuration de la solution_Avr_10_Latest_fr.mp4"
translate_video(
    input_video, 
    output_video,
    use_small_model=True,
    skip_existing=True,  # Ne pas ignorer les fichiers existants
    start_from_step=5  # Démarre directement à l'étape 5
)
