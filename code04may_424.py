import os
import re
import ffmpeg
import pysrt
import time
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
from faster_whisper import WhisperModel
from shutil import which
import nest_asyncio
from datetime import datetime
import tempfile
import asyncio
import edge_tts
import aiohttp
import ssl
import random
from pydub.silence import detect_nonsilent

nest_asyncio.apply()

# ----- Configuration -----
ffmpeg_path = which("ffmpeg")
if not ffmpeg_path:
    raise RuntimeError("ffmpeg not found. Please install ffmpeg first.")
print(f"✅ ffmpeg found at: {ffmpeg_path}")

input_video = "to translate/4.2.4_Configuration de la solution_Avr_10_Latest.mp4"
base_name = os.path.splitext(os.path.basename(input_video))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"{base_name}_run_{timestamp}"
model_size = "small"
update_existing = True

# For this version we rely on cloud-based Edge TTS.
USE_EDGE_TTS = True

# Files and paths
os.makedirs(output_dir, exist_ok=True)
input_video_name = os.path.splitext(os.path.basename(input_video))[0]
extracted_audio = os.path.join(output_dir, f"{input_video_name}-extracted-audio.wav")
subtitle_file_en = os.path.join(output_dir, f"{input_video_name}-english.srt")
translated_audio = os.path.join(output_dir, f"{input_video_name}-french.wav")
output_video = os.path.join(output_dir, f"{input_video_name}-french.mp4")
review_file = os.path.join(output_dir, "translation_review.txt")
debug_log_file = os.path.join(output_dir, "translation_debug_log.txt")

# ============== Helper Functions (extract_audio, transcribe, etc.) ==============
def extract_audio():
    try:
        (ffmpeg
         .input(input_video)
         .output(extracted_audio, ac=1, ar=16000)
         .overwrite_output()
         .run(capture_stdout=True, capture_stderr=True)
        )
        return extracted_audio
    except ffmpeg.Error as e:
        print("STDOUT:", e.stdout.decode("utf8"))
        print("STDERR:", e.stderr.decode("utf8"))
        raise

def transcribe(audio_path):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    language = info.language
    print(f"Detected language: {language}")
    transcript_segments = []
    for segment in segments:
        transcript_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    return language, transcript_segments

def time_to_subrip(seconds: float) -> pysrt.SubRipTime:
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return pysrt.SubRipTime(hours=hours, minutes=minutes, seconds=int(seconds), milliseconds=milliseconds)

def generate_subtitle_file(segments, output_path):
    subs = pysrt.SubRipFile()
    for i, segment in enumerate(segments, 1):
        sub = pysrt.SubRipItem(
            index=i,
            start=time_to_subrip(segment["start"]),
            end=time_to_subrip(segment["end"]),
            text=segment["text"]
        )
        subs.append(sub)
    subs.save(output_path, encoding="utf-8")
    return output_path

# ============== Translation & Review Functions ==============

def split_long_groups(groups, max_group_duration_secs):
    """
    For each group (list of SubRipItems), if its duration > max_group_duration_secs,
    split it at the *last* subtitle in that group whose text ends in punctuation
    (.,!? or comma) before the duration threshold.
    Falls back to a simple split if no such “safe” break exists.
    """
    new_groups = []
    for group in groups:
        start_s = group[0].start.ordinal / 1000
        end_s   = group[-1].end.ordinal   / 1000
        total   = end_s - start_s

        # if already shorter than threshold, keep it
        if total <= max_group_duration_secs:
            new_groups.append(group)
            continue

        # otherwise walk through, tracking safe_breaks
        temp = []
        temp_start = start_s
        last_safe_idx = None
        for idx, item in enumerate(group):
            temp.append(item)
            # mark this idx if it ends in punctuation or comma
            if re.search(r"[.,!?]$", item.text.strip()):
                last_safe_idx = idx

            current_end = item.end.ordinal / 1000
            if (current_end - temp_start) >= max_group_duration_secs:
                # if we have a safe break before or at idx, split there
                if last_safe_idx is not None:
                    # emit group up through last_safe_idx
                    safe_group = temp[: last_safe_idx+1 ]
                    new_groups.append(safe_group)
                    # restart temp from the items after safe_idx
                    temp = temp[last_safe_idx+1 :]
                    temp_start = temp[0].start.ordinal / 1000 if temp else current_end
                else:
                    # no safe break—just split at current idx
                    new_groups.append(temp)
                    temp = []
                    temp_start = current_end

                # reset safe marker
                last_safe_idx = None

        # anything left over
        if temp:
            new_groups.append(temp)

    return new_groups


def validate_audio_duration(original_segment, translated_audio):
    """Compares original video duration with generated audio"""
    video_dur = original_segment['end'] - original_segment['start']
    audio_dur = translated_audio.duration_seconds
    
    if abs(video_dur - audio_dur) > 0.5:  # 500ms tolerance
        compensation = (video_dur - audio_dur) * 1000  # ms
        if compensation > 0:
            return AudioSegment.silent(duration=compensation)
        else:
            return translated_audio[:int(compensation*1000)]  # ms to samples
    return translated_audio

def generate_phrase_audio(text, voice_speed):
    raw_audio = edge_tts.Communicate(text).audio
    processed = apply_speed_adjustment(raw_audio, voice_speed)
    
    # Detect and preserve natural phrase endings
    non_silent = detect_nonsilent(processed, min_silence_len=50, silence_thresh=-40)
    if non_silent:
        end_pad = 150  # Minimum ending padding
        new_end = max(non_silent[-1][1] + end_pad, len(processed))
        return processed[:new_end]
    return processed


def apply_speed_adjustment(raw_audio, speed_setting):
    speed_factor = 1 + (int(speed_setting.strip('%')) / 100)
    sped_up = raw_audio.speedup(
        playback_speed=speed_factor,
        chunk_size=150,
        crossfade=25
    )
    
    # Calculate duration difference
    original_dur = len(raw_audio)
    new_dur = len(sped_up)
    compensation = original_dur - new_dur
    
    if compensation > 0:
        return sped_up + AudioSegment.silent(duration=compensation)
    return sped_up


def parse_review_overrides(review_file_path):
    text   = open(review_file_path, "r", encoding="utf-8").read()
    # split on any line of 3+ hyphens
    blocks = re.split(r"(?m)^-{3,}\s*$", text)

    overrides = []
    for idx, blk in enumerate(blocks, start=1):
        blk = blk.strip()
        if not blk or blk.startswith("Translation Review File"):
            continue

        # defaults
        ft       = None
        vs       = "+0%"
        pre_ms   = 0.0
        post_ms  = 100.0
        inter_ms = []

        for line in blk.splitlines():
            if line.startswith("**Final Translation:**"):
                ft = line.split("**Final Translation:**",1)[1].strip()
            elif line.startswith("**Voice Speed:**"):
                vs = line.split("**Voice Speed:**",1)[1].strip()
            elif line.startswith("**Pre-Silence:**"):
                try: pre_ms = float(line.split("**Pre-Silence:**",1)[1])
                except: print(f"[Warn] Seg {idx}: bad Pre-Silence")
            elif line.startswith("**Post-Silence:**"):
                try: post_ms = float(line.split("**Post-Silence:**",1)[1])
                except: print(f"[Warn] Seg {idx}: bad Post-Silence")
            elif line.startswith("**Inter-Phrase-Silence:**"):
                            parts = line.split("**Inter-Phrase-Silence:**",1)[1].strip()
                            if parts:
                                try:
                                    # Force negative values to 0 and limit to 5000ms max
                                    raw = [float(x) for x in parts.split(",")]
                                    inter_ms = [ max(0, min(x, 5000)) for x in raw ]
                                except ValueError:
                                    print(f"[Warning] Segment {idx}: invalid Inter-Phrase-Silence list")
                                    inter_ms = []

        if ft is None:
            print(f"[Warn] Seg {idx}: no Final Translation—will use source text.")

        overrides.append({
            "final_translation":      ft,
            "voice_speed":            vs,
            "pre_silence":            pre_ms,
            "post_silence":           post_ms,
            "inter_phrase_silences":  inter_ms
        })

    print("Parsed review overrides:")
    for i,o in enumerate(overrides,1):
        print(f"  Seg {i}: final={'OK' if o['final_translation'] else '<none>'}, "
              f"speed={o['voice_speed']}, pre={o['pre_silence']}ms, post={o['post_silence']}ms, "
              f"inter={o['inter_phrase_silences']}")
    return overrides





def enforce_punctuation_boundaries(groups):
    """Ensure groups end with proper punctuation"""
    i = 0
    safe_punctuation = r"[.!?,;:]$"
    while i < len(groups):
        last_text = groups[i][-1].text.strip()
        if not re.search(safe_punctuation, last_text):
            if i+1 < len(groups):
                groups[i] += groups.pop(i+1)
            else:  # Add artificial pause for final group
                groups[i][-1].text += "."
        else:
            i += 1
    return groups



# ============== Audio Synchronization Functions ==============


def adjust_audio_duration(audio: AudioSegment, target_secs: float) -> AudioSegment:
    """
     Ajuste TTS clip pour qu'il tienne **exactement** dans target_secs :
     - Si l'audio est trop long, on le **tronque**.  
     - S'il est trop court, on ajoute du silence.  
    """
    target_ms = int(target_secs * 1000)
    curr_ms   = len(audio)
    if curr_ms > target_ms:
        # on coupe précisément à la durée allouée
        return audio[:target_ms]
    elif curr_ms < target_ms:
            # on complète par du silence
        return audio + AudioSegment.silent(duration=(target_ms - curr_ms))
    return audio


# ============== French Phrase Alignment Functions ==============
def split_french_phrases(text):
    phrases = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [phrase.strip() for phrase in phrases if phrase.strip()]

def calculate_phrase_weights(original_text, translated_phrases):
    fr_phrase_word_counts = [len(phrase.split()) for phrase in translated_phrases]
    total_fr_words = sum(fr_phrase_word_counts)
    if total_fr_words == 0:
        return [1 / len(translated_phrases)] * len(translated_phrases)
    return [count / total_fr_words for count in fr_phrase_word_counts]

# ============== TTS Functions: Edge TTS Only with Debug Logging ==============


def change_playback_speed(sound, speed=1.0):
    new_frame_rate = int(sound.frame_rate * speed)
    altered_sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
    return altered_sound.set_frame_rate(sound.frame_rate)

# ============== Updated Async Audio Generation Function ==============


def validate_audio_timing(original_duration, translated_segment):
    total_audio_time = (
        translated_segment["pre_silence"] 
        + sum(translated_segment["inter_phrase_silences"]) 
        + translated_segment["post_silence"] 
        + (translated_segment["audio"].duration_seconds * 1000)
    )
    
    if total_audio_time > original_duration * 1000:
        raise ValueError(f"Audio overflow: {total_audio_time}ms vs {original_duration*1000}ms")
    elif total_audio_time < original_duration * 1000 * 0.95:
        print(f"Warning: Audio underflow by {original_duration*1000 - total_audio_time}ms")



def adjust_review_file_based_on_debug_log(debug_log_path: str, review_file_path: str):
    """
    Pour chaque segment i :
      - Si décal_end est négatif de D ms, on ajoute D ms à post_silence
      - Si décal_start est positif de D ms, on ajoute D ms à pre_silence
    On réécrit ensuite le review_file avec ces nouvelles valeurs.
    """
    # 1) Parse le debug log
    decalages = {}  # idx -> (d_start, d_end)
    pattern = re.compile(r"Segment (\d+).*décal_start=(-?\d+)ms, décal_end=(-?\d+)ms")
    for line in open(debug_log_path, encoding="utf-8"):
        m = pattern.search(line)
        if m:
            idx = int(m.group(1))
            d_start, d_end = int(m.group(2)), int(m.group(3))
            decalages[idx] = (d_start, d_end)

    # 2) Lit tout le review file en mémoire
    text = open(review_file_path, encoding="utf-8").read()
    blocks = re.split(r"(?m)^-{3,}\s*$", text)

    out = []
    for blk in blocks:
        if not blk.strip() or blk.startswith("Translation Review File"):
            out.append(blk)
            continue

        # trouve le segment
        header = re.search(r"Segment\s+(\d+)\s+\(", blk)
        if not header:
            out.append(blk); continue
        idx = int(header.group(1))
        d_start, d_end = decalages.get(idx, (0, 0))

        # remplace les lignes Pre-Silence / Post-Silence
        def repl_pre(m):
            old = float(m.group(1))
            new = max(0.0, old + d_start)
            return f"**Pre-Silence:** {new:.0f}"
        blk = re.sub(r"\*\*Pre-Silence:\*\*\s*([0-9.]+)", repl_pre, blk)

        def repl_post(m):
            old = float(m.group(1))
            # si d_end<0, audio est trop long => il a fallu tronquer => on ne réduit pas post
            # si d_end>0, audio trop court => on ajoute
            new = max(0.0, old + d_end)
            return f"**Post-Silence:** {new:.0f}"
        blk = re.sub(r"\*\*Post-Silence:\*\*\s*([0-9.]+)", repl_post, blk)

        out.append(blk)

    # 3) Réécriture du fichier
    with open(review_file_path, "w", encoding="utf-8") as f:
        f.write("\n---\n".join(out))
    print(f"✅ Review file ajusté selon {debug_log_path}")


def generate_translation_review_file(
    source_path, review_file_path,
    from_lang="en", to_lang="fr",
    max_group_duration_secs: float = 25.0
):
    """
    1) On regroupe et on split/merge les sous-titres exactement
       comme le fera l'audio.
    2) On écrit un review file où l'on affiche :
       - phrase par phrase (la liste exacte via "- ")
       - pre / post silence
       - voice speed
       - start/end offset
       - inter-phrase silences (N–1 valeurs pour N phrases)
    L'utilisateur peut ensuite :
      * ajuster Final Translation, Voice Speed, Pre/Post-Silence,
        Start-Offset, End-Offset
      * modifier le nombre de phrases (le parser adaptera N–1 silences).
    """

    translator = GoogleTranslator(source=from_lang, target=to_lang)
    subs = pysrt.open(source_path)

    # 1) Regrouper par phrase (détection ponctuation en fin de sous-titre)
    sentence_end = re.compile(r"[.!?]\s*$")
    groups, cur = [], []
    for sub in subs:
        cur.append(sub)
        if sentence_end.search(sub.text):
            groups.append(cur); cur = []
    if cur:
        groups.append(cur)

    # 2) Éclatement des groupes trop longs
    def split_long(gs, max_s):
        out = []
        for g in gs:
            start, end = g[0].start.ordinal/1000, g[-1].end.ordinal/1000
            if end - start <= max_s:
                out.append(g)
            else:
                mid = len(g)//2
                out.extend([g[:mid], g[mid:]])
        return out
    groups = split_long(groups, max_group_duration_secs)

    # 3) Forcer ponctuation de fin de groupe
    i = 0
    safe_punct = re.compile(r"[.!?,;:]$")
    while i < len(groups):
        if not safe_punct.search(groups[i][-1].text.strip()):
            if i+1 < len(groups):
                groups[i] += groups.pop(i+1)
                continue
            else:
                groups[i][-1].text += "."
        i += 1

    # 4) Écriture du fichier de review
    with open(review_file_path, "w", encoding="utf-8") as f:
        f.write("Translation Review File\n")
        f.write("Le découpage en phrases ci-dessous est **celui utilisé** en TTS.\n")
        f.write("Ajustez si besoin **Final Translation**, **Voice Speed**, **Pre/Post-Silence**, "
                "**Start-Offset:**, **End-Offset:**, **Inter-Phrase-Silence:**\n")
        f.write("mais **ne touchez pas** la liste des phrases (lignes qui commencent par '- ').\n")
        f.write("----------------------------------------------------------------\n\n")

        for idx, group in enumerate(groups, 1):
            # Calcul des temps
            start_s = group[0].start.ordinal / 1000
            end_s   = group[-1].end.ordinal   / 1000

            # Texte original + auto-traduit
            original = " ".join(s.text for s in group)
            auto_tr  = translator.translate(text=original)

            # Découpage initial en phrases (on ne réécrit pas ces lignes, mais on calcule N)
            phrases = re.split(r"(?<=[.!?])\s+(?=[A-ZÀÂÉÈÊËÎÏÔŒÙÛÜ])", auto_tr)
            phrases = [p.strip() for p in phrases if p.strip()]

            # Préparer la liste par défaut des silences internes = N–1 × 0 ms
            n = len(phrases)
            inter_silences = ",".join("0" for _ in range(max(0, n-1)))

            # Valeurs par défaut
            pre_ms, post_ms = 0, 100
            start_offset, end_offset = 0, 0
            voice_speed = "+0%"

            # Écriture du segment
            f.write(f"Segment {idx} (start: {start_s:.2f}s, end: {end_s:.2f}s)\n")
            f.write(f"**Original:** {original}\n")
            f.write(f"**Auto Translated:** {auto_tr}\n")
            f.write(f"**Final Translation:** {auto_tr}\n")
            f.write(f"**Voice Speed:** {voice_speed}\n")
            f.write(f"**Pre-Silence:** {pre_ms}\n")
            f.write(f"**Post-Silence:** {post_ms}\n")
            f.write(f"**Start-Offset:** {start_offset}\n")
            f.write(f"**End-Offset:** {end_offset}\n")
            f.write(f"**Inter-Phrase-Silence:** {inter_silences}\n")

            # Liste des phrases pour que l'utilisateur puisse la modifier
            for ph in phrases:
                f.write(f"- {ph}\n")

            f.write("\n----------------------------------------------------------------\n\n")

    print(f"✅ Review file créé : {review_file_path} ({len(groups)} segments)")
    input("Tapez 'Y' pour continuer…")



def parse_review_fileOLDA(review_file_path):
    """
    Lit le review file écrit ci-dessus et
    renvoie une liste de dicts avec :
      - start_s, end_s, final_translation, voice_speed
      - pre_silence, post_silence, phrases (list)
    """
    text = open(review_file_path, encoding="utf-8").read()
    blocks = [b.strip() for b in re.split(r"(?m)^-{3,}\s*$", text) if b.strip()]
    segments = []
    header = re.compile(r"Segment\s+\d+\s+\(start:\s*([0-9.]+)s,\s*end:\s*([0-9.]+)s\)")
    for blk in blocks:
        m = header.search(blk)
        if not m or blk.startswith("Translation Review File"): continue
        start_s, end_s = float(m.group(1)), float(m.group(2))

        ft, vs, pre, post = None, "+0%", 0.0, 0.0
        orig = None
        start_offset = 0 
        phrases = []
        for line in blk.splitlines():
            line = line.strip()
            if line.startswith("**Final Translation:**"):
                ft = line.split("**Final Translation:**",1)[1].strip()
            elif line.startswith("**Voice Speed:**"):
                vs = line.split("**Voice Speed:**",1)[1].strip()
            elif line.startswith("**Pre-Silence:**"):
                pre = float(line.split("**Pre-Silence:**",1)[1])
            elif line.startswith("**Post-Silence:**"):
                post = float(line.split("**Post-Silence:**",1)[1])
            elif line.startswith("**Start-Offset:**"):
                 # offset en millisecondes à ajouter au start
                start_offset = int(line.split("**Start-Offset:**",1)[1])
            elif line.startswith("**End-Offset:**"):
                end_offset = int(line.split("**End-Offset:**",1)[1])                
            elif line.startswith("- "):
                phrases.append(line[2:].strip())
            elif line.startswith("**Original:**"):
                orig = line.split("**Original:**",1)[1].strip()

        segments.append({
            "start_s":           start_s,
            "end_s":             end_s,
            "original":          orig,
            "final_translation": ft or orig,
            "voice_speed":       vs,
            "pre_silence":       pre,
            "post_silence":      post,
            "start_offset_ms":   start_offset,
            "end_offset_ms":     end_offset,
            "phrases":           phrases
        })

    print(f"✅ Parsed {len(segments)} segments depuis le review file.")
    return segments

# ============== TTS Functions: Edge TTS Only with Debug Logging ==============
async def robust_synthesize_phrase(
    phrase: str,
    output_path: str,
    voice: str = "fr-FR-DeniseNeural",
    rate: str = "+0%",
    max_retries: int = 10
):
    """
    Synthesize speech using Edge TTS with robust retry logic.
    Detailed debug messages are printed for each attempt.
    """
    for attempt in range(1, max_retries+1):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                communicate = edge_tts.Communicate(
                    text=phrase,
                    voice=voice,
                    rate=rate
                )
                print(f"[Debug] Attempt {attempt}/{max_retries}: Synthesizing phrase: '{phrase[:30]}…'")
                await communicate.save(output_path)
                print(f"[Debug] Phrase synthesized successfully to {output_path}")
                return
        except Exception as e:
            wait_time = 2 ** attempt + random.random()
            print(f"[Error] Attempt {attempt}/{max_retries} failed for phrase: '{phrase[:30]}…'. Exception: {e}")
            if attempt < max_retries:
                print(f"[Debug] Retrying in {wait_time:.1f}s…")
                await asyncio.sleep(wait_time)
    raise RuntimeError(f"Failed to synthesize phrase after {max_retries} attempts: {phrase[:30]}…")

async def synthesize_phrase_edge_hybrid(
    phrase: str,
    output_path: str,
    voice: str = "fr-FR-DeniseNeural",
    rate: str = "+0%"
):
    # Pour compatibilité, on redirige vers le robust_synthesize
    await robust_synthesize_phrase(phrase, output_path, voice, rate)


def merge_short_phrases(phrases, weights, min_chars=40, max_chars=None):
    new_ph, new_wt = [], []
    buf_ph, buf_wt = "", 0.0
    for ph, wt in zip(phrases, weights):
        if not buf_ph:
            buf_ph, buf_wt = ph, wt
        else:
            if len(buf_ph) < min_chars or len(ph) < min_chars:
                cand = buf_ph + " " + ph
                # si pas de max_chars défini, on fusionne sans condition
                cond = True if max_chars is None else (len(cand) <= max_chars)
                if cond:
                    buf_ph = cand
                    buf_wt += wt
                else:
                    new_ph.append(buf_ph)
                    new_wt.append(buf_wt)
                    buf_ph, buf_wt = ph, wt
            else:
                new_ph.append(buf_ph)
                new_wt.append(buf_wt)
                buf_ph, buf_wt = ph, wt
    if buf_ph:
        new_ph.append(buf_ph)
        new_wt.append(buf_wt)
    return new_ph, new_wt



def split_long_phrasesaaa(phrases, max_chars=80):
    new = []
    for p in phrases:
        if len(p) > max_chars:
            # on découpe au premier “,” ou “ et ” qu’on trouve
            parts = re.split(r",\s+| et ", p, maxsplit=1)
            new.extend([parts[0].strip(), parts[1].strip()] if len(parts)==2 else [p])
        else:
            new.append(p)
    return new

def parse_review_file(review_file_path):
    """
    Lit le review file et renvoie une liste de dicts avec :
      - start_s, end_s, original, final_translation, voice_speed
      - pre_silence, post_silence, start_offset_ms, end_offset_ms
      - phrases (list de phrases) et inter_phrase_silences (liste de silences internes)
    """
    text = open(review_file_path, encoding="utf-8").read()
    blocks = [b.strip() for b in re.split(r"(?m)^-{3,}\s*$", text) if b.strip()]
    segments = []
    header = re.compile(r"Segment\s+\d+\s+\(start:\s*([0-9.]+)s,\s*end:\s*([0-9.]+)s\)")

    for blk in blocks:
        m = header.search(blk)
        if not m or blk.startswith("Translation Review File"): 
            continue
        start_s, end_s = float(m.group(1)), float(m.group(2))

        # valeurs par défaut
        ft, vs = None, "+0%"
        pre, post = 0.0, 0.0
        soffs, eoffs = 0, 0
        phrases = []
        inter = []

        for line in blk.splitlines():
            line = line.strip()
            if line.startswith("**Final Translation:**"):
                ft = line.split("**Final Translation:**",1)[1].strip()
            elif line.startswith("**Voice Speed:**"):
                vs = line.split("**Voice Speed:**",1)[1].strip()
            elif line.startswith("**Pre-Silence:**"):
                pre = float(line.split("**Pre-Silence:**",1)[1])
            elif line.startswith("**Post-Silence:**"):
                post = float(line.split("**Post-Silence:**",1)[1])
            elif line.startswith("**Start-Offset:**"):
                soffs = int(line.split("**Start-Offset:**",1)[1])
            elif line.startswith("**End-Offset:**"):
                eoffs = int(line.split("**End-Offset:**",1)[1])
            elif line.startswith("**Inter-Phrase-Silence:**"):
                parts = line.split("**Inter-Phrase-Silence:**",1)[1].strip()
                if parts:
                    inter = [max(0, int(x)) for x in parts.split(",")]
            elif line.startswith("- "):
                phrases.append(line[2:].strip())

        segments.append({
            "start_s": start_s,
            "end_s": end_s,
            "final_translation": ft or "",
            "voice_speed": vs,
            "pre_silence": pre,
            "post_silence": post,
            "start_offset_ms": soffs,
            "end_offset_ms": eoffs,
            "phrases": phrases,
            "inter_phrase_silences": inter
        })

    print(f"✅ Parsed {len(segments)} segments depuis le review file.")
    return segments


async def async_generate_translated_audio_with_sync_using_review(
    subtitle_source_path, output_audio_path,
    debug_log_path, review_file_path
):
    # 1) Génération / mise à jour du review file
    generate_translation_review_file(
        subtitle_source_path,
        review_file_path,
        max_group_duration_secs=25.0
    )

    # 2) Lecture du review file enrichi
    segments = parse_review_file(review_file_path)

    combined = AudioSegment.silent(duration=0)
    debug    = []

    for idx, seg in enumerate(segments):
        start_s = seg["start_s"]
        end_s   = seg["end_s"]
        total_ms = int((end_s - start_s) * 1000)

        # Récupération des settings
        text    = seg["final_translation"]
        rate    = seg["voice_speed"]
        pre_ms  = seg["pre_silence"]
        post_ms = seg["post_silence"]
        soff    = seg.get("start_offset_ms", 0)
        eoff    = seg.get("end_offset_ms",   0)

        # Phrase splitting & TTS
        phrases = split_french_phrases(text)
        weights = calculate_phrase_weights(text, phrases)
        phrases, weights = merge_short_phrases(phrases, weights, min_chars=40)

        # Budget pour TTS seule
        content_ms = max(0, total_ms - pre_ms - post_ms)

        # Synthèse phrase par phrase
        phrase_audios = []
        for i, ph in enumerate(phrases):
            dur_s  = (content_ms * weights[i]) / 1000.0
            tmp_mp3 = os.path.join(tempfile.gettempdir(), f"tmp_{idx}_{i}.mp3")
            await robust_synthesize_phrase(ph, tmp_mp3, rate=rate)
            aud = AudioSegment.from_mp3(tmp_mp3)
            os.remove(tmp_mp3)
            aud = adjust_audio_duration(aud, dur_s)
            phrase_audios.append(aud)

        # Ajustement interne par override ou répartition égale
        n_inter = max(0, len(phrases) - 1)
        if seg.get("inter_phrase_silences"):
            inter_applied = seg["inter_phrase_silences"]
            # adapter la longueur
            if len(inter_applied) < n_inter:
                inter_applied += [0] * (n_inter - len(inter_applied))
            elif len(inter_applied) > n_inter:
                inter_applied = inter_applied[:n_inter]
        else:
            available = content_ms - sum(a.duration_seconds * 1000 for a in phrase_audios)
            if n_inter > 0 and available > 0:
                sil_ms = available // n_inter
                inter_applied = [sil_ms] * n_inter
            else:
                inter_applied = [0] * n_inter

        # Reconstruction du segment audio
        seg_audio = AudioSegment.silent(duration=pre_ms)
        for i, aud in enumerate(phrase_audios):
            seg_audio += aud
            if i < len(inter_applied):
                seg_audio += AudioSegment.silent(duration=inter_applied[i])
        seg_audio += AudioSegment.silent(duration=post_ms)

        # Application offset de fin
        if eoff > 0:
            seg_audio += AudioSegment.silent(duration=eoff)
        elif eoff < 0:
            seg_audio = seg_audio[:eoff]

        # Debug timing (prise en compte de soff)
        nons2 = detect_nonsilent(seg_audio, min_silence_len=1,
                                 silence_thresh=seg_audio.dBFS - 16)
        start_a = nons2[0][0] if nons2 else pre_ms
        end_a   = nons2[-1][1] if nons2 else (total_ms - post_ms)
        abs_s_a = int(start_s * 1000) + start_a
        abs_e_a = int(start_s * 1000) + end_a
        abs_s_v = int(start_s * 1000) + soff
        abs_e_v = int(end_s   * 1000)
        decal_start = abs_s_a - abs_s_v
        decal_end   = abs_e_a - abs_e_v

        # Mise sur timeline avec offset de start
        start_ms = int(start_s * 1000) + soff
        if len(combined) < start_ms:
            combined += AudioSegment.silent(duration=(start_ms - len(combined)))
        elif len(combined) > start_ms and soff < 0:
            combined = combined[:start_ms]
        combined += seg_audio

        # Enregistrement debug
        # debug.append(
        #     f"Segment {idx+1} ({start_s:.2f}-{end_s:.2f}s): pre={pre_ms}ms, post={post_ms}ms, "
        #     f"speed={rate}, inter={inter_applied}, "
        #     f"décal_start={decal_start}ms, décal_end={decal_end}ms\n"
        # )

        debug.append(
                   f"Segment {idx+1} ({start_s:.2f}-{end_s:.2f}s): "
                   f"pre={pre_ms}ms, post={post_ms}ms, speed={rate}, "
                   f"inter={inter_applied}, "
                   f"phrases={phrases}, "
                   f"décal_start={decal_start}ms, décal_end={decal_end}ms\n"
                )



    # Export debug & wav
    with open(debug_log_path, "w", encoding="utf-8") as df:
        df.write("Translation Debug Log\n\n")
        df.writelines(debug)
    combined.export(output_audio_path, format="wav")

    return output_audio_path


# ============== Merge Audio and Video Function ==============
def merge_audio_video():
    video = VideoFileClip(input_video)
    audio = AudioFileClip(translated_audio)
    if audio.duration < video.duration:
        extra_silence = AudioSegment.silent(duration=(video.duration - audio.duration) * 1000)
        audio_path_temp = os.path.join(output_dir, "temp_full_audio.wav")
        audio_seg = AudioSegment.from_file(translated_audio, format="wav")
        full_audio = audio_seg + extra_silence
        full_audio.export(audio_path_temp, format="wav")
        audio = AudioFileClip(audio_path_temp)
    video = video.set_audio(audio)
    video.write_videofile(
        output_video,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        threads=4
    )

# ============== Main Asynchronous Flow ==============
async def async_main():
    print("Extracting audio...")
    audio_path = extract_audio()
    print("Transcribing audio...")
    language, segments = transcribe(audio_path)
    print("Generating English subtitles...")
    generate_subtitle_file(segments, subtitle_file_en)
    print("Generating French audio with synchronization and manual overrides...")
    await async_generate_translated_audio_with_sync_using_review(subtitle_file_en, translated_audio, debug_log_file, review_file)
    print("Merging audio and video...")
    merge_audio_video()
    print(f"Process completed! Output video: {output_video}")

if __name__ == "__main__":
    asyncio.run(async_main())


