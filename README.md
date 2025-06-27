# Video Translator & Voice Generator

This application translates the audio of a video into a target language, generates synchronized audio using text-to-speech (TTS), and merges the translated audio with the original video. It ensures synchronization between the video and audio, dynamically adjusts timing, and provides robust error handling.

---

## Features

- **Video Translation**: Automatically transcribes the video, translates the text into the target language, and generates audio using neural TTS voices.
- **Dynamic Synchronization**: Ensures the generated audio matches the video timing, with adjustments for mismatches.
- **Multiple Voice Options**: Choose from a variety of neural voices for the translated audio.
- **Error Handling**: Validates input and output files, handles timing mismatches, and provides detailed error messages.
- **Debugging Tools**: Generates a detailed debug log for each processing session.
- **Streamlit UI**: User-friendly interface for uploading videos, configuring settings, and downloading the translated video.

---

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**:
  - `streamlit`
  - `pydub`
  - `moviepy`
  - `deep-translator`
  - `edge-tts`
  - `whisper`
  - `nest-asyncio`
  - `ffmpeg` (must be installed manually (https://ffmpeg.org/download.html) and accessible in the system PATH)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mambamota/video_transcript_translation.git
   cd video_transcript_translation








## Transcript Generation:
The application uses Whisper – a speech recognition tool that listens to your video and converts the speech into text automatically. This is what creates the original transcript of your video.

## Translation:
Once the transcript is ready, it is then translated into another language. We have two translation options:

One option uses Google Translator, a reliable tool that quickly translates text.

Alternatively, the app supports cloud-based translation using OpenAI or a local model called Ollama. These provide advanced translation capabilities, especially for technical or specialized content.
This step ensures that the text is converted into your desired language accurately.

## Voice Generation:
After translation, the app converts the translated text into natural-sounding speech. We use `edge_tts`, a modern text-to-speech library. It is chosen because it provides high-quality, natural voices with adjustable speaking rates, which helps the narration match the video's timing.

## Audio Synchronization and Video Merging:
To ensure everything fits perfectly, the generated audio is synchronized with the original video’s timing using pydub (which handles audio editing and timing adjustments) and moviepy (which is in charge of merging the new audio with the video). This final step produces a new video file that is smooth, synchronized, and ready for sharing.

User Interface:
Finally, the whole process is wrapped in a user-friendly interface using Streamlit. Streamlit lets users easily upload videos, review and edit transcripts, and ultimately download the translated video—all through a simple web page.