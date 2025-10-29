from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import tempfile
import os
import logging
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model as None, load on first request (faster-whisper)
model: WhisperModel | None = None

def load_model() -> None:
    global model
    if model is None:
        try:
            logger.info("🚀 Loading faster-whisper model (tiny, int8, CPU)...")
            # tiny model, CPU only, int8 for low memory environments
            model = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8",
            )
            logger.info("✅ Whisper ready - 24/7 active!")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {str(e)}")
            raise e

@app.route('/health', methods=['GET'])
def health():
    return {"status": "active", "service": "linely-whisper-24-7"}

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return {"error": "No audio file provided"}, 400
        
        # Load model on first request
        load_model()
        
        audio_file = request.files['audio']
        logger.info(f"📝 Transcribing audio file: {audio_file.filename}")

        # Save upload to temp and decode with soundfile (works without ffmpeg)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            audio_path = tmp.name
            audio_file.save(audio_path)

        try:
            audio, sample_rate = sf.read(audio_path, dtype='float32', always_2d=False)
        finally:
            try:
                os.unlink(audio_path)
            except Exception:
                pass

        # Ensure mono
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Resample to 16k if needed (simple linear resample)
        target_sr = 16000
        if sample_rate != target_sr:
            x_old = np.linspace(0, 1, num=len(audio), endpoint=False, dtype=np.float32)
            x_new = np.linspace(0, 1, num=int(len(audio) * target_sr / sample_rate), endpoint=False, dtype=np.float32)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
            sample_rate = target_sr

        # Transcribe using faster-whisper (accepts numpy audio + sample_rate)
        segments, info = model.transcribe(audio, sample_rate=sample_rate)
        text_chunks = []
        for segment in segments:
            text_chunks.append(segment.text)
        full_text = " ".join(t.strip() for t in text_chunks).strip()

        logger.info(f"✅ Transcription complete: {full_text[:50]}...")
        return {"transcription": full_text}
        
    except Exception as e:
        logger.error(f"❌ Transcription failed: {str(e)}")
        return {"error": str(e)}, 500

if __name__ == '__main__':
    logger.info("🌟 Starting 24/7 Linely Whisper Service...")
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=False)
