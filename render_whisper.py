from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import tempfile
import os
import logging
import numpy as np
import wave

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Hint cache dirs for model downloads in ephemeral environments
os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
os.environ.setdefault("XDG_CACHE_HOME", "/app/.cache")

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

        # Only accept WAV to avoid external decoders
        filename_lower = (audio_file.filename or "").lower()
        if not filename_lower.endswith('.wav'):
            return {"error": "Only .wav files are supported in this deployment"}, 400

        # Save upload to temp and decode WAV using stdlib
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_path = tmp.name
            audio_file.save(audio_path)

        try:
            with wave.open(audio_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
        finally:
            try:
                os.unlink(audio_path)
            except Exception:
                pass

        # Convert raw PCM to float32 numpy
        if sampwidth == 1:
            dtype = np.uint8  # unsigned 8-bit
            audio = (np.frombuffer(raw, dtype=dtype).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 2:
            dtype = np.int16
            audio = np.frombuffer(raw, dtype=dtype).astype(np.float32) / 32768.0
        elif sampwidth == 3:
            # 24-bit little-endian
            a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            signed = (a[:,0].astype(np.int32) | (a[:,1].astype(np.int32) << 8) | (a[:,2].astype(np.int32) << 16))
            signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
            audio = signed.astype(np.float32) / 8388608.0
        elif sampwidth == 4:
            dtype = np.int32
            audio = np.frombuffer(raw, dtype=dtype).astype(np.float32) / 2147483648.0
        else:
            return {"error": f"Unsupported WAV bit depth (sample width {sampwidth})"}, 400

        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1)

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
