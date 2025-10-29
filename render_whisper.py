from flask import Flask, request, jsonify
import whisper
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model as None, load on first request
model = None

def load_model():
    global model
    if model is None:
        try:
            logger.info("🚀 Loading Whisper model for 24/7 Linely service...")
            # Use tiny model for free tier deployment
            model = whisper.load_model("tiny")
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
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            result = model.transcribe(tmp.name)
            os.unlink(tmp.name)
        
        logger.info(f"✅ Transcription complete: {result['text'][:50]}...")
        return {"transcription": result["text"]}
        
    except Exception as e:
        logger.error(f"❌ Transcription failed: {str(e)}")
        return {"error": str(e)}, 500

if __name__ == '__main__':
    logger.info("🌟 Starting 24/7 Linely Whisper Service...")
    app.run(host='0.0.0.0', port=5000, debug=False)
