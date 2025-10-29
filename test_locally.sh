#!/bin/bash
echo "🧪 Testing Linely Whisper Service Locally"

# Kill any existing process on ports 5000/5050
echo "🧹 Cleaning up ports 5000 and 5050..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
lsof -ti:5050 | xargs kill -9 2>/dev/null || true

# Start the service in background
echo "🚀 Starting Flask service on PORT=5050..."
cd /Users/mac/Desktop/linely-whisper-24-7
source whisper-env/bin/activate
export PORT=5050
python render_whisper.py &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to start (up to 30s)..."
for i in {1..30}; do
  if curl -sf http://localhost:5050/health >/dev/null; then
    echo "✅ Server is up!"
    break
  fi
  sleep 1
done

# Test health endpoint
echo "📋 Testing health endpoint..."
curl -s http://localhost:5050/health | python3 -m json.tool

# Create test audio
echo "🎤 Creating test audio..."
say "Hello this is Linely AI receptionist testing Whisper transcription" -o test_audio.aiff

# Test transcription
echo "📝 Testing transcription..."
curl -X POST http://localhost:5050/transcribe \
  -F "audio=@test_audio.aiff" \
  -s | python3 -m json.tool

# Clean up
echo "🧹 Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
rm -f test_audio.aiff

echo "✅ Local testing complete!"
