#!/bin/bash

echo "Starting Whisper Dictation..."
echo "This app needs accessibility permissions to detect keyboard shortcuts"
echo "If this is your first time running the app, please allow Terminal in"
echo "System Preferences → Privacy & Security → Privacy → Accessibility"
echo ""
echo "The app will now open. Look for the microphone icon (🎙️) in your menu bar."
echo "Press the Globe/Fn key (bottom right corner of keyboard) to start/stop recording."
echo "Or hold Right Shift to record instantly (release after 0.75s to process, before to discard)."
echo ""
echo "Performance tuning (set env vars to customize):"
echo "  WHISPER_MODEL=small.en      # tiny.en|base.en|small.en|medium.en|large-v3"
echo "  WHISPER_COMPUTE_TYPE=int8   # int8 (fast) or float32 (accurate)"
echo "  WHISPER_BEAM_SIZE=1         # 1=fastest, 5=most accurate"
echo "  WHISPER_VAD_FILTER=true     # Skip silence for faster processing"
echo ""
echo "Press Ctrl+C to quit the app."

# Run the dictation app
cd "$(dirname "$0")"
python3 src/main.py