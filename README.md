# Whisper Dictation

**Note: This application is for macOS only.**

A macOS application that converts speech to text using OpenAI's Whisper model running locally. Press the Globe/Function key to start recording, press it again to stop recording, transcribe, and paste text at your current cursor position.

## Features

- System tray (menu bar) application that runs in the background
- Global hotkey (Globe/Function key or Right Shift) to trigger dictation
- Transcribes speech to text using OpenAI's Whisper model locally
- Automatically types transcribed text at your cursor position
- Visual feedback with menu bar icon status
- Configurable model size for speed vs accuracy tradeoff

## Setup and Installation

### Development Setup

1. Install Python dependencies:

```
pip install -r requirements.txt
```

2. Install PortAudio (required for PyAudio):

```
brew install portaudio
```

3. Run the application in development mode:

```
python src/main.py
```

### Building a Standalone App

To build a standalone `.app` bundle that you can share with others:

1. Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
brew install portaudio
```

2. Build the application:

```bash
python setup.py py2app
```

3. The built app will be in the `dist/` folder as `Whisper Dictation.app`.

4. To distribute, you can:
   - Zip the `.app` file: `cd dist && zip -r "Whisper Dictation.zip" "Whisper Dictation.app"`
   - Or create a DMG (see below)

**Note for recipients:** On first launch, macOS will block the app. To open it:
1. Right-click (or Control-click) on the app
2. Select "Open" from the context menu
3. Click "Open" in the dialog that appears
4. Grant Microphone and Accessibility permissions when prompted

**First Launch:** The app will download the Whisper speech model (~500MB for `small.en`) on first launch. This is a one-time download that gets cached in `~/Library/Caches/WhisperDictation/`.

#### Creating a DMG for Distribution

To create a DMG disk image:

```bash
# Install create-dmg if you don't have it
brew install create-dmg

# Create the DMG
create-dmg \
  --volname "Whisper Dictation" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "Whisper Dictation.app" 150 200 \
  --app-drop-link 450 200 \
  "Whisper Dictation.dmg" \
  "dist/Whisper Dictation.app"
```

#### Development Build (Alias Mode)

For faster iteration during development, use alias mode:

```bash
python setup.py py2app -A
```

This creates a smaller app that references your source files (not suitable for distribution).

### Running the Script in the Background

To run the script in the background:

1. Install all dependencies:

```
pip install -r requirements.txt
```

2. Run the script in the background:

```
nohup ./run.sh >/dev/null 2>&1 & disown
```

3. The script will continue running in the background. You can then use the app as described in the Usage section.

## Usage

1. Launch the Whisper Dictation app. You'll see a microphone icon (🎙️) in your menu bar.
2. Press the Globe key or Function key on your keyboard to start recording.
3. Speak clearly into your microphone.
4. Press the Globe/Function key again to stop recording.
5. The app will transcribe your speech and automatically type it at your current cursor position.

**Alternative**: Hold Right Shift to record (release after 0.75s to process, or before to discard).

You can also interact with the app through the menu bar icon:

- Click "Start/Stop Listening" to toggle recording
- Access Settings for configuration options
- Click "Quit" to exit the application

## Permissions

The app requires the following permissions:

- Microphone access (to record your speech).
  Go to System Preferences → Security & Privacy → Privacy → Microphone and add your Terminal or the app.
- Accessibility access (to simulate keyboard presses for pasting).
  Go to System Preferences → Security & Privacy → Privacy → Accessibility and add your Terminal or the app.

## Requirements

- macOS 10.14 or later
- Microphone

## Performance Tuning

The app supports several environment variables to tune transcription speed vs accuracy:

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `WHISPER_MODEL` | `small.en` | `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3` | Smaller = faster, larger = more accurate |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8`, `float32` | `int8` is ~2x faster with minimal quality loss |
| `WHISPER_BEAM_SIZE` | `1` | `1`-`5` | `1` = greedy (fastest), `5` = beam search (most accurate) |
| `WHISPER_VAD_FILTER` | `true` | `true`, `false` | Skips silent portions for faster processing |

### Speed Comparison (approximate)

| Model | Relative Speed | Quality |
|-------|----------------|---------|
| `tiny.en` | ~10x faster | Basic - good for quick notes |
| `base.en` | ~7x faster | Good - handles most dictation well |
| `small.en` | ~4x faster | Very good - recommended balance |
| `medium.en` | 1x (baseline) | Excellent - original default |
| `large-v3` | ~0.5x (slower) | Best - for high accuracy needs |

### Example: Maximum Speed

```bash
WHISPER_MODEL=tiny.en WHISPER_BEAM_SIZE=1 ./run.sh
```

### Example: Maximum Accuracy

```bash
WHISPER_MODEL=medium.en WHISPER_BEAM_SIZE=5 WHISPER_COMPUTE_TYPE=float32 ./run.sh
```

## Troubleshooting

If something goes wrong or you need to stop the background process, you can kill it by running one of the following commands in your Terminal:

1. List the running process(es):

```
ps aux | grep 'src/main.py'
```

2. Kill the process by its PID:

```
kill -9 <PID>
```
