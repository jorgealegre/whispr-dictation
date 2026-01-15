#!/usr/bin/env python3
import os
import time
import tempfile
import threading
import pyaudio
import wave
import numpy as np
import rumps
from pynput import keyboard
from pynput.keyboard import Key, Controller
import faster_whisper
import signal
from recording_indicator import RecordingIndicator
from logger_config import setup_logging

logger = setup_logging()

# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================
# Model options (speed vs accuracy tradeoff):
#   - "tiny.en"   : Fastest (~10x), basic quality
#   - "base.en"   : Very fast (~7x), good quality
#   - "small.en"  : Fast (~4x), very good quality ← RECOMMENDED
#   - "medium.en" : Slower (~1x), excellent quality
#   - "large-v3"  : Slowest, best quality
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")

# Compute type for inference (int8 is faster on CPU with minimal quality loss)
# Options: "default", "int8", "int8_float16", "float16", "float32"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Beam size: 1 = greedy (fastest), 5 = beam search (more accurate)
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))

# VAD (Voice Activity Detection) - skips silence for faster processing
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
# ============================================================================

# Print loaded configuration
logger.info("=" * 50)
logger.info("WHISPER DICTATION - Configuration")
logger.info("=" * 50)
logger.info(f"  WHISPER_MODEL        = {WHISPER_MODEL}")
logger.info(f"  WHISPER_COMPUTE_TYPE = {WHISPER_COMPUTE_TYPE}")
logger.info(f"  WHISPER_BEAM_SIZE    = {WHISPER_BEAM_SIZE}")
logger.info(f"  WHISPER_VAD_FILTER   = {WHISPER_VAD_FILTER}")
logger.info("=" * 50)

# Set up a global flag for handling SIGINT
exit_flag = False

def signal_handler(sig, frame):
    """Global signal handler for graceful shutdown"""
    global exit_flag
    logger.info("Shutdown signal received, exiting gracefully...")
    exit_flag = True
    # Try to force exit if the app doesn't respond quickly
    threading.Timer(2.0, lambda: os._exit(0)).start()

# Set up graceful shutdown handling for interrupt and termination signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class WhisperDictationApp(rumps.App):
    def __init__(self):
        super(WhisperDictationApp, self).__init__("🎙️", quit_button=rumps.MenuItem("Quit"))

        # Status item
        self.status_item = rumps.MenuItem("Status: Ready")

        # Add menu items - use a single menu item for toggling recording
        self.recording_menu_item = rumps.MenuItem("Start Recording")

        # Recording state
        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.keyboard_controller = Controller()

        # Microphone selection (None = use default)
        self.selected_input_device = None

        # Create microphone selection submenu
        self.mic_menu = {}
        self.mic_menu_mapping = {}  # Maps menu title to device index
        self.setup_microphone_menu()

        self.menu = [self.recording_menu_item, self.mic_submenu, None, self.status_item]

        # Initialize recording indicator
        self.indicator = RecordingIndicator()
        self.indicator.set_app_reference(self)

        # Initialize Whisper model
        self.model = None
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.start()

        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024

        # Hotkey configuration - we'll listen for globe/fn key (vk=63)
        self.trigger_key = 63  # Key code for globe/fn key

        # Right Shift trigger state (optimistic recording with discard threshold)
        self.shift_press_time = None
        self.shift_held = False
        self.shift_threshold = 0.75  # Minimum hold time in seconds

        self.setup_global_monitor()

        # Show initial message
        logger.info("Started WhisperDictation app. Look for 🎙️ in your menu bar.")
        logger.info("Press and hold the Globe/Fn key (vk=63) to record. Release to transcribe.")
        logger.info("Alternatively, hold Right Shift to record (release after 0.75s to process, before to discard).")
        logger.info("Press Ctrl+C to quit the application.")
        logger.info("You may need to grant this app accessibility permissions in System Preferences.")
        logger.info("Go to System Preferences → Security & Privacy → Privacy → Accessibility")
        logger.info("and add your terminal or the built app to the list.")

        # Start a watchdog thread to check for exit flag
        self.watchdog = threading.Thread(target=self.check_exit_flag, daemon=True)
        self.watchdog.start()

    def check_exit_flag(self):
        """Monitor the exit flag and terminate the app when set"""
        while True:
            if exit_flag:
                logger.info("Watchdog detected exit flag, shutting down...")
                self.cleanup()
                rumps.quit_application()
                os._exit(0)
                break
            time.sleep(0.5)

    def cleanup(self):
        """Clean up resources before exiting"""
        logger.info("Cleaning up resources...")
        # Stop recording if in progress
        if self.recording:
            self.recording = False
            if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)

        # Close PyAudio
        if hasattr(self, 'audio'):
            try:
                self.audio.terminate()
            except:
                pass

    def load_model(self):
        self.title = "🎙️ (Loading...)"
        self.status_item.title = f"Status: Loading {WHISPER_MODEL}..."
        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL} (compute_type={WHISPER_COMPUTE_TYPE})")
            self.model = faster_whisper.WhisperModel(
                WHISPER_MODEL,
                device="cpu",  # Use CPU on macOS (MPS not yet supported by CTranslate2)
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            self.title = "🎙️"
            self.status_item.title = "Status: Ready"
            logger.info(f"Whisper model '{WHISPER_MODEL}' loaded successfully!")
        except Exception as e:
            self.title = "🎙️ (Error)"
            self.status_item.title = "Status: Error loading model"
            logger.error(f"Error loading model: {e}")

    def setup_microphone_menu(self):
        """Setup the microphone selection submenu"""
        self.mic_submenu = rumps.MenuItem("Microphone")
        devices = self.get_input_devices()

        for device in devices:
            title = device['name']
            if device['is_default']:
                title += " (Default)"

            menu_item = rumps.MenuItem(title, callback=self.select_microphone)
            # Mark the default device as selected initially
            if device['is_default']:
                menu_item.state = True

            self.mic_menu[title] = menu_item
            self.mic_menu_mapping[title] = device['index']
            self.mic_submenu.add(menu_item)

    def setup_global_monitor(self):
        # Create a separate thread to monitor for global key events
        self.key_monitor_thread = threading.Thread(target=self.monitor_keys)
        self.key_monitor_thread.daemon = True
        self.key_monitor_thread.start()

    def get_input_devices(self):
        """Get list of available audio input devices"""
        devices = []
        default_index = self.audio.get_default_input_device_info()['index']
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': info['index'],
                    'name': info['name'],
                    'is_default': info['index'] == default_index
                })
        return devices

    def select_microphone(self, sender):
        """Callback when a microphone is selected from the menu"""
        # Uncheck all items in the microphone menu
        for item in self.mic_menu.values():
            item.state = False
        # Check the selected item
        sender.state = True
        # Store the device index (stored in the menu item's title parsing or we use a mapping)
        device_index = self.mic_menu_mapping.get(sender.title)
        self.selected_input_device = device_index
        device_name = sender.title.replace(" (Default)", "")
        logger.info(f"Microphone changed to: {device_name}")

    def discard_recording(self):
        """Discard current recording without processing (held too short)"""
        self.recording = False
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=0.5)
        self.frames = []
        self.indicator.stop()
        self.title = "🎙️"
        self.status_item.title = "Status: Recording discarded (too short)"
        logger.info("Recording discarded - held for less than threshold")

    def monitor_keys(self):
        # Track state of key 63 (Globe/Fn key)
        self.is_recording_with_key63 = False

        def on_press(key):
            # If Right Shift is held and another key is pressed, cancel recording (user is typing)
            if self.shift_held and key != Key.shift_r:
                logger.info("Other key pressed while Right Shift held - canceling recording")
                self.shift_held = False
                self.discard_recording()
                return

            # Removed logging for every key press; log only when target key is pressed
            if hasattr(key, 'vk') and key.vk == self.trigger_key:
                logger.debug(f"Target key (vk={key.vk}) pressed")

            # Right Shift handling - start recording immediately (optimistic)
            if key == Key.shift_r:
                if not self.recording:
                    logger.info("Right Shift pressed - starting recording immediately")
                    self.shift_press_time = time.time()
                    self.shift_held = True
                    self.start_recording()

        def on_release(key):
            if hasattr(key, 'vk'):
                logger.debug(f"Key with vk={key.vk} released")
                if key.vk == self.trigger_key:
                    if not self.recording and not self.is_recording_with_key63:
                        logger.debug(f"Globe/Fn key (vk={key.vk}) released - STARTING recording")
                        self.is_recording_with_key63 = True
                        self.start_recording()
                    elif self.recording and self.is_recording_with_key63:
                        logger.debug(f"Globe/Fn key (vk={key.vk}) released - STOPPING recording")
                        self.is_recording_with_key63 = False
                        self.stop_recording()

            # Right Shift handling - check duration and discard or process
            if key == Key.shift_r and self.shift_held:
                hold_duration = time.time() - self.shift_press_time
                self.shift_held = False

                if hold_duration < self.shift_threshold:
                    logger.info(f"Right Shift released after {hold_duration:.2f}s - discarding (< {self.shift_threshold}s)")
                    self.discard_recording()
                else:
                    logger.info(f"Right Shift released after {hold_duration:.2f}s - processing")
                    self.stop_recording()

        try:
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                logger.debug(f"Keyboard listener started - listening for key events")
                logger.debug(f"Target key is Globe/Fn key (vk={self.trigger_key})")
                logger.debug(f"Press and release the target key to control recording")
                listener.join()
        except Exception as e:
            logger.error(f"Error with keyboard listener: {e}")
            logger.error("Please check accessibility permissions in System Preferences")

    @rumps.clicked("Start Recording")  # This will be matched by title
    def toggle_recording(self, sender):
        if not self.recording:
            self.start_recording()
            sender.title = "Stop Recording"
        else:
            self.stop_recording()
            sender.title = "Start Recording"

    def start_recording(self):
        if not hasattr(self, 'model') or self.model is None:
            logger.warning("Model not loaded. Please wait for the model to finish loading.")
            self.status_item.title = "Status: Waiting for model to load"
            return

        self.frames = []
        self.recording = True

        # Update UI
        self.title = "🎙️ (Recording)"
        self.status_item.title = "Status: Recording..."
        logger.info("Recording started. Speak now...")

        # Show recording indicator
        self.indicator.start()

        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

    def stop_recording(self):
        self.recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()

        # Hide recording indicator
        self.indicator.stop()

        # Update UI
        self.title = "🎙️ (Transcribing)"
        self.status_item.title = "Status: Transcribing..."
        logger.info("Recording stopped. Transcribing...")

        # Process in background
        transcribe_thread = threading.Thread(target=self.process_recording)
        transcribe_thread.start()

    def process_recording(self):
        # Transcribe and insert text
        try:
            self.transcribe_audio()
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            self.status_item.title = "Status: Error during transcription"
        finally:
            self.title = "🎙️"  # Reset title

    def record_audio(self):
        # Build kwargs for audio stream
        stream_kwargs = {
            'format': self.format,
            'channels': self.channels,
            'rate': self.rate,
            'input': True,
            'frames_per_buffer': self.chunk
        }
        # Use selected input device if specified
        if self.selected_input_device is not None:
            stream_kwargs['input_device_index'] = self.selected_input_device

        stream = self.audio.open(**stream_kwargs)

        while self.recording:
            data = stream.read(self.chunk)
            self.frames.append(data)

            # Update indicator with audio level
            self.indicator.update_audio_level(data)

        stream.stop_stream()
        stream.close()

    def transcribe_audio(self):
        if not self.frames:
            self.title = "🎙️"
            self.status_item.title = "Status: No audio recorded"
            logger.warning("No audio recorded")
            return

        # Save the recorded audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name

        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))

        logger.debug("Audio saved to temporary file. Transcribing...")

        # Transcribe with Whisper
        try:
            transcribe_start = time.time()
            segments, info = self.model.transcribe(
                temp_filename,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=WHISPER_VAD_FILTER,
                vad_parameters=dict(
                    min_silence_duration_ms=300,  # Shorter silence threshold
                    speech_pad_ms=200,            # Padding around speech
                ),
            )

            text = ""
            for segment in segments:
                text += segment.text
            text = text.strip()  # Remove leading/trailing whitespace (Whisper adds leading space)

            transcribe_time = time.time() - transcribe_start
            logger.info(f"Transcription took {transcribe_time:.2f}s (audio: {info.duration:.1f}s, ratio: {transcribe_time/info.duration:.2f}x)")

            if text:
                self.insert_text(text)
                logger.info(f"Transcription: {text}")
                self.status_item.title = f"Status: Transcribed: {text[:30]}..."
            else:
                logger.warning("No speech detected")
                self.status_item.title = "Status: No speech detected"
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self.status_item.title = "Status: Transcription error"
            raise
        finally:
            # Clean up the temporary file
            os.unlink(temp_filename)

    def insert_text(self, text):
        # Type text at cursor position without altering the clipboard
        logger.debug("Typing text at cursor position...")
        self.keyboard_controller.type(text)
        logger.debug("Text typed successfully")

    def handle_shutdown(self, _signal, _frame):
        """This method is no longer used with the global handler approach"""
        pass

# Wrap the main execution in a try-except to ensure clean exit
if __name__ == "__main__":
    try:
        WhisperDictationApp().run()
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received, exiting...")
        os._exit(0)