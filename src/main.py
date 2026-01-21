#!/usr/bin/env python3
import os
import sys
import time
import tempfile
import threading
import subprocess
import pyaudio
import wave
import numpy as np
import rumps
from pynput import keyboard
from pynput.keyboard import Key, Controller
import faster_whisper
import signal
import AppKit
import AVFoundation
from Quartz import (
    CGEventMaskBit, kCGEventKeyDown, kCGEventKeyUp, kCGEventFlagsChanged,
    CGEventGetIntegerValueField, kCGKeyboardEventKeycode,
    kCGEventFlagMaskShift, CGEventGetFlags
)
from ApplicationServices import AXIsProcessTrusted
from recording_indicator import RecordingIndicator
from logger_config import setup_logging, get_log_file_path

logger = setup_logging()

# ============================================================================
# APP BUNDLE DETECTION
# ============================================================================
def is_bundled_app():
    """Check if we're running as a bundled .app or from CLI."""
    # py2app sets sys.frozen when running as a bundled app
    return getattr(sys, 'frozen', False)

def send_notification(title, subtitle, message, sound=False):
    """Send a notification, but only if running as a bundled app.
    
    rumps.notification() requires an Info.plist with CFBundleIdentifier,
    which only exists in bundled apps. Skip notifications when running from CLI.
    """
    if is_bundled_app():
        try:
            rumps.notification(title=title, subtitle=subtitle, message=message, sound=sound)
        except Exception as e:
            logger.debug(f"Could not send notification: {e}")
    else:
        # Log the notification content instead when running from CLI
        logger.info(f"[Notification] {title}: {subtitle} - {message}")

# ============================================================================
# PERMISSION CHECKING
# ============================================================================
def check_microphone_permission():
    """Check if microphone permission is granted.
    
    Returns:
        str: 'authorized', 'denied', 'not_determined', or 'restricted'
    """
    status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
        AVFoundation.AVMediaTypeAudio
    )
    status_map = {
        0: 'not_determined',
        1: 'restricted',
        2: 'denied',
        3: 'authorized'
    }
    return status_map.get(status, 'unknown')

def check_accessibility_permission():
    """Check if accessibility permission is granted.
    
    Returns:
        bool: True if accessibility is enabled, False otherwise
    """
    return AXIsProcessTrusted()

def request_microphone_permission():
    """Request microphone permission (triggers system prompt on first call)."""
    # This will trigger the permission prompt if not determined
    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVFoundation.AVMediaTypeAudio,
        lambda granted: logger.info(f"Microphone permission granted: {granted}")
    )

def open_accessibility_settings():
    """Open System Settings to the Accessibility privacy pane."""
    # macOS Ventura+ uses different URL scheme
    url = AppKit.NSURL.URLWithString_(
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    )
    AppKit.NSWorkspace.sharedWorkspace().openURL_(url)

def open_microphone_settings():
    """Open System Settings to the Microphone privacy pane."""
    url = AppKit.NSURL.URLWithString_(
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
    )
    AppKit.NSWorkspace.sharedWorkspace().openURL_(url)

# ============================================================================
# MODEL CACHE CONFIGURATION
# ============================================================================
# Use macOS standard cache location for the Whisper model
# This ensures the model is downloaded once and reused across app launches
def get_model_cache_dir():
    """Get the cache directory for Whisper models."""
    # Use ~/Library/Caches/WhisperDictation for macOS standard location
    cache_dir = os.path.expanduser("~/Library/Caches/WhisperDictation/models")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

MODEL_CACHE_DIR = get_model_cache_dir()

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

        # Permissions menu item
        self.permissions_item = rumps.MenuItem("Check Permissions...", callback=self.check_permissions_clicked)
        
        # Debug menu items
        self.view_logs_item = rumps.MenuItem("View Logs...", callback=self.view_logs_clicked)

        self.menu = [self.recording_menu_item, self.mic_submenu, None, self.permissions_item, self.view_logs_item, None, self.status_item]

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
        
        # Event monitor reference (for cleanup)
        self.event_monitor = None
        self.is_recording_with_key63 = False

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
        
        # Check permissions after a short delay (don't block initialization)
        threading.Timer(1.0, self.check_permissions_on_launch).start()

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

        # Remove NSEvent monitor
        if hasattr(self, 'event_monitor') and self.event_monitor:
            AppKit.NSEvent.removeMonitor_(self.event_monitor)
            logger.info("NSEvent monitor removed")

        # Close PyAudio
        if hasattr(self, 'audio'):
            try:
                self.audio.terminate()
            except:
                pass

    def load_model(self):
        self.title = "🎙️ (Loading...)"
        
        # Check if model exists in cache
        model_path = os.path.join(MODEL_CACHE_DIR, WHISPER_MODEL.replace("/", "_"))
        model_exists = os.path.exists(model_path) and os.listdir(model_path) if os.path.isdir(model_path) else False
        
        if model_exists:
            self.status_item.title = f"Status: Loading {WHISPER_MODEL}..."
            logger.info(f"Loading cached Whisper model: {WHISPER_MODEL}")
        else:
            self.status_item.title = f"Status: Downloading {WHISPER_MODEL}... (first launch)"
            logger.info(f"Downloading Whisper model: {WHISPER_MODEL} (this may take a few minutes)")
            logger.info(f"Model will be cached at: {MODEL_CACHE_DIR}")
            # Show a notification for first-time download
            send_notification(
                title="Whisper Dictation",
                subtitle="Downloading speech model...",
                message=f"This is a one-time download. The {WHISPER_MODEL} model will be cached for future use.",
                sound=False
            )
        
        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL} (compute_type={WHISPER_COMPUTE_TYPE})")
            self.model = faster_whisper.WhisperModel(
                WHISPER_MODEL,
                device="cpu",  # Use CPU on macOS (MPS not yet supported by CTranslate2)
                compute_type=WHISPER_COMPUTE_TYPE,
                download_root=MODEL_CACHE_DIR,  # Use our cache directory
            )
            self.title = "🎙️"
            self.status_item.title = "Status: Ready"
            logger.info(f"Whisper model '{WHISPER_MODEL}' loaded successfully!")
            
            if not model_exists:
                # Notify user that download is complete
                send_notification(
                    title="Whisper Dictation",
                    subtitle="Ready to use!",
                    message="Speech model downloaded successfully. Press Globe/Fn or hold Right Shift to dictate.",
                    sound=True
                )
        except Exception as e:
            self.title = "🎙️ (Error)"
            self.status_item.title = "Status: Error loading model"
            logger.error(f"Error loading model: {e}")
            send_notification(
                title="Whisper Dictation",
                subtitle="Error",
                message=f"Failed to load model: {str(e)[:100]}",
                sound=True
            )

    def check_permissions_on_launch(self):
        """Check permissions when app launches and show prompt if needed."""
        mic_status = check_microphone_permission()
        mic_ok = mic_status == 'authorized'
        acc_ok = check_accessibility_permission()
        
        logger.info(f"Launch permission check - Microphone: {mic_status}, Accessibility: {acc_ok}")
        
        # Request microphone permission if not determined yet
        if mic_status == 'not_determined':
            logger.info("Requesting microphone permission...")
            request_microphone_permission()
            # Re-check after a moment
            time.sleep(0.5)
            mic_ok = check_microphone_permission() == 'authorized'
        
        # Update status bar
        if mic_ok and acc_ok:
            self.status_item.title = "Status: Ready"
            return  # All good, no prompt needed
        
        # Build list of missing permissions
        missing = []
        if not mic_ok:
            missing.append("Microphone")
        if not acc_ok:
            missing.append("Accessibility")
        
        self.status_item.title = f"Status: Missing ({', '.join(missing)})"
        
        # Show a helpful prompt
        window = rumps.Window(
            title="Welcome to Whisper Dictation",
            message=(
                f"To work properly, this app needs:\n\n"
                f"{'✗' if not mic_ok else '✓'} Microphone — record your speech\n"
                f"{'✗' if not acc_ok else '✓'} Accessibility — detect hotkeys & type text\n\n"
                f"Click a button below to open the relevant Settings:"
            ),
            ok="Open Accessibility Settings" if not acc_ok else "Open Microphone Settings",
            cancel="Later",
            dimensions=(320, 0)
        )
        
        # Add second button if both are missing
        if not mic_ok and not acc_ok:
            window.add_button("Open Microphone Settings")
        
        response = window.run()
        
        if response.clicked == 1:  # First button (OK)
            if not acc_ok:
                open_accessibility_settings()
            else:
                open_microphone_settings()
        elif response.clicked == 2:  # Second button (if exists)
            open_microphone_settings()
    
    def check_permissions_clicked(self, _):
        """Handle click on 'Check Permissions' menu item."""
        mic_ok = check_microphone_permission() == 'authorized'
        acc_ok = check_accessibility_permission()
        
        logger.info(f"Permission check - Microphone: {mic_ok}, Accessibility: {acc_ok}")
        
        if mic_ok and acc_ok:
            self.status_item.title = "Status: Ready"
            rumps.alert(
                title="Permissions OK",
                message="All required permissions are granted. The app is ready to use.",
                ok="Great!"
            )
        else:
            missing = []
            if not mic_ok:
                missing.append("Microphone")
            if not acc_ok:
                missing.append("Accessibility")
            
            self.status_item.title = f"Status: Missing ({', '.join(missing)})"
            
            # Show window with buttons for each missing permission
            window = rumps.Window(
                title="Permissions Required",
                message=(
                    f"{'✗' if not mic_ok else '✓'} Microphone — record your speech\n"
                    f"{'✗' if not acc_ok else '✓'} Accessibility — detect hotkeys & type text\n\n"
                    f"Click a button to open Settings:"
                ),
                ok="Open Accessibility Settings" if not acc_ok else "Open Microphone Settings",
                cancel="Later",
                dimensions=(320, 0)
            )
            
            if not mic_ok and not acc_ok:
                window.add_button("Open Microphone Settings")
            
            response = window.run()
            
            if response.clicked == 1:  # First button
                if not acc_ok:
                    open_accessibility_settings()
                else:
                    open_microphone_settings()
            elif response.clicked == 2:  # Second button
                open_microphone_settings()
    
    def view_logs_clicked(self, _):
        """Open the log file in Console.app or Finder."""
        log_path = get_log_file_path()
        logger.info(f"Opening log file: {log_path}")
        
        if os.path.exists(log_path):
            # Open with Console.app for better log viewing
            subprocess.run(['open', '-a', 'Console', log_path])
        else:
            rumps.alert(
                title="No Logs Found",
                message=f"Log file not found at:\n{log_path}",
                ok="OK"
            )

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
        """Set up keyboard monitoring using native macOS NSEvent."""
        logger.info("Setting up global keyboard monitor using NSEvent...")
        
        # Key codes
        self.RIGHT_SHIFT_KEYCODE = 60  # Right Shift key code
        self.GLOBE_FN_KEYCODE = 63     # Globe/Fn key code
        
        # Track modifier state for Right Shift detection
        self.right_shift_down = False
        
        # Define the event mask for key events and modifier changes
        mask = (
            AppKit.NSEventMaskKeyDown |
            AppKit.NSEventMaskKeyUp |
            AppKit.NSEventMaskFlagsChanged
        )
        
        def handle_event(event):
            """Handle keyboard events from NSEvent global monitor."""
            try:
                event_type = event.type()
                keycode = event.keyCode()
                
                logger.debug(f"NSEvent: type={event_type}, keyCode={keycode}")
                
                # Handle modifier key changes (for Right Shift)
                if event_type == AppKit.NSEventTypeFlagsChanged:
                    # Check if this is the Right Shift key
                    if keycode == self.RIGHT_SHIFT_KEYCODE:
                        flags = event.modifierFlags()
                        shift_pressed = bool(flags & AppKit.NSEventModifierFlagShift)
                        
                        if shift_pressed and not self.right_shift_down:
                            # Right Shift pressed
                            self.right_shift_down = True
                            logger.info("Right Shift PRESSED (NSEvent)")
                            if not self.recording:
                                self.shift_press_time = time.time()
                                self.shift_held = True
                                self.start_recording()
                        elif not shift_pressed and self.right_shift_down:
                            # Right Shift released
                            self.right_shift_down = False
                            logger.info("Right Shift RELEASED (NSEvent)")
                            if self.shift_held:
                                hold_duration = time.time() - self.shift_press_time
                                self.shift_held = False
                                
                                if hold_duration < self.shift_threshold:
                                    logger.info(f"Held for {hold_duration:.2f}s - discarding (< {self.shift_threshold}s)")
                                    self.discard_recording()
                                else:
                                    logger.info(f"Held for {hold_duration:.2f}s - processing")
                                    self.stop_recording()
                
                # Handle Globe/Fn key (regular key down/up)
                elif event_type == AppKit.NSEventTypeKeyDown:
                    if keycode == self.GLOBE_FN_KEYCODE:
                        logger.info(f"Globe/Fn key DOWN (keycode={keycode})")
                        # Toggle recording on key down for Globe/Fn
                        if not self.recording and not self.is_recording_with_key63:
                            self.is_recording_with_key63 = True
                            self.start_recording()
                            
                elif event_type == AppKit.NSEventTypeKeyUp:
                    if keycode == self.GLOBE_FN_KEYCODE:
                        logger.info(f"Globe/Fn key UP (keycode={keycode})")
                        if self.recording and self.is_recording_with_key63:
                            self.is_recording_with_key63 = False
                            self.stop_recording()
                            
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Add global monitor for key events
        self.event_monitor = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            mask,
            handle_event
        )
        
        if self.event_monitor:
            logger.info("NSEvent global monitor registered successfully")
        else:
            logger.error("Failed to register NSEvent global monitor")
            # Fall back to pynput
            logger.info("Falling back to pynput keyboard listener...")
            self.key_monitor_thread = threading.Thread(target=self.monitor_keys)
            self.key_monitor_thread.daemon = True
            self.key_monitor_thread.start()
            logger.info("Keyboard monitor thread started")

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
        
        logger.info("monitor_keys() started - initializing keyboard listener")
        logger.info(f"Accessibility permission check: {check_accessibility_permission()}")

        def on_press(key):
            # Log ALL key presses for debugging
            key_info = f"key={key}"
            if hasattr(key, 'vk'):
                key_info += f", vk={key.vk}"
            if hasattr(key, 'char'):
                key_info += f", char={key.char}"
            logger.debug(f"KEY PRESS: {key_info}")
            
            # If Right Shift is held and another key is pressed, cancel recording (user is typing)
            if self.shift_held and key != Key.shift_r:
                logger.info("Other key pressed while Right Shift held - canceling recording")
                self.shift_held = False
                self.discard_recording()
                return

            # Log when target key is pressed
            if hasattr(key, 'vk') and key.vk == self.trigger_key:
                logger.info(f"Target key (vk={key.vk}) pressed")

            # Right Shift handling - start recording immediately (optimistic)
            if key == Key.shift_r:
                logger.info(f"Right Shift detected! recording={self.recording}")
                if not self.recording:
                    logger.info("Right Shift pressed - starting recording immediately")
                    self.shift_press_time = time.time()
                    self.shift_held = True
                    self.start_recording()

        def on_release(key):
            # Log ALL key releases for debugging
            key_info = f"key={key}"
            if hasattr(key, 'vk'):
                key_info += f", vk={key.vk}"
            logger.debug(f"KEY RELEASE: {key_info}")
            
            if hasattr(key, 'vk'):
                if key.vk == self.trigger_key:
                    if not self.recording and not self.is_recording_with_key63:
                        logger.info(f"Globe/Fn key (vk={key.vk}) released - STARTING recording")
                        self.is_recording_with_key63 = True
                        self.start_recording()
                    elif self.recording and self.is_recording_with_key63:
                        logger.info(f"Globe/Fn key (vk={key.vk}) released - STOPPING recording")
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
            logger.info("Creating keyboard.Listener...")
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            logger.info(f"Listener created: {listener}")
            listener.start()
            logger.info("Keyboard listener STARTED successfully - waiting for key events")
            logger.info(f"Listening for: Globe/Fn key (vk={self.trigger_key}) and Right Shift")
            listener.join()
            logger.warning("Keyboard listener exited join() - this shouldn't happen normally")
        except Exception as e:
            logger.error(f"FAILED to start keyboard listener: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error("This usually means Accessibility permissions are not granted")

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