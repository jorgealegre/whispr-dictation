"""
py2app build script for Whisper Dictation

To build the application:
    python setup.py py2app

For development/testing (creates alias mode app):
    python setup.py py2app -A
"""

import os
import subprocess
from setuptools import setup

APP = ['src/main.py']
DATA_FILES = []

# ============================================================================
# NATIVE LIBRARY DETECTION
# ============================================================================
# PortAudio is required by PyAudio. We need to bundle it with the app
# so users don't need to install it separately.

def find_portaudio():
    """Find the PortAudio dynamic library on the system."""
    # Common locations for PortAudio on macOS
    possible_paths = [
        '/opt/homebrew/lib/libportaudio.dylib',      # Apple Silicon Homebrew
        '/opt/homebrew/lib/libportaudio.2.dylib',    # Apple Silicon Homebrew (versioned)
        '/usr/local/lib/libportaudio.dylib',          # Intel Homebrew
        '/usr/local/lib/libportaudio.2.dylib',        # Intel Homebrew (versioned)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found PortAudio at: {path}")
            return path
    
    # Try using brew to find it
    try:
        result = subprocess.run(
            ['brew', '--prefix', 'portaudio'],
            capture_output=True, text=True, check=True
        )
        brew_prefix = result.stdout.strip()
        dylib_path = os.path.join(brew_prefix, 'lib', 'libportaudio.dylib')
        if os.path.exists(dylib_path):
            print(f"Found PortAudio via Homebrew at: {dylib_path}")
            return dylib_path
        # Try versioned name
        dylib_path = os.path.join(brew_prefix, 'lib', 'libportaudio.2.dylib')
        if os.path.exists(dylib_path):
            print(f"Found PortAudio via Homebrew at: {dylib_path}")
            return dylib_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    print("WARNING: PortAudio not found. The app may not work on other machines.")
    print("Install it with: brew install portaudio")
    return None

PORTAUDIO_PATH = find_portaudio()
FRAMEWORKS = [PORTAUDIO_PATH] if PORTAUDIO_PATH else []

# Application metadata
APP_NAME = 'Whisper Dictation'
APP_VERSION = '1.0.0'
APP_BUNDLE_ID = 'com.whisper-dictation.app'

# Info.plist configuration
PLIST = {
    'CFBundleName': APP_NAME,
    'CFBundleDisplayName': APP_NAME,
    'CFBundleIdentifier': APP_BUNDLE_ID,
    'CFBundleVersion': APP_VERSION,
    'CFBundleShortVersionString': APP_VERSION,
    'LSMinimumSystemVersion': '10.14.0',
    'LSUIElement': True,  # Makes it a menu bar app (no dock icon)
    'LSBackgroundOnly': False,
    'NSHighResolutionCapable': True,
    
    # Privacy permissions - these will prompt the user on first use
    'NSMicrophoneUsageDescription': 'Whisper Dictation needs microphone access to record your speech for transcription.',
    'NSAppleEventsUsageDescription': 'Whisper Dictation needs accessibility access to type transcribed text at your cursor position.',
    
    # For accessibility/keyboard monitoring
    'NSAccessibilityUsageDescription': 'Whisper Dictation needs accessibility access to detect keyboard shortcuts and type transcribed text.',
}

OPTIONS = {
    'argv_emulation': False,  # Not needed for menu bar apps
    'plist': PLIST,
    'iconfile': None,  # Add path to .icns file if you have one: 'icon.icns'
    
    # Packages that need to be included
    'packages': [
        'rumps',
        'faster_whisper',
        'ctranslate2',
        'huggingface_hub',
        'tokenizers',
        'numpy',
        'pyaudio',
        'pynput',
        'wave',
    ],
    
    # Modules to include
    'includes': [
        'recording_indicator',
        'logger_config',
    ],
    
    # Frameworks to include (for native dependencies)
    # This bundles PortAudio so users don't need to install it
    'frameworks': FRAMEWORKS,
    
    # Resources to include
    'resources': [],
    
    # Exclude unnecessary modules to reduce app size
    'excludes': [
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        'cv2',
        'torch',  # faster-whisper uses ctranslate2, not torch
        'tensorflow',
        'keras',
    ],
    
    # Semi-standalone mode - still requires Python framework
    # Set to True for fully standalone (larger app)
    'semi_standalone': False,
}

setup(
    name=APP_NAME,
    version=APP_VERSION,
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
