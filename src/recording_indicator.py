#!/usr/bin/env python3
import numpy as np
from logger_config import setup_logging

logger = setup_logging()


class RecordingIndicator:
    """Simple recording indicator using menu bar icon animation"""

    def __init__(self, width=50, height=300):
        # These parameters are kept for compatibility but not used
        self.width = width
        self.height = height
        self.audio_levels = []
        self.max_bars = 20
        self.running = False
        self.app_reference = None
        self.animation_frames = ["🔴", "🟠", "🟡", "🟢"]
        self.current_frame = 0

    def set_app_reference(self, app):
        """Set reference to the main app for icon updates"""
        self.app_reference = app

    def start(self):
        """Start the recording indicator"""
        if self.running:
            return

        self.running = True
        self.audio_levels = []
        self.current_frame = 0
        logger.info("Recording indicator started")

    def stop(self):
        """Stop the recording indicator"""
        self.running = False
        logger.info("Recording indicator stopped")

    def update_audio_level(self, audio_data):
        """Update the audio level based on incoming audio data

        Args:
            audio_data: Raw audio bytes from pyaudio
        """
        if not self.running or not self.app_reference:
            return

        try:
            # Convert bytes to numpy array, then to float to avoid integer overflow when squaring
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float64)

            # Calculate RMS (Root Mean Square) level
            rms = np.sqrt(np.mean(audio_array**2))

            # Normalize to 0-1 range
            max_rms = 3000
            normalized_level = min(rms / max_rms, 1.0)

            # Keep a rolling window of levels
            self.audio_levels.append(normalized_level)
            if len(self.audio_levels) > self.max_bars:
                self.audio_levels.pop(0)

            # Update menu bar icon based on audio level
            if normalized_level > 0.7:
                icon = "🔴"  # Red - loud
            elif normalized_level > 0.4:
                icon = "🟡"  # Yellow - medium
            elif normalized_level > 0.1:
                icon = "🟢"  # Green - quiet
            else:
                icon = "⚪"  # White - very quiet/silence

            # Animate the icon
            self.app_reference.title = f"{icon} REC"
        except Exception as e:
            logger.debug(f"Error updating audio level: {e}")
