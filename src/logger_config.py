import logging
import os
import sys
from dotenv import load_dotenv

# Log file location
LOG_DIR = os.path.expanduser("~/Library/Logs/WhisperDictation")
LOG_FILE = os.path.join(LOG_DIR, "whisper-dictation.log")

def get_log_file_path():
    """Return the path to the log file."""
    return LOG_FILE

class ColoredFormatter(logging.Formatter):
    """Custom logging formatter with color support."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[1;31m' # Bold Red
    }
    RESET = '\033[0m'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = os.getenv('NO_COLOR', 'false').lower() != 'true'
    
    def format(self, record):
        if self.use_colors:
            level_color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{level_color}{record.levelname}{self.RESET}"
            
            log_message = super().format(record)
            
            if level_color:
                parts = log_message.split(' - ', 2)
                if len(parts) >= 3:
                    timestamp, level, message = parts[0], parts[1], ' - '.join(parts[2:])
                    return f"{timestamp} - {level} - {level_color}{message}{self.RESET}"
            
            return log_message
        else:
            return super().format(record)

def setup_logging():
    """Setup logging configuration with color support and file output."""
    load_dotenv()
    
    # Use DEBUG level for bundled app to capture more info
    is_bundled = getattr(sys, 'frozen', False)
    default_level = 'DEBUG' if is_bundled else 'INFO'
    log_level = os.getenv('LOG_LEVEL', default_level).upper()
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[]
    )
    
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console_handler)
    
    # File handler (always write to file for debugging)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        file_handler.setLevel(logging.DEBUG)  # Always capture debug in file
        logger.addHandler(file_handler)
        
        # Log startup marker
        logger.debug("=" * 60)
        logger.debug(f"Whisper Dictation starting (bundled={is_bundled})")
        logger.debug(f"Log file: {LOG_FILE}")
        logger.debug("=" * 60)
    except Exception as e:
        logger.warning(f"Could not set up file logging: {e}")
    
    return logger