import json
import logging
from pathlib import Path

# Load configuration file
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.json"
try:
    CONFIG = json.loads(CONFIG_PATH.read_text())
except Exception:
    CONFIG = {}

# Strict Jetson configuration
CLASSES = CONFIG.get("classes")
DEBUG = CONFIG.get("debug", False)

# Set up global logging format and level
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name."""
    return logging.getLogger(name)
