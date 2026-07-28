import logging
from pathlib import Path

from config import LOG_DIR

log_file = LOG_DIR / "voice_agent.log"

logging.basicConfig(

    filename=log_file,

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger("GenesisVoiceAgent")