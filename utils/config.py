import yaml
from pathlib import Path
from utils.logging import get_logger

logger = get_logger("config")

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise
