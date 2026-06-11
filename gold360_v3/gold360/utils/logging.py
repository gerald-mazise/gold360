import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional

from gold360.utils.config import load_config, get_project_root


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    try:
        config = load_config("default")
        log_config = config.get("logging", {})
        if log_config:
            log_config["handlers"]["file"]["filename"] = str(
                get_project_root() / log_config["handlers"]["file"]["filename"]
            )
            logging.config.dictConfig(log_config)
        else:
            _basic_setup()
    except (FileNotFoundError, KeyError):
        _basic_setup()

    logger = logging.getLogger(name or "gold360")
    return logger


def _basic_setup() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root_logger.addHandler(handler)
