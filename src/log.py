import logging
from pathlib import Path
from typing import Optional


_LOGGER: Optional[logging.Logger] = None
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "app.log"


def get_log() -> logging.Logger:
	"""Retorna uma instância singleton de logger configurada para INFO."""
	global _LOGGER

	if _LOGGER is not None:
		return _LOGGER

	logger = logging.getLogger("imec_analysis")
	logger.setLevel(logging.INFO)
	logger.propagate = False

	if not logger.handlers:
		handler = logging.StreamHandler()
		_LOG_DIR.mkdir(parents=True, exist_ok=True)
		_LOG_FILE.touch(exist_ok=True)
		file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
		formatter = logging.Formatter(
			fmt="%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S",
		)
		handler.setFormatter(formatter)
		file_handler.setFormatter(formatter)
		logger.addHandler(handler)
		logger.addHandler(file_handler)

	_LOGGER = logger
	return _LOGGER
