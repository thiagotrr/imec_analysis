import logging
from typing import Optional

_LOGGER: Optional[logging.Logger] = None

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
		formatter = logging.Formatter(
			fmt="%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S",
		)
		handler.setFormatter(formatter)
		logger.addHandler(handler)

	_LOGGER = logger
	return _LOGGER