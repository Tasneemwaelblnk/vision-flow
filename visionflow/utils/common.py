import logging
from urllib.parse import urlparse

def get_extension_from_url(url: str, default: str = "jpg") -> str:
    if not url: return default
    path = urlparse(str(url)).path
    if '.' in path:
        ext = path.split('.')[-1].lower()
        if len(ext) <= 4: return ext
    return default

def setup_logger(name: str = "visionflow"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger