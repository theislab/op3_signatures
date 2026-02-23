import sys
import logging

FMT = "%(asctime)s | [%(levelname)s] %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
_formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)

_h_out = logging.StreamHandler(sys.stdout)
_h_out.setLevel(logging.INFO)
_h_out.addFilter(lambda record: record.levelno == logging.INFO)
_h_out.setFormatter(_formatter)

_h_err = logging.StreamHandler(sys.stderr)
_h_err.setLevel(logging.WARNING)
_h_err.setFormatter(_formatter)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that routes INFO to stdout and WARNING+ to stderr.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__``.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.handlers = [_h_out, _h_err]
    return logger
