from __future__ import annotations

import logging
from pathlib import Path


def init_logger(name: str, *, log_dir: str | Path | None = None, log_filename: str = "run.log") -> logging.Logger:
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_dir is not None:
        resolved_log_dir = Path(log_dir)
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = resolved_log_dir / log_filename
        log_path_str = str(log_path.resolve())
        has_file_handler = any(
            isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve().as_posix() == Path(log_path_str).as_posix()
            for handler in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
