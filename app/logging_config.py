import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_NAMES = ("", "uvicorn", "uvicorn.error", "uvicorn.access", "httpx")
COMBINED_LOG_NAME = "api.log"
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5


class CombinedAndModuleFileHandler(logging.Handler):
    def __init__(self, log_dir: Path, formatter: logging.Formatter) -> None:
        super().__init__(level=logging.INFO)
        self._log_dir = log_dir
        self._formatter = formatter
        self._combined = self._rotating_handler(COMBINED_LOG_NAME)
        self._module_handlers: dict[str, RotatingFileHandler] = {}

    def emit(self, record: logging.LogRecord) -> None:
        self._combined.emit(record)
        self._handler_for(record.name).emit(record)

    def close(self) -> None:
        self._combined.close()
        for handler in self._module_handlers.values():
            handler.close()
        super().close()

    def _handler_for(self, logger_name: str) -> RotatingFileHandler:
        stem = logger_name.rsplit(".", 1)[-1] if logger_name else "root"
        handler = self._module_handlers.get(stem)
        if handler is None:
            handler = self._rotating_handler(f"{stem}.log")
            self._module_handlers[stem] = handler
        return handler

    def _rotating_handler(self, filename: str) -> RotatingFileHandler:
        handler = RotatingFileHandler(
            self._log_dir / filename,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(self._formatter)
        handler.setLevel(logging.INFO)
        return handler


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = CombinedAndModuleFileHandler(
        log_dir,
        logging.Formatter(LOG_FORMAT),
    )

    for name in LOG_NAMES:
        log = logging.getLogger(name)
        log.handlers.clear()
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
