"""
utils/logger.py
═══════════════════════════════════════════════════════════════
Structured logging for the Rollover Betting AI backend.
Outputs JSON-formatted logs for easy parsing and monitoring.
═══════════════════════════════════════════════════════════════
"""
import os
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


LOG_DIR   = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


class StructuredFormatter(logging.Formatter):
    """Formats log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level":     record.levelname,
            "module":    record.module,
            "message":   record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger instance."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Console handler (Railway picks this up in deploy logs)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)

    # File handler (persists across requests)
    try:
        file_handler = logging.FileHandler(f"{LOG_DIR}/app.log")
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)
    except Exception:
        pass

    return logger


class RequestLogger:
    """Logs API request/response metadata."""

    def __init__(self):
        self.logger = get_logger("request")

    def log_request(self, method: str, path: str, client_ip: str,
                    status: int, duration_ms: float, extra: Dict = None):
        entry = {
            "type":        "request",
            "method":      method,
            "path":        path,
            "client_ip":   client_ip,
            "status_code": status,
            "duration_ms": round(duration_ms, 2),
        }
        if extra:
            entry.update(extra)
        self.logger.info("API request", extra={"extra": entry})

    def log_ai_call(self, model: str, duration_ms: float,
                    success: bool, n_tips: int = 0, error: str = None):
        entry = {
            "type":        "ai_call",
            "model":       model,
            "duration_ms": round(duration_ms, 2),
            "success":     success,
            "n_tips":      n_tips,
        }
        if error:
            entry["error"] = error[:200]
        self.logger.info("AI call completed", extra={"extra": entry})

    def log_tips_generated(self, n_fixtures: int, n_tips: int,
                            active_ais: list, ml_used: bool,
                            duration_ms: float):
        entry = {
            "type":        "tips_generated",
            "n_fixtures":  n_fixtures,
            "n_tips":      n_tips,
            "active_ais":  active_ais,
            "ml_used":     ml_used,
            "duration_ms": round(duration_ms, 2),
        }
        self.logger.info("Tips generated", extra={"extra": entry})

    def log_error(self, error: Exception, context: str = ""):
        entry = {
            "type":      "error",
            "context":   context,
            "error":     str(error)[:300],
            "traceback": traceback.format_exc()[-500:],
        }
        self.logger.error("Error occurred", extra={"extra": entry})


# Singletons
_request_logger: Optional[RequestLogger] = None

def get_request_logger() -> RequestLogger:
    global _request_logger
    if _request_logger is None:
        _request_logger = RequestLogger()
    return _request_logger
