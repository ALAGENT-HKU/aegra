import logging
import logging.config
import os
from typing import Any

import structlog


def get_logging_config() -> dict[str, Any]:
    """
    Returns a unified logging configuration dictionary that uses structlog
    for consistent, structured logging across the application and Uvicorn.

    This configuration solves the multiprocessing "pickling" error on Windows
    by using string references for streams (e.g., "ext://sys.stdout").
    
    Logging is configurable via environment variables:
    - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - LOG_TO_FILE: Enable file logging (true/false, default: false)
    - LOG_FILE_PATH: Path to log file (default: logs/aegra.log)
    - LOG_FILE_MAX_BYTES: Max file size before rotation (default: 10MB)
    - LOG_FILE_BACKUP_COUNT: Number of backup files to keep (default: 5)
    """
    # Determine log level from environment or set a default
    env_mode = os.getenv("ENV_MODE", "LOCAL").upper()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # File logging configuration
    log_to_file = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/aegra.log")
    log_file_max_bytes = int(os.getenv("LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024)))  # 10MB
    log_file_backup_count = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))

    # These processors will be used by BOTH structlog and standard logging
    # to ensure consistent output for all logs.
    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        structlog.processors.TimeStamper(fmt="iso"),
        # This processor must be last in the shared chain to format positional args.
        structlog.stdlib.PositionalArgumentsFormatter(),
    ]

    # Determine the final renderer based on the environment
    # Use a colorful console renderer for local development, and JSON for production.
    if env_mode in ("LOCAL", "DEVELOPMENT"):
        console_renderer = structlog.dev.ConsoleRenderer(colors=True, pad_level=True)
    else:
        console_renderer = structlog.processors.JSONRenderer()
    
    # File logs always use JSON format for easier parsing
    file_renderer = structlog.processors.JSONRenderer()

    config = {
        "version": 1,
        "disable_existing_loggers": False,  # Important for library logging
        "formatters": {
            "console": {
                # Use structlog's formatter as the bridge
                "()": "structlog.stdlib.ProcessorFormatter",
                # The final processor is the renderer.
                "processor": console_renderer,
                # These processors are run on ANY log record, including those from Uvicorn.
                "foreign_pre_chain": shared_processors,
            },
            "file": {
                "()": "structlog.stdlib.ProcessorFormatter",
                "processor": file_renderer,
                "foreign_pre_chain": shared_processors,
            },
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "console",
                # IMPORTANT: Use the string reference to avoid the pickling error.
                # This defers the lookup of sys.stdout until the config is loaded
                # in the child process.
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            # Configure the root logger to catch everything
            "": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,  # Don't pass to other handlers
            },
            # Uvicorn's loggers will now inherit the root logger's settings,
            # ensuring they use the same handler and formatter.
            # We explicitly set their level here.
            "uvicorn.error": {
                "level": "INFO",
            },
            "uvicorn.access": {
                "level": "WARNING",
            },
        },
    }
    
    # Add file handler if enabled
    if log_to_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        config["handlers"]["file"] = {
            "level": log_level,
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "file",
            "filename": log_file_path,
            "maxBytes": log_file_max_bytes,
            "backupCount": log_file_backup_count,
            "encoding": "utf-8",
        }
        
        # Add file handler to root logger
        config["loggers"][""]["handlers"].append("file")
    
    return config


def setup_logging():
    """
    Configures both standard logging and structlog based on the
    dictionary from get_logging_config(). This should be called
    once at application startup.
    """
    config = get_logging_config()

    # Configure the standard logging module
    logging.config.dictConfig(config)
    # Propagate uvicorn logs instead of letting uvicorn configure the format
    for name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True

    # Reconfigure log levels for some overly chatty libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    # Configure structlog to route its logs through the standard logging
    # system that we just configured.
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            # Add shared processors to structlog's pipeline
            *config["formatters"]["console"]["foreign_pre_chain"],
            # Prepare the log record for the standard library's formatter
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
