import datetime as dt
import json
import logging
import logging.config
import os
import atexit
import logging
from typing import Dict, List, Union
import pandas as pd
from datetime import datetime
import sys

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


def setup_logging(
    folder="logs/",
    console_level="INFO",
    file_level="DEBUG",
    continuation=False,
    filtered_loggers: Union[List[str], None] = None,
):
    """Setup logging to log to a file and to stderr.

    The logs are saved in `{logs}/current.log.jsonl` while the program is running. When the program finishes, the logs are saved as a csv file which is named with the current timestamp.

    In the program, the standard logging module can be used to log messages. All logging methods accept an additional parameter `extra` with arbitrary key-value pairs. These key-value pairs are added to the log message that is logged to the file. On the console, only the message is printed.

    Args:
        folder (str, optional): The folder where the logs are saved to. Defaults to "logs".
        console_level (str, optional): The log level that is logged to the console. Defaults to "INFO".
        file_level (str, optional): The log level that is logged to the file. Defaults to "DEBUG".
        continuation (bool, optional): If True, the logs are appended to the existing log file. Defaults to False.
        filtered_loggers (Union[List[str], None], optional): A list of logger names that are filtered from the file log. Defaults to None.
    """
    log_filename = f"{folder}{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    if continuation:
        log_filename = log_filename.replace(".csv", "-continuation.csv")

    def program_exit():
        df = pd.read_json(lines=True, path_or_buf=f"{folder}current.log.jsonl")
        df.to_csv(log_filename, index=False)

    def program_exception(exc_type, exc_value, exc_traceback):
        df = pd.read_json(lines=True, path_or_buf=f"{folder}current.log.jsonl")
        df.to_csv(log_filename, index=False)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    if not os.path.exists(f"{folder}"):
        os.makedirs(f"{folder}")
    if os.path.exists(f"{folder}current.log.jsonl"):
        if continuation:
            with open(f"{folder}current.log.jsonl", "a") as f:
                f.write("\n")
        else:
            os.remove(f"{folder}current.log.jsonl")

    logger_filter = FilteredLoggerFilter(filtered_loggers)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                # console logging
                "simple": {
                    "format": "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S%z",
                },
                # file logging
                "json": {
                    "()": "utils.logger.MyJSONFormatter",
                    "fmt_keys": {
                        "level": "levelname",
                        "message": "message",
                        "timestamp": "timestamp",
                        "logger": "name",
                        "module": "module",
                        "function": "funcName",
                        "line": "lineno",
                        "thread_name": "threadName",
                    },
                },
            },
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "level": console_level,
                    "formatter": "simple",
                    "stream": "ext://sys.stderr",
                },
                "file_json": {
                    "class": "logging.FileHandler",
                    "level": file_level,
                    "formatter": "json",
                    "filename": f"{folder}current.log.jsonl",
                    "filters": ["filtered_loggers"],
                },
            },
            "filters": {"filtered_loggers": {"()": lambda: logger_filter}},
            "loggers": {"": {"level": "DEBUG", "handlers": ["stderr", "file_json"]}},
        }
    )
    atexit.register(program_exit)
    sys.excepthook = program_exception


class FilteredLoggerFilter(logging.Filter):
    def __init__(self, filtered_loggers: Union[List[str], None]):
        super().__init__()
        self.filtered_loggers = filtered_loggers or []

    def filter(self, record):
        return record.name not in self.filtered_loggers


class MyJSONFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        fmt_keys: Union[Dict[str, str], None] = None,
    ):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created).isoformat(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message
