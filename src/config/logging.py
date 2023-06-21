import functools
import os
import sys
import time
from typing import Any, Callable, Dict

from loguru import logger

LOG_FORMAT = "<cyan>{file}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> <level>{level}</level>: <level>{message}</level>"  # noqa: E501

config: Dict[Any, Any] = {
    "handlers": [
        dict(
            sink=sys.stdout,
            enqueue=True,
            backtrace=True,
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=LOG_FORMAT,
        ),
    ],
    "extra": {},
}

logger.configure(**config)


def logger_wraps(
    *,
    level="INFO",
    start: bool = True,
    end: bool = True,
    inputs: bool = False,
    outputs: bool = False,
):
    """Decorator to instrument functions with a logger.

    Args:
        level (str, optional): log level for the messages. Defaults to "INFO".
        start (bool, optional): Whether to log the start of the function. Defaults to True.
        end (bool, optional): Whether or not to log the end of the function. Defaults to True.
        inputs (bool, optional): Whether or not to log the arguments and kwargs of the function. Defaults to False.
        outputs (bool, optional): Whether or not to log the function output. Defaults to False.
    """

    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            if start:
                msg = "Start | {}"
                if inputs:
                    msg += " | (args={}, kwargs={})"
                    logger_.log(level, msg, name, args, kwargs)
                else:
                    logger_.log(level, msg, name)
            result = func(*args, **kwargs)
            if end:
                msg = "End | {}"
                if outputs:
                    msg += " | (result={})"
                    logger_.log(level, msg, name, result)
                else:
                    logger_.log(level, msg, name)
            return result

        return wrapped

    return wrapper


def log_time(level: str = "INFO", unit: str = "seconds") -> Callable:
    """Decorator function to log a function duration.

    It will use the passed logger to print the time the function took to finish.
    It can be for class methods too.

    Examples:

        >>> ## Configure the logger to print to stdout a simplified message for tests
        >>> logger.configure(handlers=[dict(sink=sys.stdout, format="{message}")])
        [2]
        >>> @log_time(level="ERROR", unit="seconds")
        ... def my_func():
        ...     time.sleep(1)
        ...     return "success"
        >>> my_func()
        Function my_func duration = 1.0 seconds
        'success'

    Args:
        level (str, optional): Defaults to INFO.
        unit (str, optional): Defaults to "seconds".

    Raises:
        ValueError: Bad unit provided

    Returns:
        Callable: decorator function
    """

    factor_map = {"seconds": 1, "minutes": 60, "hours": 3600}
    if unit not in factor_map:
        raise ValueError(f"Bad unit {unit}. Unit must be one of {factor_map.keys()}")

    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            msg = f"Function {name} duration = {(te - ts)/factor_map.get(unit):.1f} {unit}"  # type: ignore
            logger_.log(level, msg, name)
            return result

        return wrapped

    return wrapper
