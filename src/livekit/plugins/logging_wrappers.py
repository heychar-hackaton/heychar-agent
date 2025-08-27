import asyncio
import inspect
import logging
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


class _LoggingProxy:
    """Generic dynamic logging proxy for provider instances.

    - Logs attribute access and method calls.
    - Supports both sync and async callables.
    - Does not alter behavior/return values.
    """

    def __init__(self, target: Any, name: str) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, item: str) -> Any:
        target = object.__getattribute__(self, "_target")
        name = object.__getattribute__(self, "_name")
        attr = getattr(target, item)

        # Log attribute access to see which methods are being used
        logger.info("[%s.%s] getattr -> %s", name, item, type(attr).__name__)

        if callable(attr):
            # Wrap callable to log calls and results
            if inspect.iscoroutinefunction(attr):

                @wraps(attr)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    logger.info("[%s.%s] call (async) args=%s kwargs=%s", name, item, _short(args), _short(kwargs))
                    try:
                        res = await attr(*args, **kwargs)
                        logger.info("[%s.%s] ok -> %s", name, item, _short(res))
                        return res
                    except Exception:
                        logger.exception("[%s.%s] error", name, item)
                        raise

                return async_wrapper

            else:

                @wraps(attr)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    logger.info("[%s.%s] call args=%s kwargs=%s", name, item, _short(args), _short(kwargs))
                    try:
                        res = attr(*args, **kwargs)
                        # If it returns a coroutine, we won't await here; just log type
                        logger.info("[%s.%s] ok -> %s", name, item, _short(res))
                        return res
                    except Exception:
                        logger.exception("[%s.%s] error", name, item)
                        raise

                return sync_wrapper

        return attr

    def __setattr__(self, key: str, value: Any) -> None:
        target = object.__getattribute__(self, "_target")
        name = object.__getattribute__(self, "_name")
        logger.info("[%s.%s] setattr value=%s", name, key, _short(value))
        setattr(target, key, value)

    def __repr__(self) -> str:  # pragma: no cover
        target = object.__getattribute__(self, "_target")
        name = object.__getattribute__(self, "_name")
        return f"<LoggingProxy name={name} target={target!r}>"

    # Support callable targets (rare, but safe)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        target = object.__getattribute__(self, "_target")
        name = object.__getattribute__(self, "_name")
        logger.info("[%s.__call__] args=%s kwargs=%s", name, _short(args), _short(kwargs))
        try:
            res = target(*args, **kwargs)
            logger.info("[%s.__call__] ok -> %s", name, _short(res))
            return res
        except Exception:
            logger.exception("[%s.__call__] error", name)
            raise

    # Support async context manager "async with ..."
    async def __aenter__(self):  # pragma: no cover
        target = object.__getattribute__(self, "_target")
        name = object.__getattribute__(self, "_name")
        logger.info("[%s.__aenter__]", name)
        aenter = getattr(target, "__aenter__", None)
        if aenter is not None:
            if inspect.iscoroutinefunction(aenter):
                res = await aenter()
            else:
                res = aenter()
            logger.info("[%s.__aenter__] ok -> %s", name, type(res).__name__)
        # Return proxy to keep logging for subsequent method calls
        return self

    async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover
        target = object.__getattribute__(self, "_target")
        name = object.__getattribute__(self, "_name")
        logger.info("[%s.__aexit__] exc=%s", name, exc)
        aexit = getattr(target, "__aexit__", None)
        if aexit is None:
            return False
        if inspect.iscoroutinefunction(aexit):
            res = await aexit(exc_type, exc, tb)
        else:
            res = aexit(exc_type, exc, tb)
        logger.info("[%s.__aexit__] ok -> %s", name, res)
        return res

    # Support sync context manager "with ..."
    def __enter__(self):  # pragma: no cover
        target = object.__getattribute__(self, "_target")
        name = object.__getattribute__(self, "_name")
        logger.info("[%s.__enter__]", name)
        enter = getattr(target, "__enter__", None)
        if enter is not None:
            try:
                res = enter()
                logger.info("[%s.__enter__] ok -> %s", name, type(res).__name__)
            except Exception:
                logger.exception("[%s.__enter__] error", name)
                raise
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover
        target = object.__getattribute__(self, "_target")
        name = object.__getattribute__(self, "_name")
        logger.info("[%s.__exit__] exc=%s", name, exc)
        exit_ = getattr(target, "__exit__", None)
        if exit_ is None:
            return False
        try:
            res = exit_(exc_type, exc, tb)
            logger.info("[%s.__exit__] ok -> %s", name, res)
            return res
        except Exception:
            logger.exception("[%s.__exit__] error", name)
            raise


def _short(obj: Any, limit: int = 200) -> str:
    try:
        s = repr(obj)
    except Exception:
        s = f"<{type(obj).__name__}>"
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


class LoggingLLM(_LoggingProxy):
    def __init__(self, llm: Any) -> None:
        super().__init__(llm, name="LLM")


class LoggingSTT(_LoggingProxy):
    def __init__(self, stt: Any) -> None:
        super().__init__(stt, name="STT")


class LoggingTTS(_LoggingProxy):
    def __init__(self, tts: Any) -> None:
        super().__init__(tts, name="TTS")
