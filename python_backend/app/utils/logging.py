import logging
from datetime import datetime


SENSITIVE_WORDS = ("api_key", "apikey", "authorization", "token", "secret")


class PrivacyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage()).lower()
        if any(word in msg for word in SENSITIVE_WORDS):
            record.msg = "[redacted sensitive log message]"
            record.args = ()
        return True


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger().addFilter(PrivacyFilter())
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def current_log_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")
