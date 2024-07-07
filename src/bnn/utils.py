import datetime

__all__ = (
    'TIME_FORMAT',
    'NOW_STRING',
    'now_string',
)

TIME_FORMAT = '%Y-%m-%dT%H:%M:%S'


def now_string() -> str:
    return datetime.datetime.now().strftime(TIME_FORMAT)


NOW_STRING = now_string()
