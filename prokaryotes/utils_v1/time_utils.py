from datetime import (
    UTC,
    datetime,
)
from functools import cache
from zoneinfo import ZoneInfo


@cache
def tz(time_zone: str | None = None) -> ZoneInfo:
    return ZoneInfo("UTC" if not time_zone else time_zone)


def utc_now() -> datetime:
    return datetime.now(UTC)


def local_now(time_zone: str | None) -> datetime:
    return utc_now().astimezone(tz(time_zone))


def local_dt_str(dt: datetime, time_zone: str | None, fmt='%Y-%m-%d %H:%M') -> str:
    return dt.astimezone(tz(time_zone)).strftime(fmt)


def local_now_str(time_zone: str | None, fmt='%Y-%m-%d %H:%M') -> str:
    return local_now(time_zone).strftime(fmt)
