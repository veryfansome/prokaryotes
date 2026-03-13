import grp
import os
import platform
import pwd
import stat
import time
from datetime import datetime
from functools import (
    cache,
    lru_cache,
)

@lru_cache(maxsize=128)
def format_st_mtime(mtime: int) -> str:
    dt = datetime.fromtimestamp(mtime)
    # Like ls
    six_months_ago = time.time() - (6 * 30 * 24 * 60 * 60)
    if mtime < six_months_ago or mtime > time.time():
        return dt.strftime("%b %e  %Y")
    else:
        return dt.strftime("%b %e %H:%M")

@lru_cache(maxsize=128)
def format_st_size(size: int):
    # Define the units and the divisor (1024 for binary units)
    for unit in ['B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            # Return as integer for bytes, or with 1 decimal for larger units
            return f"{size:3.1f}{unit}" if unit != 'B' else f"{int(size)}{unit}"
        size /= 1024.0
    return f"{size:.1f}Y"

@cache
def get_cwd() -> str:
    return os.getcwd()

@cache
def get_platform() -> str:
    return platform.platform(terse=True)

@cache
def get_process_uid() -> int:
    return os.getuid()

@cache
def get_python_version() -> str:
    return platform.python_version()

@cache
def gid_to_name(gid: int) -> str:
    try:
        return grp.getgrgid(gid).gr_name
    except Exception:
        return str(gid)

@cache
def st_mode_to_symbolic_mode(st_mode: int) -> str:
    return stat.filemode(st_mode)

@cache
def uid_to_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except Exception:
        return str(uid)
