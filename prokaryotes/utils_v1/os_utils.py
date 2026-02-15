import os
import platform
import pwd
from functools import (
    cache,
)

import psutil


@cache
def get_cwd() -> str:
    return os.getcwd()


@cache
def get_listening_ports() -> list[int]:
    proc = psutil.Process(get_pid())
    listening = [
        conn.laddr.port 
        for conn in proc.net_connections() 
        if conn.status == 'LISTEN'
    ]
    return listening


@cache
def get_pid() -> int:
    return os.getpid()


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
def uid_to_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except Exception:
        return str(uid)
