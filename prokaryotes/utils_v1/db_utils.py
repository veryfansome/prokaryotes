import os

import asyncpg
from asyncpg import Pool
from redis.asyncio import Redis


async def get_postgres_pool() -> Pool:
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "prokaryotes")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    ssl_mode = os.getenv("POSTGRES_SSL_MODE", "disable")
    if host and password and user:
        return await asyncpg.create_pool(
            dsn=f"postgresql://{user}:{password}@{host}:{port}/{db}",
            ssl=ssl_mode,
            min_size=int(os.getenv("POSTGRES_POOL_MIN_SIZE", "1")),
            max_size=int(os.getenv("POSTGRES_POOL_MAX_SIZE", "3")),
        )
    raise RuntimeError("Unable to initialize postgres pool")


def get_redis_client() -> Redis:
    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT", "6379")
    db = os.getenv("REDIS_DB", "0")
    if host:
        return Redis.from_url(f"redis://{host}:{port}/{db}", decode_responses=False)
    raise RuntimeError("Unable to initialize Redis client")
