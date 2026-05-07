from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def es_mock() -> AsyncMock:
    es = AsyncMock()
    es.search = AsyncMock()
    return es
