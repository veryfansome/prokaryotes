"""Tier B: ES bootstrap produced the unified-conversation indices.

`scripts/search_init.py` (run by the docker-compose `elasticsearch-init` service) creates `conversations` and
`turn-executions` with strict mappings.

The bootstrap no longer creates the legacy `context-partitions` index but does not delete a pre-existing one, so a
migrated-in-place ES may still carry a stale one — hence no assertion of its absence. The contract is just "the new
indices exist and are strict".
"""

from __future__ import annotations

import pytest

from prokaryotes.search_v1.conversations import CONVERSATIONS_INDEX, TURN_EXECUTIONS_INDEX

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("web_harness", ["anthropic"], indirect=True)
@pytest.mark.asyncio(loop_scope="session")
async def test_conversation_indices_bootstrapped_with_strict_mappings(web_harness):
    es = web_harness.search_client.es

    for index in (CONVERSATIONS_INDEX, TURN_EXECUTIONS_INDEX):
        assert await es.indices.exists(index=index), f"{index} was not bootstrapped"
        mapping = await es.indices.get_mapping(index=index)
        assert mapping[index]["mappings"]["dynamic"] == "strict"
