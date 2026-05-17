from prokaryotes.context_v1 import (
    PartitionCompactor,
    PartitionSyncer,
    _partition_can_follow_client,
    get_redis_client,
)
from prokaryotes.harness_v1.base import HarnessBase
from prokaryotes.web_v1 import (
    PartitionCompactor as WebPartitionCompactor,
)
from prokaryotes.web_v1 import (
    PartitionSyncer as WebPartitionSyncer,
)
from prokaryotes.web_v1 import (
    WebBase,
)
from prokaryotes.web_v1 import (
    _partition_can_follow_client as web_partition_can_follow_client,
)
from prokaryotes.web_v1 import (
    get_redis_client as web_get_redis_client,
)
from prokaryotes.web_v1.compaction import CompactionStatusHandler


def test_context_v1_surface_is_reexported_from_web_v1():
    assert WebPartitionCompactor is PartitionCompactor
    assert WebPartitionSyncer is PartitionSyncer
    assert web_partition_can_follow_client is _partition_can_follow_client
    assert web_get_redis_client is get_redis_client


def test_webbase_inherits_new_shared_layers_without_new_abstractmethods():
    assert issubclass(WebBase, HarnessBase)
    assert issubclass(WebBase, CompactionStatusHandler)
    assert WebBase.__abstractmethods__ == frozenset()
    WebBase("scripts/static")
