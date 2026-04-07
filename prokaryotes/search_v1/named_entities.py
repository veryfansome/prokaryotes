import logging
from abc import (
    ABC,
    abstractmethod,
)

from elasticsearch import AsyncElasticsearch, helpers

from prokaryotes.utils_v1.text_utils import (
    normalize_text_for_identity,
    text_to_md5,
)

logger = logging.getLogger(__name__)

named_entity_mappings = {
    "dynamic": "strict",
    "properties": {
        "emb": {
            "type":       "dense_vector",
            "dims":       256,
            "index":      True,
            "similarity": "cosine",
        },
        "name": {
            "type": "text",
            "fields": {
                "keyword": {
                    "type": "keyword",
                    "ignore_above": 256,
                }
            }
        },
    }
}


class NamedEntitySearcher(ABC):
    @property
    @abstractmethod
    def es(self) -> AsyncElasticsearch:
        pass

    async def index_named_entities(self, named_entities: list[str], named_entity_embs: list[list[float]]):
        seen_named_entity_ids = set()
        actions = []
        for idx, named_entity in enumerate(named_entities):
            named_entity = normalize_text_for_identity(named_entity)
            if not named_entity:
                continue
            named_entity_id = text_to_md5(named_entity)
            if named_entity_id in seen_named_entity_ids:
                continue
            seen_named_entity_ids.add(named_entity_id)
            actions.append({
                "_index": "named-entities",
                "_id": named_entity_id,
                "_op_type": "create",
                "_source": {"emb": named_entity_embs[idx], "name": named_entity}
            })
        if not actions:
            return
        success_cnt, errors = await helpers.async_bulk(self.es, actions, raise_on_error=False)
        if success_cnt:
            logger.info(f"Indexed {success_cnt} named entity(s)")
        if errors:
            skipped_cnt = len([e for e in errors if e.get("create", {}).get("status") == 409])  # type: ignore
            if skipped_cnt:
                logger.error(f"Skipped indexing {skipped_cnt} named entity(s)")
            error_cnt = len(errors) - skipped_cnt
            if error_cnt:
                logger.error(f"Failed to index {error_cnt} named entity(s)")

    async def search_named_entities(
            self,
            match: str,
            match_emb: list[float],
            knn_num_candidates: int = 100,
            knn_top_k: int = 10,
            min_score: float = 0.5,
    ) -> list[str]:
        match = normalize_text_for_identity(match)
        if not match:
            return []
        query = {
            "should": [
                {"match": {"name": {"query": match, "boost": 1.0}}},
                {"term": {"name.keyword": {"value": match, "boost": 2.0}}}
            ]
        }
        search_kwargs = {
            "index": "named-entities",
            "query": {
                "bool": query,
            },
        }
        if min_score is not None:
            search_kwargs["min_score"] = min_score
        if match_emb:
            search_kwargs["knn"] = {
                "field": "emb",
                "query_vector": match_emb,
                "boost": 1.0,
                "num_candidates": knn_num_candidates,
                "k": knn_top_k,
            }
        response = await self.es.search(**search_kwargs)
        hits = response["hits"]["hits"]
        for h in hits:
            name = h['_source']['name']
            logger.debug(f"Score: {h['_score']:.4f} | Named entity: {name}")
        seen_named_entities = set()
        named_entities = []
        for h in hits:
            named_entity = normalize_text_for_identity(h["_source"]["name"])
            if not named_entity or named_entity in seen_named_entities:
                continue
            seen_named_entities.add(named_entity)
            named_entities.append(named_entity)
        return named_entities
