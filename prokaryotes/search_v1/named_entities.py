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

    @staticmethod
    def _lexical_score_info(hit: dict) -> tuple[float, set[str]]:
        matched_queries = hit.get("matched_queries")
        if isinstance(matched_queries, dict):
            lexical_score = 0.0
            for value in matched_queries.values():
                if isinstance(value, (int, float)):
                    lexical_score += value
            return lexical_score, set(matched_queries)
        if isinstance(matched_queries, list):
            return 0.0, {
                matched_query for matched_query in matched_queries if isinstance(matched_query, str)
            }
        return 0.0, set()

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
            exact_match_boost: float = 2.0,
            knn_num_candidates: int = 100,
            knn_top_k: int = 10,
            knn_boost: float = 1.0,
            lexical_minimum_should_match: str | int | None = None,
            match_boost: float = 1.0,
            min_score: float = 0.5,
            track_matched_queries: bool = True,
    ) -> list[str]:
        match = normalize_text_for_identity(match)
        if not match:
            return []
        match_query = {
            "query": match,
            "boost": match_boost,
            "_name": "named_entity_name_match",
        }
        if lexical_minimum_should_match is not None:
            match_query["minimum_should_match"] = lexical_minimum_should_match
        query = {
            "should": [
                {"match": {"name": match_query}},
                {
                    "term": {
                        "name.keyword": {
                            "value": match,
                            "boost": exact_match_boost,
                            "_name": "named_entity_name_exact",
                        }
                    }
                }
            ]
        }
        search_kwargs = {
            "index": "named-entities",
            "query": {
                "bool": query,
            },
        }
        if track_matched_queries:
            search_kwargs["include_named_queries_score"] = True
        if min_score is not None:
            search_kwargs["min_score"] = min_score
        if match_emb:
            search_kwargs["knn"] = {
                "field": "emb",
                "query_vector": match_emb,
                "boost": knn_boost,
                "num_candidates": knn_num_candidates,
                "k": knn_top_k,
            }
        response = await self.es.search(**search_kwargs)
        hits = response["hits"]["hits"]
        for h in hits:
            name = h["_source"]["name"]
            total_score = float(h.get("_score", 0.0) or 0.0)
            lexical_score, lexical_match_queries = self._lexical_score_info(h)
            semantic_score_approx = max(0.0, total_score - lexical_score)
            logger.debug(
                f"Score: {total_score:.4f}"
                f" | lexical: {lexical_score:.4f}"
                f" | semantic_approx: {semantic_score_approx:.4f}"
                f" | matched_queries: {sorted(lexical_match_queries)}"
                f" | Named entity: {name}"
            )
        seen_named_entities = set()
        named_entities = []
        for h in hits:
            named_entity = normalize_text_for_identity(h["_source"]["name"])
            if not named_entity or named_entity in seen_named_entities:
                continue
            seen_named_entities.add(named_entity)
            named_entities.append(named_entity)
        return named_entities
