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
            "type":            "text",
            "analyzer":        "standard",
            "search_analyzer": "custom_query_analyzer",
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
            excluded_entities: list[str] | None = None,
            keyword_match_boost: float = 2.0,
            knn_boost: float = 1.0,
            knn_num_candidates: int = 100,
            knn_top_k: int = 10,
            lexical_match_boost: float = 1.0,
            min_score: float = 0.5,
    ) -> list[str]:
        if not match:
            return []
        deduped_excluded_entities = []
        seen_excluded_entities = set()
        for named_entity in excluded_entities or []:
            if not named_entity or named_entity in seen_excluded_entities:
                continue
            seen_excluded_entities.add(named_entity)
            deduped_excluded_entities.append(named_entity)
        query = {
            "should": [
                {
                    "match": {
                        "name": {
                            "query": match,
                            "boost": lexical_match_boost,
                            "_name": "named_entity_name_match",
                        }
                    }
                },
                {
                    "term": {
                        "name.keyword": {
                            "value": match,
                            "boost": keyword_match_boost,
                            "_name": "named_entity_name_exact",
                        }
                    }
                }
            ]
        }
        if deduped_excluded_entities:
            query["must_not"] = [{"terms": {"name.keyword": deduped_excluded_entities}}]
        search_kwargs = {
            "index": "named-entities",
            "include_named_queries_score": True,
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
                "boost": knn_boost,
                "num_candidates": knn_num_candidates,
                "k": knn_top_k,
            }
            if deduped_excluded_entities:
                search_kwargs["knn"]["filter"] = {
                    "bool": {
                        "must_not": [{"terms": {"name.keyword": deduped_excluded_entities}}],
                    }
                }
        response = await self.es.search(**search_kwargs)
        hits = response["hits"]["hits"]
        for h in hits:
            name = h["_source"]["name"]
            total_score = float(h.get("_score", 0.0) or 0.0)
            matched_queries = h.get("matched_queries")
            lexical_score = 0.0
            matched_query_names = set()
            if isinstance(matched_queries, dict):
                matched_query_names = set(matched_queries)
                lexical_score = sum(
                    value for value in matched_queries.values()
                    if isinstance(value, (int, float))
                )
            elif isinstance(matched_queries, list):
                matched_query_names = {
                    matched_query for matched_query in matched_queries if isinstance(matched_query, str)
                }
            semantic_score = max(0.0, total_score - lexical_score)
            keyword_hit = "named_entity_name_exact" in matched_query_names
            logger.debug(
                f"Score: {total_score:.4f}"
                f" | lex: {lexical_score:.4f}"
                f" | sem: {semantic_score:.4f}"
                f" | final: {total_score:.4f}"
                f" | keyword: {keyword_hit}"
                f" | matched: {sorted(matched_query_names)}"
                f" | named_entity: {name}"
            )
        return list(dict.fromkeys(
            named_entity for named_entity in (h["_source"]["name"] for h in hits)
            if named_entity
        ))
