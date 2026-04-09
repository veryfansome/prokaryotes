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

topic_mappings = {
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


class TopicSearcher(ABC):
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

    async def index_topics(self, topics: list[str], topic_embs: list[list[float]]):
        seen_topic_ids = set()
        actions = []
        for idx, topic in enumerate(topics):
            topic = normalize_text_for_identity(topic)
            if not topic:
                continue
            topic_id = text_to_md5(topic)
            if topic_id in seen_topic_ids:
                continue
            seen_topic_ids.add(topic_id)
            actions.append({
                "_index": "topics",
                "_id": topic_id,
                "_op_type": "create",
                "_source": {"emb": topic_embs[idx], "name": topic}
            })
        if not actions:
            return
        success_cnt, errors = await helpers.async_bulk(self.es, actions, raise_on_error=False)
        if success_cnt:
            logger.info(f"Indexed {success_cnt} topic(s)")
        if errors:
            skipped_cnt = len([e for e in errors if e.get("create", {}).get("status") == 409])  # type: ignore 
            if skipped_cnt:
                logger.error(f"Skipped indexing {skipped_cnt} topic(s)")
            error_cnt = len(errors) - skipped_cnt
            if error_cnt:
                logger.error(f"Failed to index {error_cnt} topic(s)")

    async def search_topics(
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
            "_name": "topic_name_match",
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
                            "_name": "topic_name_exact",
                        }
                    }
                }
            ]
        }
        search_kwargs = {
            "index": "topics",
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
                f" | Topic: {name}"
            )
        seen_topics = set()
        topics = []
        for h in hits:
            topic = normalize_text_for_identity(h["_source"]["name"])
            if not topic or topic in seen_topics:
                continue
            seen_topics.add(topic)
            topics.append(topic)
        return topics
