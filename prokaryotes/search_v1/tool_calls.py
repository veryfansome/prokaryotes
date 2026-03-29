import logging
from abc import (
    ABC,
    abstractmethod,
)
from datetime import datetime, timezone
from elasticsearch import AsyncElasticsearch

from prokaryotes.models_v1 import ToolCallDoc

logger = logging.getLogger(__name__)

tool_call_mappings = {
    "dynamic": "strict",
    "properties": {
        "label_cnt": {"type": "integer"},
        "labels":    {"type": "keyword"},
        "output":    {"type": "object", "enabled": False},
        "anchors": {
            "type": "nested",
            "properties": {
                "text": {
                    "type":            "text",
                    "analyzer":        "standard",
                    "search_analyzer": "custom_query_analyzer",
                },
                "text_emb": {
                    "type":       "dense_vector",
                    "dims":       256,
                    "index":      True,
                    "similarity": "cosine",
                },
            },
        },
        "updated_at": {"type": "date"},
    }
}


class ToolCallSearcher(ABC):
    @property
    @abstractmethod
    def es(self) -> AsyncElasticsearch:
        pass

    async def index_tool_call(
            self,
            labels: list[str],
            output: str,
            anchor_emb: list[float] = None,
            anchor_text: str = None,
    ):
        now = datetime.now(timezone.utc)
        doc = ToolCallDoc(
            anchors=[], label_cnt=len(labels), labels=labels, output=output, updated_at=now,
        )
        anchor = {}
        if anchor_text:
            anchor["text"] = anchor_text
            if anchor_emb:
                anchor["text_emb"] = anchor_emb
        try:
            result = await self.es.index(
                index="tool-calls",
                document=(doc.model_dump() | {"anchors": [anchor]} if anchor_text else {})
            )
            doc.doc_id = result["_id"]
            return doc
        except Exception:
            logger.exception(f"Failed to index {doc}")

    async def search_tool_call_by_anchor(
            self,
            match: str,
            match_emb: list[float],
            min_score: float = 0.5,
            knn_num_candidates: int = 100,
            knn_top_k: int = 30,
            top_k: int = 10,
    ) -> list[ToolCallDoc]:
        response = await self.es.search(
            index="tool-calls",
            knn={
                "field": "anchors.text_emb",
                "query_vector": match_emb,
                "boost": 1.0,
                "k": knn_top_k,
                "num_candidates": knn_num_candidates,
            },
            query={
                "bool": {
                    "should": [
                        {
                            "nested": {
                                "path": "anchors",
                                "query": {
                                    "match": {
                                        "anchors.text": {
                                            "query": match,
                                            "boost": 1.0
                                        }
                                    }
                                },
                            }
                        }
                    ]
                }
            },
            size=top_k,
        )
        hits = response["hits"]["hits"]
        for h in hits:
            labels = h['_source'].get('labels')
            logger.debug(f"ToolCallDoc ID: {h['_id']} | Score: {h['_score']:.4f} | Labels: {labels}")
        # TODO: Not efficient or scalable, pass min_score to es.search()
        return [ToolCallDoc(doc_id=h["_id"], **h["_source"])
                for h in hits if not min_score or h["_score"] >= min_score]

    async def search_tool_call_by_labels(self, filter_labels: list[str]) -> list[ToolCallDoc]:
        response = await self.es.search(
            index="tool-calls",
            query={
                "bool": {
                    "filter": [
                        *[{"term": {"labels": label}} for label in filter_labels],
                        {"term": {"label_cnt": len(filter_labels)}}  # Exact match
                    ]
                }
            }
        )
        hits = response["hits"]["hits"]
        for h in hits:
            labels = h['_source'].get('labels')
            logger.debug(f"ToolCallDoc ID: {h['_id']} | Labels: {labels}")
        return [ToolCallDoc(doc_id=h["_id"], **h["_source"]) for h in hits]

    async def update_tool_call(
            self,
            tool_call: ToolCallDoc,
            anchor_emb: list[float] = None,
            anchor_text: str = None,
            output: str = None,
    ):
        now = datetime.now(timezone.utc)
        script = [
            "ctx._source.updated_at = params.updated_at;",
        ]
        tool_call.updated_at = now

        anchor = {}
        if anchor_text:
            script.append("ctx._source.anchors.add(params.anchor);")
            anchor["text"] = anchor_text
            if anchor_emb:
                anchor["text_emb"] = anchor_emb
            tool_call.add_anchor(anchor_text)
        if output:
            script.append("ctx._source.output = params.output;")
            tool_call.output = output
        try:
            await self.es.update(
                index="tool-calls",
                id=tool_call.doc_id,
                body={
                    "script": {
                        "source": "\n".join(script),
                        "params": (
                            {"updated_at": now.isoformat()}
                            | ({"anchor": anchor} if anchor_text else {})
                            | ({"output": output} if output else {})
                        )
                    }
                }
            )
        except Exception:
            logger.exception(f"Failed to update tool-calls document: id={tool_call.doc_id}")
