from typing import List

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore

from src.rag.vector_store import get_vector_store
from src.rag.llm_chain import get_embedding


_index_singleton = None


def _build_index() -> VectorStoreIndex:
    global _index_singleton
    if _index_singleton is None:
        store = get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=store)
        embed_model: BaseEmbedding = get_embedding()
        _index_singleton = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=embed_model,
        )
    return _index_singleton


def retrieve(query: str, top_k: int = 4) -> List[NodeWithScore]:
    index = _build_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return nodes
