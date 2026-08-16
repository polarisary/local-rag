import os
from typing import List

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore

from src.rag.vector_store import get_vector_store
from src.rag.llm_chain import get_embedding

load_dotenv()

# 相似度阈值：低于该分数的 chunk 视为无关，直接丢弃
# bge-m3 经验值：相关文档通常 >0.35，无关文档通常 <0.30
_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
_DEFAULT_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "4"))

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


def retrieve(query: str, top_k: int = None) -> List[NodeWithScore]:
    """检索 top_k 个候选 chunk，并按相似度阈值过滤掉无关文档。

    策略：先多召回一些（top_k * 2）保证候选池足够大，再用阈值过滤，
    最后截断到 top_k。这样既避免无关文档混入，也保证相关文档不被误删。
    """
    if top_k is None:
        top_k = _DEFAULT_TOP_K
    index = _build_index()
    # 多召回一倍，给阈值过滤留余量
    candidate_k = max(top_k * 2, top_k + 2)
    retriever = index.as_retriever(similarity_top_k=candidate_k)
    nodes = retriever.retrieve(query)

    # 按相似度阈值过滤
    filtered = []
    for node in nodes:
        score = getattr(node, "score", None) or 0.0
        if score >= _SIMILARITY_THRESHOLD:
            filtered.append(node)

    # 如果过滤后结果为空（极端情况：所有候选都低于阈值），
    # 退回原始 top_k 结果，避免完全无上下文
    if not filtered:
        return nodes[:top_k]

    return filtered[:top_k]
