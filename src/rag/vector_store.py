import logging
import os
from typing import List

from dotenv import load_dotenv
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import utility, connections

load_dotenv()

logger = logging.getLogger(__name__)

_MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
_MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
_MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "enterprise_rag_demo")

_store_singleton = None


def _connect_once():
    alias = "default"
    try:
        connections.connect(
            alias=alias,
            host=_MILVUS_HOST,
            port=_MILVUS_PORT,
        )
        logger.info(f"Milvus 连接成功: {_MILVUS_HOST}:{_MILVUS_PORT}")
    except Exception as e:
        logger.error(f"Milvus 连接失败 {_MILVUS_HOST}:{_MILVUS_PORT}: {e}")
        raise


def get_vector_store() -> MilvusVectorStore:
    global _store_singleton
    if _store_singleton is None:
        _connect_once()
        _store_singleton = MilvusVectorStore(
            uri=f"http://{_MILVUS_HOST}:{_MILVUS_PORT}",
            collection_name=_MILVUS_COLLECTION,
            dim=1024,
            overwrite=False,
        )
    return _store_singleton


def clear_collection() -> None:
    _connect_once()
    if utility.has_collection(_MILVUS_COLLECTION):
        utility.drop_collection(_MILVUS_COLLECTION)
    global _store_singleton
    _store_singleton = None


def add_nodes(nodes: List[BaseNode]) -> None:
    from src.rag.llm_chain import get_embedding

    embed_model = get_embedding()
    for node in nodes:
        node.embedding = embed_model.get_text_embedding(node.get_content())

    store = get_vector_store()
    store.add(nodes)
