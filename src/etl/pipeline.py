import logging
import time
from typing import Any, Dict

from src.etl.document_loader import load_documents_from_folder
from src.etl.chunker import chunk_documents
from src.rag.vector_store import clear_collection, add_nodes

logger = logging.getLogger(__name__)


def run_folder_ingest(folder_path: str) -> Dict[str, Any]:
    start = time.time()

    try:
        docs = load_documents_from_folder(folder_path)
    except (FileNotFoundError, NotADirectoryError) as e:
        logger.error(f"加载文档失败，路径不可用: {folder_path} - {e}")
        raise
    except Exception as e:
        logger.exception(f"加载文档异常: {folder_path}")
        raise

    doc_count = len(docs)
    logger.info(f"加载 {doc_count} 个文档 from {folder_path}")

    try:
        nodes = chunk_documents(docs)
    except Exception as e:
        logger.exception("文档分块失败")
        raise
    node_count = len(nodes)
    logger.info(f"生成 {node_count} 个 chunk")

    try:
        clear_collection()
        if node_count > 0:
            add_nodes(nodes)
    except Exception as e:
        logger.exception("写入 Milvus 失败")
        raise
    logger.info("向量入库完成")

    elapsed = round(time.time() - start, 2)
    return {
        "folder_path": folder_path,
        "documents": doc_count,
        "chunks": node_count,
        "elapsed_seconds": elapsed,
    }
