import time
from typing import Any, Dict

from src.etl.document_loader import load_documents_from_folder
from src.etl.chunker import chunk_documents
from src.rag.vector_store import clear_collection, add_nodes


def run_folder_ingest(folder_path: str) -> Dict[str, Any]:
    start = time.time()

    docs = load_documents_from_folder(folder_path)
    doc_count = len(docs)

    nodes = chunk_documents(docs)
    node_count = len(nodes)

    clear_collection()
    if node_count > 0:
        add_nodes(nodes)

    elapsed = round(time.time() - start, 2)
    return {
        "folder_path": folder_path,
        "documents": doc_count,
        "chunks": node_count,
        "elapsed_seconds": elapsed,
    }
