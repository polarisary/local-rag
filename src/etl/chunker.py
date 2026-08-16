from typing import List

from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core.schema import Document, BaseNode


def _is_markdown(doc: Document) -> bool:
    file_name = doc.metadata.get("file_name", "")
    return file_name.lower().endswith((".md", ".markdown"))


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 80,
) -> List[BaseNode]:
    md_parser = MarkdownNodeParser()
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes: List[BaseNode] = []
    for doc in documents:
        if _is_markdown(doc):
            md_nodes = md_parser.get_nodes_from_documents([doc], show_progress=False)
            # 对每个标题块做二次切分，防止超长章节超出 chunk_size
            for md_node in md_nodes:
                nodes.extend(
                    splitter.get_nodes_from_documents([md_node], show_progress=False)
                )
        else:
            nodes.extend(
                splitter.get_nodes_from_documents([doc], show_progress=False)
            )
    return nodes
