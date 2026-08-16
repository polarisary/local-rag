import logging
import os
from pathlib import Path
from typing import List

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


def _is_ignored(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    if name == ".git":
        return True
    if name.endswith("~") or name.endswith(".tmp"):
        return True
    return False


def _is_supported(path: Path) -> bool:
    ext = path.suffix.lower()
    return ext in (".pdf", ".docx", ".pptx", ".md", ".markdown")


def _extract_pdf(file_path: Path) -> List[Document]:
    """提取 PDF 文本，按页生成 Document。

    跨页 overlap：每页文本前面追加上一页末尾的 N 字符，
    防止跨页信息（如"工作单位"在 p.2 末尾、"工作内容"在 p.3 开头）
    被切到不同 chunk 导致检索丢失上下文。
    """
    import fitz

    _PAGE_OVERLAP_CHARS = 300
    docs: List[Document] = []
    prev_tail = ""
    with fitz.open(str(file_path)) as pdf:
        for page_idx, page in enumerate(pdf, start=1):
            text = page.get_text("text") or ""
            if not text.strip():
                continue
            # 前面追加上一页末尾内容，保证跨页上下文连续
            if prev_tail:
                text = prev_tail + "\n" + text
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": file_path.name,
                        "page_number": page_idx,
                        "source_path": str(file_path),
                    },
                )
            )
            # 保存本页末尾 N 字符供下一页使用
            prev_tail = text[-_PAGE_OVERLAP_CHARS:] if len(text) > _PAGE_OVERLAP_CHARS else text
    return docs


def _extract_markdown(file_path: Path) -> List[Document]:
    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return [
        Document(
            text=text,
            metadata={
                "file_name": file_path.name,
                "page_number": None,
                "source_path": str(file_path),
            },
        )
    ]


def _extract_docx_pptx(file_path: Path) -> List[Document]:
    from unstructured.partition.auto import partition

    elements = partition(filename=str(file_path))
    text_chunks = [str(el) for el in elements if str(el).strip()]
    text = "\n".join(text_chunks).strip()
    if not text:
        return []
    return [
        Document(
            text=text,
            metadata={
                "file_name": file_path.name,
                "page_number": None,
                "source_path": str(file_path),
            },
        )
    ]


def load_documents_from_folder(folder_path: str) -> List[Document]:
    root = Path(folder_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    all_docs: List[Document] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not _is_ignored(current_dir / d)]

        for fname in filenames:
            fpath = current_dir / fname
            if _is_ignored(fpath) or not _is_supported(fpath):
                continue
            try:
                ext = fpath.suffix.lower()
                if ext == ".pdf":
                    docs = _extract_pdf(fpath)
                elif ext in (".md", ".markdown"):
                    docs = _extract_markdown(fpath)
                else:
                    docs = _extract_docx_pptx(fpath)
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"解析文件失败 {fpath}: {e}")
                continue
    return all_docs
