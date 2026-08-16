import os
from pathlib import Path
from typing import List

from llama_index.core.schema import Document


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
    import fitz

    docs: List[Document] = []
    with fitz.open(str(file_path)) as pdf:
        for page_idx, page in enumerate(pdf, start=1):
            text = page.get_text("text") or ""
            if not text.strip():
                continue
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
                print(f"[WARN] Failed to parse {fpath}: {e}")
                continue
    return all_docs
