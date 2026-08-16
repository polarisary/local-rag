import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
_LLM_API_KEY = os.getenv("LLM_API_KEY", "")
_LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3-embedding")

_embedding_singleton = None


_HF_ENDPOINTS = [ep for ep in [
    os.getenv("HF_ENDPOINT"),
    "https://hf-mirror.com",
    "https://huggingface.co",
] if ep]


def _dedupe_endpoints(endpoints):
    seen = set()
    result = []
    for ep in endpoints:
        key = ep.rstrip("/")
        if key not in seen:
            seen.add(key)
            result.append(ep)
    return result


def _apply_hf_endpoint(endpoint: str, modules_to_clear=None) -> str:
    """Set HF endpoint to env + all huggingface* modules. Returns previous env value."""
    if modules_to_clear is None:
        modules_to_clear = [
            "huggingface_hub",
            "huggingface_hub.constants",
            "transformers",
            "transformers.utils.hub",
            "sentence_transformers",
            "llama_index.embeddings.huggingface",
        ]
    import sys as _sys

    for mod in modules_to_clear:
        _sys.modules.pop(mod, None)

    os.environ["HF_ENDPOINT"] = endpoint
    # re-import and patch constants immediately
    from huggingface_hub import constants as hf_constants

    hf_constants.ENDPOINT = endpoint.rstrip("/")
    return os.environ.get("HF_ENDPOINT")


def _try_load_embedding(model_name: str) -> BaseEmbedding:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # 1) 如果本地已经有缓存，直接用离线模式加载（最快、避免网络问题）
    try:
        return HuggingFaceEmbedding(model_name=model_name, local_files_only=True)
    except Exception:
        pass

    # 2) 在线加载，按端点依次尝试
    last_exc = None
    tried = []
    for endpoint in _dedupe_endpoints(_HF_ENDPOINTS):
        tried.append(endpoint)
        os.environ["HF_ENDPOINT"] = endpoint
        try:
            from huggingface_hub import constants as hf_constants

            hf_constants.ENDPOINT = endpoint.rstrip("/")
        except Exception:
            pass
        try:
            return HuggingFaceEmbedding(model_name=model_name)
        except Exception as exc:
            last_exc = exc
            time.sleep(3)
    raise RuntimeError(
        f"Failed to download embedding model {model_name} "
        f"after trying endpoints {tried}: {last_exc}"
    )


RAG_PROMPT_TMPL = PromptTemplate(
    "你是企业知识库助手，基于下面参考文档回答用户问题。\n"
    "如果文档没有答案，直接说明知识库没有相关信息，不要编造。\n"
    "直接回答问题即可，不要在回答中列出引用来源，系统会自动附加。\n\n"
    "【参考文档】\n"
    "{context_str}\n\n"
    "用户问题：{user_query}\n"
)


def _call_llm(prompt: str) -> str:
    """用 openai SDK 直接调用 chat completions API，绕过 OpenAILike 的请求构造问题。"""
    from openai import OpenAI

    client = OpenAI(api_key=_LLM_API_KEY, base_url=_LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=_LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是企业知识库助手，基于参考文档回答问题。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""


def get_embedding() -> BaseEmbedding:
    global _embedding_singleton
    if _embedding_singleton is None:
        _embedding_singleton = _try_load_embedding(_EMBEDDING_MODEL)
    return _embedding_singleton


def _build_context(nodes) -> str:
    blocks = []
    for idx, node in enumerate(nodes, start=1):
        meta = node.metadata or {}
        fn = meta.get("file_name", "unknown")
        pn = meta.get("page_number")
        src = f"[{idx}] 文件: {fn}"
        if pn is not None:
            src += f", 页码: {pn}"
        text = node.get_content() or ""
        blocks.append(f"{src}\n内容: {text}")
    return "\n\n".join(blocks)


def _collect_references(nodes) -> List[Dict[str, Any]]:
    seen = set()
    refs: List[Dict[str, Any]] = []
    for node in nodes:
        meta = node.metadata or {}
        fn = meta.get("file_name", "unknown")
        pn = meta.get("page_number")
        key = (fn, pn)
        if key in seen:
            continue
        seen.add(key)
        refs.append({"file_name": fn, "page_number": pn})
    return refs


def _strip_llm_references(text: str) -> str:
    """清理 LLM 回答中自带的引用来源部分，统一由代码附加。"""
    import re
    # 匹配末尾的"引用来源：..."、"引用：..."、"References:..."等整段
    text = re.sub(
        r"\n*(?:引用来源|引用|参考资料|References?|Source[s]?)[：:].*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


def query_rag(user_query: str) -> Dict[str, Any]:
    from src.rag.retriever import retrieve

    nodes = retrieve(user_query, top_k=4)
    context_str = _build_context(nodes)
    refs = _collect_references(nodes)

    prompt = RAG_PROMPT_TMPL.format(context_str=context_str, user_query=user_query)
    answer = _strip_llm_references(_call_llm(prompt))

    if refs:
        ref_lines = []
        for i, r in enumerate(refs, start=1):
            if r["page_number"] is None:
                ref_lines.append(f"- {r['file_name']}")
            else:
                ref_lines.append(f"- {r['file_name']} (p.{r['page_number']})")
        answer += "\n\n**引用来源：**\n" + "\n".join(ref_lines)

    return {"answer": answer, "references": refs}
