import logging
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

logger = logging.getLogger(__name__)

_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
_LLM_API_KEY = os.getenv("LLM_API_KEY", "")
_LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
_LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3-embedding")

_embedding_singleton = None


def _try_load_embedding(model_name: str) -> BaseEmbedding:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # 1) 如果本地已经有缓存，直接用离线模式加载（最快、避免网络问题）
    try:
        return HuggingFaceEmbedding(model_name=model_name, local_files_only=True)
    except Exception:
        pass

    # 2) 在线加载
    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        from huggingface_hub import constants as hf_constants

        hf_constants.ENDPOINT = endpoint.rstrip("/")
    except Exception:
        pass
    try:
        return HuggingFaceEmbedding(model_name=model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model {model_name} "
            f"via {endpoint}: {exc}"
        ) from exc


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
    from openai import OpenAI, APITimeoutError, APIConnectionError

    client = OpenAI(api_key=_LLM_API_KEY, base_url=_LLM_BASE_URL, timeout=_LLM_TIMEOUT)
    try:
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
    except APITimeoutError as e:
        logger.error(f"LLM 调用超时 ({_LLM_TIMEOUT}s): {e}")
        raise RuntimeError(f"LLM 调用超时，请稍后重试或增大 LLM_TIMEOUT 配置") from e
    except APIConnectionError as e:
        logger.error(f"LLM 连接失败: {e}")
        raise RuntimeError(f"LLM 服务连接失败，请检查 LLM_BASE_URL/LLM_API_KEY 配置") from e


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
    """按文件名去重，并收集同一文件涉及的所有页码，用于合并展示。"""
    file_pages: Dict[str, set] = {}
    for node in nodes:
        meta = node.metadata or {}
        fn = meta.get("file_name", "unknown")
        pn = meta.get("page_number")
        if fn not in file_pages:
            file_pages[fn] = set()
        if pn is not None:
            file_pages[fn].add(pn)
    refs = []
    for fn, pages in file_pages.items():
        refs.append({"file_name": fn, "pages": sorted(pages) if pages else []})
    return refs


def _strip_llm_references(text: str) -> str:
    """清理 LLM 回答中自带的引用来源部分，统一由代码附加。"""
    # 匹配末尾的"引用来源：..."、"引用：..."、"References:..."等整段
    text = re.sub(
        r"\n*(?:引用来源|引用|参考资料|References?|Source[s]?)[：:].*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


def _is_no_info_answer(answer: str) -> bool:
    """识别 LLM 回答是否表示"知识库没有相关信息"。

    当 LLM 判断参考文档无法回答用户问题时，会返回类似"没有相关信息"、
    "知识库中没有"、"未找到相关内容"等措辞。此时不应展示引用来源，
    因为引用的文档实际上与问题无关。
    """
    patterns = [
        r"没有相关(信息|内容|答案|资料)",
        r"知识库(中)?没有",
        r"未(找到|发现|包含|涉及)相关",
        r"无法(从|在).{0,10}(参考文档|知识库|文档).{0,10}(找到|找到答案|回答)",
        r"文档(中)?(未|没有|无法).{0,10}(提供|包含|涉及|提到|回答)",
        r"不(在|属于).{0,10}(知识库|参考文档|文档).{0,10}(范围|内容)",
        r"(没有|未)在.{0,10}(参考文档|知识库|文档).{0,10}(中)?(找到|提及|涉及|说明|回答)",
        r"参考文档(中)?(没有|未|无法).{0,10}(提供|找到|包含|涉及|回答)",
        r"no (relevant )?information",
        r"not (found|covered|mentioned|available) (in|in the) (knowledge base|document|reference)",
        r"cannot (find|answer|provide) (any |relevant )?(information|answer)",
    ]
    for p in patterns:
        if re.search(p, answer, flags=re.IGNORECASE):
            return True
    return False


def query_rag(user_query: str, top_k: int = None) -> Dict[str, Any]:
    from src.rag.retriever import retrieve

    nodes = retrieve(user_query, top_k=top_k)
    context_str = _build_context(nodes)
    refs = _collect_references(nodes)

    prompt = RAG_PROMPT_TMPL.format(context_str=context_str, user_query=user_query)
    answer = _strip_llm_references(_call_llm(prompt))

    # 如果 LLM 判断"没有相关信息"，不展示引用来源（因为引用的文档实际无关）
    if refs and not _is_no_info_answer(answer):
        ref_lines = []
        for r in refs:
            fn = r["file_name"]
            pages = r.get("pages", [])
            if pages:
                ref_lines.append(f"- {fn} (p.{', p.'.join(str(p) for p in pages)})")
            else:
                ref_lines.append(f"- {fn}")
        answer += "\n\n**引用来源：**\n" + "\n".join(ref_lines)
    elif _is_no_info_answer(answer):
        # 无相关信息时清空 references，保持 API 返回一致性
        refs = []

    return {"answer": answer, "references": refs}
