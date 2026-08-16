import os
import re
from collections import defaultdict
from typing import Dict, List

from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core.schema import Document, BaseNode

load_dotenv()

_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# 通用结构型标题匹配：Label：Value 或 Label: Value（每行开头）
# 支持："工作单位：Shopee深圳"、"项目名称：Flink平台"、"客户：XXX"、"部门：XXX" 等
# Label长度 2~15（单个"项目1"等短标题也覆盖），Value 单行长度<=80（防止误匹配长句）
_HEADER_VALUE_RE = re.compile(
    r"^\s*(.{2,15}?)[：:]\s*(.{1,80})\s*$"
)

# Label黑名单：联系信息、元数据等噪音（即便符合结构也不提取）
_LABEL_BLACKLIST = {
    "e-mail", "email", "邮箱", "mail",
    "电话", "手机", "tel", "phone", "mobile",
    "地址", "住址", "address",
    "姓名", "name", "性别", "age", "年龄",
    "主修", "主课",
}

# Label黑名单（前缀匹配）：只要Label以前缀开头即跳过
# "项目描述/主要职责"类通常跟随长正文描述，不适合作为上下文传播
_LABEL_PREFIX_BLACKLIST = (
    "项目描述",
    "职责描述",
    "工作描述",
    "职位描述",
    "岗位描述",
    "功能描述",
    "需求描述",
    "问题描述",
    "业务描述",
    "描述：",
)


def _is_markdown(doc: Document) -> bool:
    file_name = doc.metadata.get("file_name", "")
    return file_name.lower().endswith((".md", ".markdown"))


def _split_document(
    doc: Document,
    md_parser: MarkdownNodeParser,
    splitter: SentenceSplitter,
) -> List[BaseNode]:
    if _is_markdown(doc):
        md_nodes = md_parser.get_nodes_from_documents([doc], show_progress=False)
        nodes: List[BaseNode] = []
        for md_node in md_nodes:
            nodes.extend(
                splitter.get_nodes_from_documents([md_node], show_progress=False)
            )
        return nodes
    return splitter.get_nodes_from_documents([doc], show_progress=False)


def _extract_headers(text: str) -> Dict[str, str]:
    """从文本中提取所有 Label：Value 结构型标题行。

    过滤策略：
    - Label 清洗：去掉 PDF 表格分隔符 | 等前后噪音字符
    - 精确黑名单 Label（联系方式等）跳过
    - 前缀黑名单 Label（xx描述 等正文类）跳过
    - Label 包含句末标点（。，,.;；!?等）视为一句话，跳过
    - Value 长度 > 50 且包含中文标点时视为正文描述，跳过
    """
    headers: Dict[str, str] = {}
    for line in text.splitlines():
        m = _HEADER_VALUE_RE.match(line)
        if not m:
            continue
        label = m.group(1).strip()
        value = m.group(2).strip()

        # 清洗Label：去掉可能的PDF表格分隔符|前缀、以及前后噪音字符
        label = label.lstrip("|｜·•■◆▎-–— \t")
        label = label.strip()
        if len(label) < 2:
            continue

        # 精确黑名单
        if label.lower() in _LABEL_BLACKLIST:
            continue
        # 前缀黑名单（xx描述 等正文类标题）
        if label.startswith(_LABEL_PREFIX_BLACKLIST):
            continue
        # Label本身是一句话（有句末标点）
        if re.search(r"[。，,.;；!?]", label):
            continue
        # Value是长正文且含中文标点（如"项目描述：一长段话..."的变种）
        if len(value) > 50 and re.search(r"[。，,；]", value):
            continue
        if label and value:
            headers[label] = value
    return headers


# 标识"换上下文边界"的 Label：当这些 Label 出现新值时，
# 清理掉 active 中属于上一段的项目/职责类标签，避免跨公司残留
_BOUNDARY_LABELS = {"工作单位", "公司", "公司名称", "客户", "部门"}
_PROJECT_LABEL_PREFIXES = ("项目", "主导项目", "参与项目", "项目描述", "主要职责")


def _is_project_like(label: str) -> bool:
    return label.startswith(_PROJECT_LABEL_PREFIXES)


def _propagate_context(nodes: List[BaseNode]) -> List[BaseNode]:
    """通用上下文传播：将前面 chunk 中出现的结构型标题（Label：Value）
    注入到后续不含该信息的 chunk 中，解决分页/分块导致上下文断裂问题。

    典型场景：
      - "工作单位：Shopee深圳" 在 p.1，但 p.2 的项目内容 chunk 不含 Shopee
        → 注入 "[上下文：工作单位：Shopee深圳]"，保证检索 Shopee 项目可命中
      - "项目2：Flink平台智能排障Agent" 标题与内容被切开
        → 内容 chunk 继承项目名，使"项目2排障Agent"等查询能召回对应内容
      - 任何文档的 Label:Value 结构（客户、部门、产品名等）均可受益

    机制：
      1. 按 chunk 顺序维护 active_headers（Label→Value）
      2. 同 Label 新值覆盖旧值（如跳槽后工作单位变更）
      3. 遇到边界 Label（工作单位/公司等）且值变化时，清理项目类标签残留
      4. 当前 chunk 未提及的 active_headers 条目，注入 "[上下文：...]" 前缀
    """
    active_headers: Dict[str, str] = {}
    for node in nodes:
        text = node.get_content() or ""
        current = _extract_headers(text)

        # 检测是否跨越了"换工作单位/公司/客户/部门"的边界
        boundary_changed = False
        for b_label in _BOUNDARY_LABELS:
            if b_label in current and active_headers.get(b_label) != current[b_label]:
                boundary_changed = True
                break

        if boundary_changed:
            # 清理上一段落的项目/职责类标签，避免跨公司残留
            active_headers = {
                k: v for k, v in active_headers.items()
                if not _is_project_like(k)
            }

        # 更新active：同 Label 新值覆盖旧值
        if current:
            active_headers.update(current)

        # 找出 active_headers 中当前 chunk 未提及的条目，作为上下文注入
        missing: List[str] = []
        for label, value in active_headers.items():
            if label not in text and value not in text:
                missing.append(f"{label}：{value}")

        if missing:
            prefix = "[上下文：" + " | ".join(missing) + "]\n"
            node.set_content(prefix + text)

    return nodes


def chunk_documents(
    documents: List[Document],
    chunk_size: int = _CHUNK_SIZE,
    chunk_overlap: int = _CHUNK_OVERLAP,
) -> List[BaseNode]:
    md_parser = MarkdownNodeParser()
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 按文件分组 + 按页码排序，确保传播仅在同一文件内、并按原文顺序推进
    docs_by_file = defaultdict(list)
    for doc in documents:
        fn = doc.metadata.get("file_name", "unknown")
        docs_by_file[fn].append(doc)

    all_nodes: List[BaseNode] = []
    for docs in docs_by_file.values():
        docs.sort(key=lambda d: d.metadata.get("page_number") or 0)

        file_nodes: List[BaseNode] = []
        for doc in docs:
            file_nodes.extend(_split_document(doc, md_parser, splitter))

        # 跨 chunk 传播结构型标题上下文（解决分页断裂）
        file_nodes = _propagate_context(file_nodes)
        all_nodes.extend(file_nodes)

    return all_nodes
