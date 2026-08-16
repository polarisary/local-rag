# Enterprise-Local-RAG

企业私有知识库 RAG 最小可运行 Demo：本地文档批量导入 → 解析分块 → Milvus 入库 → RAG 问答 → 溯源输出，对接 Open-WebUI 作为前端。

## 快速启动

### 1. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填入真实的 LLM API Key：

```env
LLM_API_KEY=sk-你的真实KEY
# 如使用非 deepseek，修改 LLM_BASE_URL / LLM_MODEL
```

> `MILVUS_HOST=milvus` 在 Docker 网络内用服务名解析，不要修改。

### 2. 放入测试文档

把 PDF / DOCX / PPTX 放进 `test_data/` 文件夹：

```
test_data/
  ├── handbook.pdf
  └── policy.docx
```

### 3. 一键启动全部服务

```bash
docker compose up -d --build
```

首次启动会拉取 etcd / minio / milvus / open-webui 镜像并构建 fastapi 镜像。

确认全部 healthy：

```bash
docker compose ps
```

### 4. 触发文档入库

等 fastapi 启动完成后调用入库接口：

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/app/test_data"}'
```

返回示例：

```json
{
  "code": 0,
  "msg": "ingest finished",
  "data": {
    "folder_path": "/app/test_data",
    "documents": 12,
    "chunks": 87,
    "elapsed_seconds": 42.31
  }
}
```

> 首次 BGE-M3 下载模型可能较慢，可观察日志：`docker logs -f rag-fastapi`

### 5. 前端问答

浏览器打开 **http://localhost:3000** 进入 Open-WebUI：

1. 首次进入注册一个任意邮箱/密码（仅本地用，不发邮件）
2. 左上角 Settings → Connections：确认 OpenAI API Base 是 `http://fastapi:8000/api/v1`、API Key 随便填（例如 `sk-dummy`）
3. 顶部选任意 Model（名字不影响，实际走 `.env` 里配的 LLM）
4. 开始提问

回答末尾会带 `**引用来源：**` 列表，列出文件名与（如有）页码。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/ingest` | 触发文件夹批量入库，body: `{"folder_path": "/app/test_data"}` |
| POST | `/api/v1/chat/completions` | OpenAI 兼容问答接口，返回 `extra.references` 溯源信息 |

## 常见问题

| 现象 | 排查方向 |
|------|----------|
| `rag-fastapi` 启动退出 | `docker logs rag-fastapi` 看报错；多半是 Milvus 未就绪，`docker compose restart fastapi` |
| 入库 chunks=0 | 检查 test_data 下文件后缀是否为 `.pdf/.docx/.pptx`，内容是否为空/扫描件 |
| Open-WebUI 报 500 | 看 fastapi 日志，大概率 LLM API Key 错或模型名不匹配 |
| BGE-M3 模型下载卡住 | 在 fastapi 的 environment 加 `HF_ENDPOINT=https://hf-mirror.com` 后重新构建 |
| 扫描版 PDF 没内容 | 第一周不支持 OCR，只能解析原生可复制文本 PDF |

## 停止与重置

```bash
# 停止
docker compose down

# 停止并清除持久化数据（重新入库/重置）
docker compose down -v
```

## 项目结构

```
enterprise-local-rag/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── requirements.txt
├── test_data/
├── docs/
│   ├── deploy_week1.md
│   └── td.txt
└── src/
    ├── etl/
    │   ├── document_loader.py   # 文档解析
    │   ├── chunker.py           # 分块
    │   └── pipeline.py           # 入库流程
    ├── rag/
    │   ├── vector_store.py       # Milvus 封装
    │   ├── retriever.py          # 向量检索
    │   └── llm_chain.py           # RAG 主链路
    └── api/
        └── main.py               # FastAPI 入口
```

详细部署说明见 [docs/deploy_week1.md](docs/deploy_week1.md)。
