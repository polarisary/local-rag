# Enterprise-Local-RAG 第一周部署指南

## 1. 准备工作

### 1.1 环境依赖
- Docker Engine 24+ 与 Docker Compose v2
- 可访问外网（拉镜像、HuggingFace 下载 BGE-M3 模型、调外部 LLM API）
- 至少 4C / 8G RAM 空闲

### 1.2 配置环境变量
```bash
cd enterprise-local-rag
cp .env.example .env
```

编辑 `.env`，至少填入以下两项：

```env
LLM_API_KEY=sk-你的真实KEY
# 如使用非 deepseek，修改 LLM_BASE_URL / LLM_MODEL
```

> 注意：`MILVUS_HOST=milvus` 在 Docker 网络内用服务名解析，**不要修改**。

## 2. 放入测试文档

把你的 PDF / DOCX / PPTX 放进项目根目录的 `test_data/` 文件夹（空文件也能启动，但没有内容问答）。

```
test_data/
  ├── handbook.pdf
  └── policy.docx
```

## 3. 一键启动全部服务

```bash
docker compose up -d --build
```

首次启动会拉 4 个镜像（etcd/minio/milvus/open-webui）并构建 fastapi 镜像，约 3-10 分钟。

确认全部 healthy：

```bash
docker compose ps
```

预期 `milvus`、`fastapi`、`open-webui` 都处于 Up/healthy。

## 4. 触发文档入库

等 fastapi 启动完成后调用入库接口：

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/app/test_data"}'
```

同步执行，返回类似：

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

> 首次 BGE-M3 下载模型可能较慢，容器日志 `docker logs -f rag-fastapi` 可观察进度。

## 5. 前端问答

浏览器打开 **http://localhost:3000** 进入 Open-WebUI：

1. 首次进入注册一个任意邮箱/密码（仅本地用，不发邮件）
2. 左上角 Settings → Connections：确认 OpenAI API Base 是 `http://fastapi:8000/api/v1`、API Key 随便填（例如 `sk-dummy`）
3. 顶部选任意 Model（名字不重要，实际走你 `.env` 里配的 LLM）
4. 开始提问，例如：
   - "请总结 handbook 的核心内容"
   - "policy 中关于 X 的规定是什么"

回答末尾会带 `**引用来源：**` 列表，列出文件名与（如有）页码。

## 6. 常见问题排查

| 现象 | 排查方向 |
|------|----------|
| `rag-fastapi` 启动退出 | `docker logs rag-fastapi` 看报错；多半是 Milvus 未就绪，重启 `docker compose restart fastapi` |
| 入库 chunks=0 | 检查 test_data 下文件后缀是否为 `.pdf/.docx/.pptx`，内容是否为空/扫描件 |
| Open-WebUI 报 500 | 看 fastapi 日志，大概率 LLM API Key 错或模型名不匹配 |
| BGE-M3 模型下载卡住 | 配置 `HF_ENDPOINT=https://hf-mirror.com`（加到 `.env` 或 docker-compose.yml fastapi 的 environment 中）后 `docker compose up -d --build fastapi` |
| 扫描版 PDF 没内容 | 第一周不支持 OCR，只能解析原生可复制文本 PDF |

## 7. 停止与重置

```bash
# 停止
docker compose down

# 停止并清除 Milvus/WebUI 持久化数据（重新入库/重置 UI）
docker compose down -v
```
