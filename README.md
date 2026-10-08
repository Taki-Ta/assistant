# Interview AI

[English](#english) | [中文](#中文)

## English

Interview AI is a multi-user RAG backend for Markdown knowledge bases. It uses
FastAPI, OpenAI-compatible Embedding and Responses APIs, PostgreSQL, and
pgvector to provide document ingestion, vector search, tool-using chat,
retrieval evaluation, and persistent multi-turn conversations.

### Features

- Upload UTF-8 Markdown documents and split them into traceable chunks.
- Generate embeddings in batches and store vectors in PostgreSQL/pgvector.
- Filter documents and search results by the authenticated user.
- Run a Responses API tool loop backed by the local knowledge search tool.
- Persist sessions, turns, messages, tool events, and history summaries.
- Build budgeted context and summarize older history while retaining recent turns.
- Restore the latest summary, uncovered completed messages, and failed user requests.
- Stream answer text, tool calls, and tool results over SSE with idle heartbeats.
- Return the union of all sources retrieved during a turn, deduplicated by
  chunk ID and ordered by score.
- Evaluate retrieval quality with Hit@K, MRR, Recall, and latency metrics.
- Preview Markdown indexing changes from the CLI without modifying the index.

### Architecture

```text
FastAPI
├── Document API ── IndexService ── Embedding API
│                                └── PostgreSQL / pgvector
├── Search API ──── SearchService ── Embedding API + pgvector
└── Chat API ────── ConversationService
                    ├── ContextBuilder ── token estimation / history compaction
                    ├── ConversationRepository ── conversation database session
                    └── OpenAIProvider
                        └── ToolRegistry ── SearchService ── search database session
```

Conversation writes and tool searches use different request-scoped
`AsyncSession` instances. Model and tool network calls run outside database
transactions, while turn creation and completion use short transactions.

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL with the pgvector extension
- An OpenAI-compatible Embedding API
- An OpenAI-compatible Responses API with function calling and streaming support

### Quick start

1. Install dependencies:

```powershell
uv sync --dev
```

2. Create the environment file:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux, use `cp .env.example .env`.

3. Configure `.env`. A minimal example is:

```dotenv
OPENAI_API_KEY=your-embedding-api-key
OPENAI_HOST=https://your-embedding-endpoint/v1
EMBEDDING_MODEL_NAME=your-embedding-model
DIMENSIONS=1536
EMBEDDING_BATCH_SIZE=20

CHAT_API_KEY=your-chat-api-key
CHAT_API_HOST=https://your-chat-endpoint/v1
CHAT_MODEL=your-chat-model
CHAT_CONTEXT_WINDOW=32768
CHAT_CONTEXT_BUDGET=32768
MAX_TOOL_CALLS=8

DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/interview_ai
JWT_SECRET=replace-with-a-long-random-secret
JWT_EXP_SECONDS=300

SEARCH_TOP_K=5
SEARCH_SCORE_THRESHOLD=0.5
IS_DEBUG=true
```

The embedding dimension must match the `VECTOR` dimension in the database
schema. The current schema uses 1536 dimensions.

Set `CHAT_CONTEXT_WINDOW` to the model's supported context length in tokens, and
`CHAT_CONTEXT_BUDGET` to the total per-request budget, at most that length. Both
currently default to 32768; reduce both if your model has a smaller window.
Output, safety margin, and tool reserve are calculated in code. At the default
budget they occupy 8192, 4096, and 4096 tokens, leaving 16384 for initial input.
Token counts are estimates, not model-specific tokenizer measurements.

4. Initialize a new database:

```powershell
psql -d "postgresql://postgres:postgres@localhost:5432/interview_ai" `
  -f src/interview_ai/migration/init_sql
```

For an existing database, apply the numbered migration scripts that have not
yet been executed, including
`src/interview_ai/migration/002_create_agent_conversations.sql`.

5. Start the API:

```powershell
uv run fastapi dev src/interview_ai/api/app.py
```

When `IS_DEBUG=true`, Swagger UI is available at
`http://127.0.0.1:8000/docs`.

### Authentication

Protected endpoints require an HS256 bearer token containing a non-empty `sub`
claim. For local development, generate one with the configured `JWT_SECRET`:

```powershell
uv run python -c "from interview_ai.util import jwt_encode; print(jwt_encode({'sub': 'demo-user'}))"
```

Use it as:

```text
Authorization: Bearer <token>
```

The `sub` value is the owner ID used to isolate documents, searches, and chat
sessions.

### API overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/documents` | List the current user's documents |
| `POST` | `/api/v1/documents` | Upload a UTF-8 `.md` file, up to 5 MiB |
| `POST` | `/api/v1/documents/{document_id}/index` | Chunk and index a document |
| `POST` | `/api/v1/search` | Search indexed knowledge |
| `POST` | `/api/v1/chat` | Create or continue a RAG conversation |
| `POST` | `/api/v1/chat/stream` | Stream a RAG conversation as SSE |

Typical RAG workflow:

```bash
# Upload a document
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@knowledge.md"

# Index it with the document ID returned above
curl -X POST http://127.0.0.1:8000/api/v1/documents/$DOCUMENT_ID/index \
  -H "Authorization: Bearer $TOKEN"

# Search directly
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?","limit":5}'

# Start a conversation
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain RAG using my knowledge base."}'
```

The chat response contains a new `session_id`. Send it with the next message to
continue the conversation:

```json
{
  "message": "How does chunking affect it?",
  "session_id": "01900000-0000-7000-8000-000000000010"
}
```

Example response:

```json
{
  "answer": "...",
  "retrieved_sources": [
    {
      "chunk_id": "01900000-0000-7000-8000-000000000001",
      "content": "...",
      "score": 0.82,
      "document_id": "01900000-0000-7000-8000-000000000002",
      "document_name": "knowledge.md",
      "headings": ["RAG", "Chunking"],
      "start_line": 10,
      "end_line": 24
    }
  ],
  "session_id": "01900000-0000-7000-8000-000000000010"
}
```

`retrieved_sources` is the deduplicated union of successful retrievals made in
the current turn. It describes what the model could use; it does not claim that
every returned chunk was explicitly cited in the final answer.

### Streaming chat

Use the same request body and authorization header as the JSON chat endpoint.
Add `session_id` when continuing an existing conversation:

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message":"Explain RAG using my knowledge base."}'
```

| SSE event | Main fields and meaning |
|---|---|
| `started` | `session_id`, `turn_id`: the turn has been created |
| `text_delta` | `delta`, `provider_item_id`, `provider_response_id`: incremental text |
| `function_call` | `call_id`, `tool_name`, `arguments`: a tool invocation |
| `function_call_output` | `call_id`, `tool_name`, `output`, `succeeded`, `sources`: tool result |
| `completed` | `session_id`, `turn_id`, `answer`, `retrieved_sources`: committed final result |
| `error` | `code`, `message`, `session_id`, `turn_id`: stream failure |

Tool events and text can repeat across model rounds. Pair tool events by `call_id`
and group text by provider IDs. Use `completed.answer` as the final answer;
`completed` is sent after the database commit. Individual text deltas are not
persisted. An idle stream sends a comment heartbeat (`: ping`) every 15 seconds.

Browser clients should POST with `fetch()`, decode UTF-8 incrementally, and buffer
until a blank line completes an SSE frame. Network chunks are not event boundaries;
ignore comment heartbeats. Abort the fetch with `AbortController.abort()` to stop
generation. Disconnections close the upstream stream and mark unfinished turns
failed; this does not guarantee that the remote provider stops computation or billing.
There is no separate cancel API or resumable stream.

Errors before streaming starts use normal HTTP responses. Errors after `started`
use an SSE `error` event while HTTP status remains 200. A stream ending without
`completed` must not be treated as a successful turn.

### CLI

Scan Markdown files:

```powershell
uv run interview_ai scan <knowledge-directory>
```

Preview added, modified, deleted, and unchanged documents without writing a
manifest or changing the vector index:

```powershell
uv run interview_ai plan <knowledge-directory>
uv run interview_ai plan <knowledge-directory> --manifest <manifest.json>
uv run interview_ai plan <knowledge-directory> --json --show-unchanged
```

Run the retrieval benchmark. This command rebuilds an isolated evaluation index
and makes real embedding requests, which may incur cost:

```powershell
uv run interview_ai eval-retrieval `
  --dataset evals/python_tutorial `
  --limit 5 `
  --report evals/python_tutorial/reports/baseline.json
```

### Tests and quality checks

```powershell
uv run pytest -q -m "not network"
uv run ruff check src tests
uv run ruff format --check src tests
```

Tests marked `network` require configured external services.

### Current limitations

- Streaming requires provider and proxy support; there is no resumable stream or separate cancel API.
- Conversation recovery uses the latest summary, uncovered completed messages,
  and failed user requests. It does not replay old tool traces or other sessions.
- Context uses approximate token counts and a configurable total budget. Older
  history is summarized in bounded batches; recent turns and the current question
  are retained when space permits. Checkpoints persist as compaction items, and
  subsequent requests restore the latest checkpoint plus uncovered history.
- Initial context reserves space for tool interactions. Knowledge search results
  are limited to the remaining request capacity, keeping complete relevant chunks
  when possible and truncating a single oversized chunk when necessary. Sources
  match the content sent to the model; active tool traces are not compressed.
  Requests still fail if even a minimal useful result cannot fit.
  Summary calls and failed model requests are not fully included in Turn usage
  totals. Reasoning items are not currently persisted.
- Migration execution is not tracked by Alembic; deployments must track applied
  SQL scripts.

### Documentation

See [Technology stack and dependency usage](docs/技术栈与依赖使用说明.md) (Chinese)
for context budgets, compaction, tool result limits, SSE, cancellation, database
transactions, and code navigation.

### Project layout

```text
src/interview_ai/
├── agent/          Providers, tools, events, conversation service/repository
├── api/            FastAPI app, routes, authentication, dependencies, schemas
├── db/             SQLModel models and PostgreSQL repositories
├── evaluation/     Retrieval dataset loading, metrics, and benchmark runner
├── indexing/       Embedding, indexing, search, and vector stores
├── ingestion/      Markdown scanning, chunking, and manifests
└── migration/      Database initialization and incremental SQL migrations

evals/              Versioned retrieval corpora and questions
scripts/            Dataset and benchmark helper scripts
tests/              Unit, API, database, and integration tests
docs/               Architecture and dependency documentation
```

---

## 中文

Interview AI 是一个面向 Markdown 知识库的多用户 RAG 后端。项目使用
FastAPI、兼容 OpenAI 的 Embedding 与 Responses API、PostgreSQL 和 pgvector，
提供文档摄取、向量检索、工具调用问答、检索评测以及持久化多轮会话。

### 功能特性

- 上传 UTF-8 Markdown 文档，并切分为可追溯的 Chunk。
- 分批生成 Embedding，并将向量保存到 PostgreSQL/pgvector。
- 根据 JWT 用户身份隔离文档、检索结果和会话。
- 通过本地知识检索工具执行 Responses API 工具调用循环。
- 持久化 Session、Turn、消息、工具事件和历史摘要。
- 按预算构建上下文，压缩较早历史并尽量保留最近完整轮次。
- 继续会话时恢复最新摘要、未覆盖的已完成消息和失败轮次的用户请求。
- 通过 SSE 流式返回回答、工具调用与工具结果，空闲时发送心跳。
- 返回一个 Turn 内全部检索结果的并集，按 Chunk ID 去重并按分数排序。
- 使用 Hit@K、MRR、Recall 和延迟指标评估检索质量。
- 通过 CLI 只读预览 Markdown 索引变更。

### 架构

```text
FastAPI
├── 文档 API ── IndexService ── Embedding API
│                              └── PostgreSQL / pgvector
├── 检索 API ── SearchService ── Embedding API + pgvector
└── 对话 API ── ConversationService
                ├── ContextBuilder ── Token 估算 / 历史压缩
                ├── ConversationRepository ── 会话数据库 Session
                └── OpenAIProvider
                    └── ToolRegistry ── SearchService ── 检索数据库 Session
```

会话写入和工具检索使用两个不同的请求级 `AsyncSession`。模型与工具的网络调用
位于数据库事务之外；Turn 的创建和完成分别使用短事务。

### 环境要求

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- 安装 pgvector 扩展的 PostgreSQL
- 兼容 OpenAI 的 Embedding API
- 支持函数调用与流式响应的 OpenAI 兼容 Responses API

### 快速开始

1. 安装依赖：

```powershell
uv sync --dev
```

2. 创建环境变量文件：

```powershell
Copy-Item .env.example .env
```

macOS 或 Linux 使用 `cp .env.example .env`。

3. 配置 `.env`：

```dotenv
OPENAI_API_KEY=你的-Embedding-API-Key
OPENAI_HOST=https://你的-Embedding-服务地址/v1
EMBEDDING_MODEL_NAME=你的-Embedding-模型
DIMENSIONS=1536
EMBEDDING_BATCH_SIZE=20

CHAT_API_KEY=你的-Chat-API-Key
CHAT_API_HOST=https://你的-Chat-服务地址/v1
CHAT_MODEL=你的-Chat-模型
CHAT_CONTEXT_WINDOW=32768
CHAT_CONTEXT_BUDGET=32768
MAX_TOOL_CALLS=8

DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/interview_ai
JWT_SECRET=替换为足够长的随机字符串
JWT_EXP_SECONDS=300

SEARCH_TOP_K=5
SEARCH_SCORE_THRESHOLD=0.5
IS_DEBUG=true
```

Embedding 维度必须与数据库的 `VECTOR` 维度一致，当前表结构使用 1536 维。

`CHAT_CONTEXT_WINDOW` 填写模型支持的最大上下文 Token 数，`CHAT_CONTEXT_BUDGET`
是每次模型请求使用的总预算，不能超过前者。两项默认均为 32768，模型窗口较小时
需要一起调小。输出、安全余量和工具预留由代码自动计算；默认分别为 8192、4096、
4096 Token，首次输入可用 16384 Token。当前采用估算，不是模型专用 tokenizer 的精确计数。

4. 初始化新数据库：

```powershell
psql -d "postgresql://postgres:postgres@localhost:5432/interview_ai" `
  -f src/interview_ai/migration/init_sql
```

已有数据库需要按顺序执行尚未应用的增量迁移，包括
`src/interview_ai/migration/002_create_agent_conversations.sql`。

5. 启动 API：

```powershell
uv run fastapi dev src/interview_ai/api/app.py
```

当 `IS_DEBUG=true` 时，可以访问 `http://127.0.0.1:8000/docs` 查看 Swagger UI。

### 身份认证

受保护接口要求 HS256 Bearer Token，并且 Token 必须包含非空的 `sub`。本地开发时
可以使用 `.env` 中的 `JWT_SECRET` 生成 Token：

```powershell
uv run python -c "from interview_ai.util import jwt_encode; print(jwt_encode({'sub': 'demo-user'}))"
```

请求头格式：

```text
Authorization: Bearer <token>
```

`sub` 会作为 `owner_id`，用于隔离不同用户的文档、检索和会话。

### API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/v1/documents` | 获取当前用户的文档列表 |
| `POST` | `/api/v1/documents` | 上传不超过 5 MiB 的 UTF-8 `.md` 文件 |
| `POST` | `/api/v1/documents/{document_id}/index` | 切分并索引文档 |
| `POST` | `/api/v1/search` | 检索已索引的知识库 |
| `POST` | `/api/v1/chat` | 创建或继续 RAG 会话 |
| `POST` | `/api/v1/chat/stream` | 通过 SSE 流式返回 RAG 回答 |

一次完整的 RAG 使用流程：

```bash
# 上传文档
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@knowledge.md"

# 使用上一步返回的文档 ID 建立索引
curl -X POST http://127.0.0.1:8000/api/v1/documents/$DOCUMENT_ID/index \
  -H "Authorization: Bearer $TOKEN"

# 直接检索
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"什么是 RAG？","limit":5}'

# 创建会话
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"根据我的知识库解释 RAG。"}'
```

聊天响应会返回新的 `session_id`。后续请求携带它即可继续会话：

```json
{
  "message": "文档切块会对它产生什么影响？",
  "session_id": "01900000-0000-7000-8000-000000000010"
}
```

响应示例：

```json
{
  "answer": "……",
  "retrieved_sources": [
    {
      "chunk_id": "01900000-0000-7000-8000-000000000001",
      "content": "……",
      "score": 0.82,
      "document_id": "01900000-0000-7000-8000-000000000002",
      "document_name": "knowledge.md",
      "headings": ["RAG", "文档切块"],
      "start_line": 10,
      "end_line": 24
    }
  ],
  "session_id": "01900000-0000-7000-8000-000000000010"
}
```

`retrieved_sources` 是当前 Turn 内所有成功检索结果去重后的并集，表示模型可以使用的
资料，并不代表最终回答明确引用了其中每一个 Chunk。

### 流式回答

请求体和认证方式与普通聊天接口一致，继续会话时携带 `session_id`：

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message":"根据我的知识库解释 RAG。"}'
```

| SSE 事件 | 主要字段与含义 |
|---|---|
| `started` | `session_id`、`turn_id`：轮次已创建 |
| `text_delta` | `delta`、`provider_item_id`、`provider_response_id`：文本增量 |
| `function_call` | `call_id`、`tool_name`、`arguments`：工具调用 |
| `function_call_output` | `call_id`、`tool_name`、`output`、`succeeded`、`sources`：工具结果 |
| `completed` | `session_id`、`turn_id`、`answer`、`retrieved_sources`：已提交的最终结果 |
| `error` | `code`、`message`、`session_id`、`turn_id`：流内错误 |

工具事件和文本可能在多次模型请求中重复出现。通过 `call_id` 配对工具调用与结果，
通过 Provider ID 区分文本消息，最终以 `completed.answer` 为准。完成事件只在数据库
提交后发送，文本增量不单独入库。空闲时每 15 秒发送注释心跳 `: ping`。

浏览器使用 `fetch()` 发起 POST，增量解码 UTF-8，并缓存到空行后再解析完整 SSE 事件。
网络分块不等于事件边界，注释心跳可以忽略。停止生成时调用 `AbortController.abort()`
断开请求；服务端关闭上游流并将未完成轮次标记失败，但不能保证远端模型立即停止计算
或计费。目前没有独立取消接口或断线续传。

流式响应开始前的错误使用普通 HTTP 错误响应；`started` 之后的错误通过 SSE `error`
表达，此时 HTTP 状态仍为 200。连接结束但没有收到 `completed` 时，不能视为成功。

### CLI

扫描 Markdown 文件：

```powershell
uv run interview_ai scan <知识库目录>
```

只读预览新增、修改、删除和未变化的文档，不写入 Manifest，也不修改向量索引：

```powershell
uv run interview_ai plan <知识库目录>
uv run interview_ai plan <知识库目录> --manifest <manifest.json>
uv run interview_ai plan <知识库目录> --json --show-unchanged
```

运行检索基准测试。命令会重建隔离的评测索引，并发起真实 Embedding 请求，可能产生费用：

```powershell
uv run interview_ai eval-retrieval `
  --dataset evals/python_tutorial `
  --limit 5 `
  --report evals/python_tutorial/reports/baseline.json
```

### 测试与质量检查

```powershell
uv run pytest -q -m "not network"
uv run ruff check src tests
uv run ruff format --check src tests
```

标记为 `network` 的测试需要可用的外部服务配置。

### 当前限制

- 流式接口需要模型服务和代理支持，目前没有断线续传或独立取消接口。
- 会话恢复使用最新摘要、未覆盖的已完成消息和失败轮次的用户消息，不重放旧工具
  轨迹，也不自动注入其他会话。
- 上下文采用 Token 估算和可配置总预算，超限时分批压缩较早历史，在空间允许时
  保留最近完整问答和当前问题。摘要作为 Compaction Item 保存，后续请求恢复最新
  摘要及未覆盖的历史。原始消息不会删除。
- 初始上下文为工具交互预留空间，知识检索结果按实际剩余容量限制，优先保留完整
  相关片段，必要时截断单个过长片段。返回来源与模型看到的内容一致；暂不压缩
  正在执行的工具链，连最小有效结果都无法容纳时仍明确失败。摘要调用使用聊天模型，
  其用量及失败模型请求的用量暂未完整计入 Turn；当前不保存 Reasoning Item。
- 项目尚未使用 Alembic 跟踪迁移，部署流程需要自行记录已经执行的 SQL 脚本。

### 技术说明

[技术栈与依赖使用说明](docs/技术栈与依赖使用说明.md) 介绍当前依赖的实际用法、上下文
预算与压缩、工具结果限制、SSE 与取消处理、数据库事务及代码阅读路径。

### 项目结构

```text
src/interview_ai/
├── agent/          Provider、工具、事件、会话 Service 与 Repository
├── api/            FastAPI 应用、路由、认证、依赖和 Schema
├── db/             SQLModel 模型和 PostgreSQL Repository
├── evaluation/     检索数据集加载、指标和基准运行器
├── indexing/       Embedding、索引、检索和向量存储
├── ingestion/      Markdown 扫描、切块和 Manifest
└── migration/      数据库初始化与增量 SQL 迁移

evals/              带版本的检索语料和问题集
scripts/            数据集与基准测试辅助脚本
tests/              单元、API、数据库和集成测试
docs/               架构与依赖说明
```
