<div align="center">
  <h1>知引 · Zhiyin</h1>
  <p><strong>从一个问题开始，让知识触手可及。</strong></p>
  <p>Markdown 知识库助手 · 流式问答 · 可追溯来源 · 持久化会话</p>
  <p>
    <a href="#界面预览">界面预览</a> ·
    <a href="#快速开始">快速开始</a> ·
    <a href="#文档导航">文档</a> ·
    <a href="#english">English</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12" />
    <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/React-19-149ECA?logo=react&logoColor=white" alt="React 19" />
    <img src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL with pgvector" />
  </p>
</div>

![知引会话界面：知识库检索、Markdown 回答与参考来源](docs/screenshots/conversation.png)

## 你的知识，你的对话

将 Markdown 资料建立索引，在「知引」中提问、追问，再展开来源核对原文。
React 前端连接 FastAPI 后端，使用兼容 OpenAI 的 Embedding 与 Responses API
完成检索增强问答。

| 提问与阅读 | 检索与核对 | 会话与知识库 |
| :--- | :--- | :--- |
| SSE 流式回答，支持停止生成 | 展示知识库工具调用及结果 | 新建、切换、删除和恢复会话 |
| Markdown、表格、代码块与复制 | 文档名、章节、行号和原文片段 | JWT 用户隔离与 PostgreSQL 持久化 |
| 多轮追问，延续当前主题 | pgvector 向量检索与检索质量评测 | 按预算构建上下文并压缩较早历史 |

## 界面预览

截图由仓库中的真实前端渲染，使用隔离的演示数据，不包含真实账号、凭证或私人文档。
来源列表展示本轮检索到的资料，不代表回答逐条引用了所有片段。

<table>
  <tr>
    <td width="50%"><img src="docs/screenshots/welcome.png" alt="新建会话：问题建议与最近会话列表" width="100%" /></td>
    <td width="50%"><img src="docs/screenshots/sources.png" alt="来源核对：展开文档章节、行号与原文片段" width="100%" /></td>
  </tr>
  <tr>
    <td align="center"><strong>从一个问题开始</strong><br />快捷建议与最近会话</td>
    <td align="center"><strong>让回答有据可查</strong><br />展开来源，核对原文</td>
  </tr>
</table>

<details>
<summary><strong>查看代码问答界面</strong> · Python 示例与代码复制</summary>

![代码问答：Python 异步编程示例与复制按钮](docs/screenshots/code.png)

</details>

## 快速开始

需要 Python 3.12、[uv](https://docs.astral.sh/uv/)、Node.js 22.12+、
带 pgvector 扩展的 PostgreSQL，以及支持函数调用和流式响应的 OpenAI 兼容服务。

### 1. 配置并启动后端

在仓库根目录执行：

```powershell
uv sync --dev
Copy-Item .env.example .env
```

编辑 `.env`，配置 Embedding、Chat 服务、`DATABASE_URL` 和 `JWT_SECRET`。
Embedding 维度必须匹配数据库结构，当前为 **1536**。
完整配置及已有数据库的迁移步骤见 [后端指南](docs/backend-guide.md#中文)。
macOS / Linux 使用 `cp .env.example .env`。

初始化新数据库并启动 API：

```powershell
psql -d "postgresql://postgres:postgres@localhost:5432/interview_ai" `
  -f src/interview_ai/migration/init_sql
uv run fastapi dev src/interview_ai/api/app.py
```

数据库需提前创建；连接地址应与 `.env` 一致。已有数据库按指南应用尚未执行的迁移。
开启 `IS_DEBUG=true` 后，可在 `http://127.0.0.1:8000/docs` 查看接口文档。

### 2. 启动知引前端

另开一个终端：

```powershell
cd frontend
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173`。前端默认将 `/api` 代理到 `http://127.0.0.1:8000`；
其他后端地址通过 `frontend/.env.local` 中的 `API_PROXY_TARGET` 配置。

### 3. 连接知识库，开始会话

在根目录生成本地开发 Token：

```powershell
uv run python -c "from interview_ai.util import jwt_encode; print(jwt_encode({'sub': 'demo-user'}))"
```

在页面「我的工作空间」设置中填写 Token。凭证保存在 `sessionStorage`，
关闭浏览器会话后需重新配置；Token 到期后需要重新生成。
使用文档 API 上传 UTF-8 `.md` 文件并建立索引，即可在前端提问。
上传与索引命令见 [API 使用流程](docs/backend-guide.md#api-概览)。

> 前端目前提供会话与问答界面，文档上传和索引通过 API 完成。
> 停止生成后的部分回答仅保留在当前页面，刷新后可能消失；不支持断线续传。

## 文档导航

| 想了解什么 | 从这里开始 |
| :--- | :--- |
| 后端配置、认证、API、SSE 与 CLI | [后端指南（中文 / English）](docs/backend-guide.md) |
| 前端运行、交互、测试与部署 | [知引前端说明](frontend/README.md) |
| 会话接口、字段与分页 | [会话接口说明](docs/会话接口.md) |
| 上下文预算、压缩与事务设计 | [技术栈与依赖使用说明](docs/技术栈与依赖使用说明.md) |
| 知识库结构与用户隔离 | [知识库结构与权限](docs/知识库结构与权限.md) |
| 检索数据集与评测 | [Python 教程评测集](evals/python_tutorial/README.md) · [检索评测](evals/retrieval/README.md) |

<details>
<summary><strong>开发校验与项目结构</strong></summary>

后端：

```powershell
uv run pytest -q -m "not network"
uv run ruff check src tests
uv run ruff format --check src tests
```

前端（在 `frontend/` 中执行）：

```powershell
npm run test
npm run build
npm run format:check
```

```text
frontend/           知引 · React / TypeScript / Vite 前端
src/interview_ai/   FastAPI、会话服务、检索、摄取与数据库迁移
evals/              版本化检索语料与评测问题
scripts/            数据集、评测与 README 截图脚本
tests/              后端单元、API、数据库与集成测试
docs/               技术文档与界面截图
```

截图复现方式见 [截图说明](docs/screenshots/README.md)。

</details>

## English

**Zhiyin (知引)** is a multi-user RAG assistant
for Markdown knowledge bases. It combines React, FastAPI, OpenAI-compatible
Embedding and Responses APIs, PostgreSQL, and pgvector.

- Chat with your knowledge base, receive streamed answers, and continue persistent conversations.
- Read Markdown, tables, and code; inspect tool activity and retrieved source passages.
- Trace results to document names, headings, and line numbers.
- Isolate user data with JWT authentication; manage context budgets and compact older history.
- Upload and index documents through the API, and evaluate retrieval through the CLI.

The screenshots above show the actual frontend with synthetic demo responses.
Retrieved sources describe what was found during a turn; they do not guarantee that
every passage was cited in the answer. Streaming cannot be resumed after disconnection.

Configure and launch the backend, then run `npm ci` and `npm run dev` in `frontend/`.
Open `http://127.0.0.1:5173`, enter a development Bearer Token in workspace settings,
and upload/index your Markdown files through the API.
Prerequisites: Python 3.12, uv, Node.js 22.12+, PostgreSQL with pgvector, and
OpenAI-compatible embedding and streaming Responses services.

**[Full backend setup & API guide](docs/backend-guide.md#english)** ·
**[Frontend setup & deployment](frontend/README.md)** ·
**[Screenshots](#界面预览)**
