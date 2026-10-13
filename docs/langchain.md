# LangChain 会话 Provider

在 `.env` 中设置 `CHAT_PROVIDER=langchain`，重启 FastAPI 后，普通聊天和 SSE
聊天接口会使用 `LangChainProvider`。默认值为 `openai`。

两种 Provider 共用 `ConversationService`、上下文构建器以及现有数据库表。
LangChain 不配置 checkpointer；历史由数据库恢复，当前问题和历史通过
`generate(messages, ...)` 传入，避免重复恢复上下文。

## 事件与数据库

| LangChain 数据 | 项目事件 | 保存方式 |
|---|---|---|
| `messages` 中的文本 chunk | `AssistantTextDeltaEvent` | SSE 推送，不逐段落库 |
| `updates` 中完整 AIMessage 的正文 | `AssistantMessageEvent` | 助手消息 |
| AIMessage.tool_calls | `FunctionCallEvent` | 工具调用，保留 call_id 和参数 |
| ToolMessage | `FunctionCallOutputEvent` | 结果、成功状态和完整来源快照 |
| Agent 正常结束 | `ModelCompletedEvent` | 汇总多次模型调用的用量并完成轮次 |

工具返回 `content_and_artifact`：模型只接收正文，来源通过 artifact 传递给
应用。工具失败返回通用错误，不把底层异常直接发给模型或浏览器。

文本增量使用 `stream_key` 区分不同模型节点执行；它不冒充供应商响应 ID。
真实 response ID 可用时仍保存到原有字段，缺失的 item ID 保持为空。

## 预算与生命周期

每次生成创建独立 middleware，限制工具调用次数，估算每次模型请求的输入，
并为同批工具结果分配预算。检索按预算裁剪，来源对应实际返回的片段。
共享同一个请求数据库 session 的工具通过锁串行执行。

历史摘要复用现有 `OpenAIProvider` 的独立、无工具 Responses 请求。
每个请求的 LangChain 模型使用独立的同步和异步 HTTP 客户端，退出时关闭，
避免关闭 LangChain 缓存的默认客户端后导致后续或并发请求报连接错误。
正常退出或取消会关闭 LangGraph 流；会话事务与失败收尾仍由
`ConversationService` 负责。`completed` 仍在数据库提交之后发送。

Token 估算不是模型专用 tokenizer 的精确值；summary 和失败调用用量与原
Provider 一样，尚未全部计入 Turn 用量。数据库 reasoning 枚举暂未接入事件保存。
