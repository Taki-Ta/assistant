# 检索评测集

本目录保存用于检索质量评测的固定语料和人工标注问题。

## 目录结构

```text
retrieval/
├── corpus/          固定版本的 Markdown 语料
├── questions.jsonl 每行一个评测问题
├── README.md        数据格式与使用说明
└── SOURCE.md        语料来源、版本与许可证
```

## 问题格式

`questions.jsonl` 中每行都是一个独立 JSON 对象：

```json
{
  "id": "deployment-001",
  "query": "部署 Web API 的最终目标是什么？",
  "relevant_sources": [
    {
      "document": "fastapi-deployment-concepts.md",
      "heading": "部署概念",
      "contains": "避免中断"
    }
  ],
  "tags": ["overview", "paraphrase"]
}
```

- `id`：稳定且唯一的用例标识。
- `query`：发送给检索服务的自然语言问题。
- `relevant_sources`：人工确认的相关来源，可以有多个。
- `document`：`corpus` 目录中的文件名。
- `heading`：相关内容所在的 Markdown 标题，不包含标题锚点。
- `contains`：必须出现在该标题范围内的稳定原文片段。
- `tags`：用于按主题或问题类型拆分统计。

使用 `document + heading + contains` 标注相关性，不直接绑定 Chunk ID。这样调整切块长度或切块算法后，评测标注仍然有效。

当前数据集用于建立第一版中文检索基线。后续应增加多个主题相近的文档，避免单文档评测高估真实检索效果。

## 运行评测

确保 PostgreSQL、pgvector 和 Embedding 配置可用，然后执行：

```powershell
uv run interview_ai eval-retrieval
```

指定数据集、Top-K 和报告路径：

```powershell
uv run interview_ai eval-retrieval `
  --dataset evals/retrieval `
  --limit 5 `
  --report evals/retrieval/reports/baseline.json
```

命令会删除并重建专用用户 `__retrieval_evaluation__` 名下的评测文档，不会删除普通用户数据。它会调用真实 Embedding 服务，因此会产生网络请求和相应费用。

报告包含 Hit@1、Hit@K、MRR@K、Recall@K、平均延迟、P95 延迟，以及每个问题的完整 Top-K 结果。
