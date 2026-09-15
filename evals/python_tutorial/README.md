# Python 教程检索评测集

该数据集用于检验多文档、相邻主题和长文档场景下的中文 RAG 检索质量。语料由
learn-py.org 对应的开源教程仓库按章整理而成，共 29 个 Markdown 文档。

## 目录结构

```text
python_tutorial/
├── corpus/                按章合并后的检索语料
├── questions.jsonl        人工复核后用于基线评测的问题
├── source_manifest.json   输出文档、原始路径及 SHA-256
├── SOURCE.md              来源、版本、修改和授权说明
└── LICENSE.learn-py       上游 CC BY 4.0 许可证
```

`questions.jsonl` 沿用 `evals/retrieval` 的格式。相关性标签使用
`document + heading + contains`，不绑定可能因切块参数变化而失效的 Chunk ID。

## 重新构建语料

先检出清单指定的上游提交：

```powershell
git clone https://github.com/walter201230/Python.git ..\Python-tutorial-source
git -C ..\Python-tutorial-source checkout abfd0ab613b2f3a4b5003a69acda53ea67b69b03
```

然后在本项目根目录运行：

```powershell
uv run python scripts/build_learn_py_corpus.py `
  --source ..\Python-tutorial-source `
  --output evals/python_tutorial
```

构建器会拒绝版本不匹配的源码，避免上游更新导致基线悄悄变化。它只覆盖清单中
明确管理的输出文件，不会清空输出目录。

## 运行评测

```powershell
uv run interview_ai eval-retrieval `
  --dataset evals/python_tutorial `
  --limit 5 `
  --report evals/python_tutorial/reports/baseline.json
```

该命令需要可用的 PostgreSQL、pgvector 和 Embedding 配置，并会产生真实的
Embedding 网络请求及费用。
