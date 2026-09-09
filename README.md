# Interview AI

扫描 Markdown 知识库，并预览下一次索引需要执行的变更。

## 预览索引计划

首次扫描时，所有文档都会显示为新增：

```powershell
uv run interview_ai plan <知识库目录>
```

已有上一次成功索引生成的 Manifest 时，可以对比新增、修改、删除和未变化的文档：

```powershell
uv run interview_ai plan <知识库目录> --manifest <manifest.json>
```

常用选项：

- `--json`：输出适合程序处理的 JSON。
- `--show-unchanged`：文本模式下列出未变化文件。
- `--max-chunk-length <长度>`：指定分段的最大字符数；与旧 Manifest 不一致时，共有文档会被标记为需要重建。

`plan` 是只读命令，不会保存 Manifest，也不会修改向量索引。
