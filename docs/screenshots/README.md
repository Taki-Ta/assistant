# README 界面截图

这些 PNG 来自当前 `frontend/` 的真实页面。Playwright 拦截 `/api/v1/**`，
返回固定的演示会话，不连接后端、不调用模型服务、不读取真实用户资料。
`readme-demo-token` 只是模拟请求中的占位字符串。

| 文件 | 展示内容 |
| :--- | :--- |
| `conversation.png` | 知识库问答、Markdown 表格与工具状态 |
| `welcome.png` | 新建会话首页与最近会话 |
| `sources.png` | 展开的来源、章节、行号与原文片段 |
| `code.png` | Python 代码回答与复制按钮 |

## 重新生成

先在 `frontend/` 中运行 `npm ci` 和 `npm run dev`。截图脚本需要 Node.js
能够解析 `playwright` 包，以及 Playwright 的 Chromium。
可在独立工具目录安装 Playwright，并通过 `NODE_PATH` 指向该目录的 `node_modules`，
避免修改项目依赖。已安装的 Chrome 也可通过 `SCREENSHOT_CHROME` 环境变量指定。

在仓库根目录执行：

```powershell
node scripts/capture_frontend.mjs
# 如果 Vite 使用其他端口：
node scripts/capture_frontend.mjs http://127.0.0.1:5174
```

截图尺寸为 1440 × 1000，设备缩放为 1。脚本使用隔离的浏览器上下文，
完成后关闭浏览器，并覆盖本目录中的四张 PNG。
