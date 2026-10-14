// Capture the real frontend with isolated, synthetic API responses.
// Requires Playwright (and Chromium), and a running Vite dev server.
// Usage: node scripts/capture_frontend.mjs [http://127.0.0.1:5173]
import { createRequire } from 'node:module'
import { mkdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const { chromium } = createRequire(import.meta.url)('playwright')
const destination = fileURLToPath(new URL('../docs/screenshots/', import.meta.url))
await mkdir(destination, { recursive: true })
const browser = await chromium.launch({
  headless: true,
  ...(process.env.SCREENSHOT_CHROME ? { executablePath: process.env.SCREENSHOT_CHROME } : {}),
})
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 })
  const page = await context.newPage()
  const timestamp = '2026-10-14T08:00:00Z'
  const sessions = [
    { id: 'demo-rag', title: '理解 RAG：让回答有据可查' },
    { id: 'demo-sources', title: '检索结果从哪里来？' },
    { id: 'demo-code', title: 'Python 异步编程速查' },
  ].map((session) => ({ ...session, created_at: timestamp, updated_at: timestamp }))
  const sources = [
    { chunk_id: 'demo-chunk-1', document_id: 'demo-document-1', document_name: 'RAG 学习笔记.md', score: 0.91, headings: ['检索增强生成', '工作流程'], start_line: 12, end_line: 24,
      content: 'RAG 将检索与生成结合起来。系统先把问题转成向量，从知识库中找到相关文档片段，再将这些片段作为上下文交给模型。\n\n文档名、章节与行号让检索结果能够回到原始资料，方便阅读和核对。' },
    { chunk_id: 'demo-chunk-2', document_id: 'demo-document-2', document_name: '知识库实践指南.md', score: 0.86, headings: ['文档切块'], start_line: 36, end_line: 48,
      content: '切块时保留标题层级和原始行号，避免把一个完整概念拆散。合理的切块大小与检索数量需要结合语料和问题进行评测。' },
  ]
  function turn(id, question, answer, retrieved = sources) {
    return { id, sequence: 0, status: 'completed', created_at: timestamp, completed_at: timestamp, error_message: null,
      items: [
        { type: 'message', id: `${id}-user`, sequence: 0, role: 'user', text: question },
        { type: 'function_call', id: `${id}-call`, sequence: 1, call_id: `${id}-search`, tool_name: 'search_knowledge', arguments: { query: question, limit: 5 } },
        { type: 'function_call_output', id: `${id}-output`, sequence: 2, call_id: `${id}-search`, tool_name: 'search_knowledge', output: `找到 ${retrieved.length} 条相关资料。`, succeeded: true, sources: retrieved },
        { type: 'message', id: `${id}-answer`, sequence: 3, role: 'assistant', text: answer },
      ] }
  }
  const histories = {
    'demo-rag': [turn('rag-turn', '结合知识库，解释一下 RAG 的工作原理和适用场景。',
      '## 让模型先查资料，再回答\n\n**RAG（检索增强生成）** 会在回答问题前，从你的知识库中找出相关内容，让模型基于这些资料组织答案。\n\n### 一次问答，三个步骤\n\n1. **理解问题**：将提问转换成向量，表达它的语义。\n2. **检索资料**：在知识库中寻找相关片段，保留文档名、章节和行号。\n3. **生成回答**：把问题与检索结果一起交给模型，返回回答和参考来源。\n\n### 什么时候适合使用？\n\n| 场景 | 知识库里的资料 |\n| :--- | :--- |\n| 学习与复习 | 课程笔记、概念解释、代码示例 |\n| 团队知识问答 | 项目文档、技术规范、常见问题 |\n| 面试准备 | 知识点整理、专题笔记 |\n\n> 检索结果提供可核对的资料，但仍需要检查回答是否准确。')],
    'demo-sources': [turn('sources-turn', '如何核对回答用到的知识库资料？',
      '## 从回答回到原始资料\n\n展开下方的 **参考来源**，可以查看本轮检索到的文档、章节、原始行号和片段内容。\n\n这些信息帮助你核对上下文，也方便继续阅读原文。来源列表表示本轮检索结果，不能等同于回答逐条引用了所有片段。')],
    'demo-code': [turn('code-turn', '用一个简短的 Python 示例说明如何并发执行异步任务。',
      '## 使用 asyncio.gather 并发等待\n\n对互不依赖的 I/O 任务，可以使用 `asyncio.gather()` 一起等待结果：\n\n```python\nimport asyncio\nasync def fetch_note(name: str) -> str:\n    await asyncio.sleep(0.1)  # 模拟异步 I/O\n    return f"已读取：{name}"\n\nasync def main() -> None:\n    notes = await asyncio.gather(\n        fetch_note("检索笔记"),\n        fetch_note("异步编程笔记"),\n    )\n    for note in notes:\n        print(note)\nasyncio.run(main())\n```\n\n**要点**：`await` 让出执行权；`gather()` 的结果顺序与传入任务的顺序一致。CPU 密集计算需要另外选择合适的执行方式。',
      [{ ...sources[0], chunk_id: 'demo-python', document_name: 'Python 异步编程.md', headings: ['asyncio', '并发任务'], start_line: 18, end_line: 42, content: 'asyncio.gather 并发运行可等待对象，并按输入顺序收集结果。' }])],
  }
  await context.addInitScript(() => sessionStorage.setItem('zhiye.token', 'readme-demo-token'))
  await page.route('**/api/v1/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    const session = sessions.find((item) => path.includes(item.id))
    const data = path.endsWith('/turns') ? { items: histories[session.id], next_cursor: null }
      : session || { items: sessions, next_cursor: null }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(data) })
  })
  await page.goto(process.argv[2] || 'http://127.0.0.1:5173')
  await page.getByRole('button', { name: sessions[0].title, exact: true }).waitFor()
  await page.evaluate(() => document.fonts.ready)
  await page.screenshot({ path: `${destination}/welcome.png` })
  for (const [index, filename] of ['conversation', 'sources', 'code'].entries()) {
    await page.getByRole('button', { name: sessions[index].title, exact: true }).click()
    await page.locator('article').waitFor()
    if (filename === 'sources') {
      await page.locator('summary').filter({ hasText: '参考来源' }).click()
      await page.locator('summary').filter({ hasText: 'RAG 学习笔记.md' }).click()
      await page.locator('summary').filter({ hasText: '知识库实践指南.md' }).click()
    }
    await page.screenshot({ path: `${destination}/${filename}.png` })
  }
  console.log(`Saved four frontend screenshots to ${destination}`)
} finally {
  await browser.close()
}
