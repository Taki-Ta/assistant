import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, expect, it, vi } from 'vitest'
import App from './App'
import * as api from './api/chat'
import type { ChatEvent, Session, Turn } from './api/types'
vi.mock('./api/chat', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./api/chat')>()),
  listSessions: vi.fn(),
  getSession: vi.fn(),
  listTurns: vi.fn(),
  deleteSession: vi.fn(),
  streamChat: vi.fn(),
}))
const session: Session = {
  id: 'session-a',
  title: '历史会话',
  created_at: '2026-10-10',
  updated_at: '2026-10-10',
}
const history: Turn = {
  id: 'old',
  sequence: 0,
  status: 'completed',
  created_at: '2026-10-10',
  completed_at: null,
  error_message: null,
  items: [
    { type: 'message', id: 'u', sequence: 0, role: 'user', text: '历史问题' },
    { type: 'message', id: 'a', sequence: 1, role: 'assistant', text: '历史回答' },
  ],
}
beforeEach(() => {
  vi.mocked(api.listSessions)
    .mockReset()
    .mockResolvedValue({ items: [session], next_cursor: null })
  vi.mocked(api.getSession).mockReset().mockResolvedValue(session)
  vi.mocked(api.listTurns)
    .mockReset()
    .mockResolvedValue({ items: [history], next_cursor: null })
  vi.mocked(api.deleteSession).mockReset().mockResolvedValue(undefined)
  vi.mocked(api.streamChat).mockReset()
})
it('configures a token and loads actual server conversations', async () => {
  const user = userEvent.setup()
  render(<App />)
  expect(screen.getByLabelText('输入消息')).toBeDisabled()
  await user.click(screen.getByRole('button', { name: '设置登录凭证，连接知识库' }))
  await user.type(screen.getByLabelText('登录凭证'), 'Bearer test-token')
  await user.click(screen.getByRole('button', { name: '保存' }))
  await screen.findByRole('button', { name: '历史会话' })
  expect(sessionStorage.getItem('zhiye.token')).toBe('test-token')
  await user.click(screen.getByRole('button', { name: '历史会话' }))
  await screen.findByText('历史回答')
  expect(api.getSession).toHaveBeenCalledWith('test-token', 'session-a', expect.any(AbortSignal))
})
it('renders tools and sources, uses final answer, then continues with server session ID', async () => {
  sessionStorage.setItem('zhiye.token', 'token')
  const source = {
    chunk_id: 'chunk',
    document_name: '来源文档',
    content: '来源正文',
    score: 0.9,
    headings: [],
  }
  let turnNumber = 0
  vi.mocked(api.streamChat).mockImplementation(async function* () {
    const turnId = `new-turn-${++turnNumber}`
    yield { type: 'started', session_id: 'new-session', turn_id: turnId }
    yield {
      type: 'function_call',
      call_id: 'call',
      tool_name: 'search_knowledge',
      arguments: { query: '问题' },
    }
    yield {
      type: 'function_call_output',
      call_id: 'call',
      tool_name: 'search_knowledge',
      output: '资料',
      succeeded: true,
      sources: [source],
    }
    yield { type: 'text_delta', delta: '初稿' }
    yield {
      type: 'completed',
      session_id: 'new-session',
      turn_id: turnId,
      answer: '**最终回答**',
      retrieved_sources: [source],
    }
  })
  const user = userEvent.setup()
  render(<App />)
  await user.type(screen.getByLabelText('输入消息'), '问题{Enter}')
  await screen.findByText('最终回答')
  expect(screen.queryByText('初稿')).not.toBeInTheDocument()
  expect(screen.getByText('检索知识库')).toBeInTheDocument()
  expect(screen.getByText('来源文档')).toBeInTheDocument()
  await waitFor(() => expect(screen.getByLabelText('输入消息')).toBeEnabled())
  await user.type(screen.getByLabelText('输入消息'), '继续{Enter}')
  await waitFor(() =>
    expect(api.streamChat).toHaveBeenLastCalledWith(
      'token',
      '继续',
      'new-session',
      expect.any(AbortSignal),
    ),
  )
})
it('stops an in-flight request and preserves partial text', async () => {
  sessionStorage.setItem('zhiye.token', 'token')
  let signal: AbortSignal | undefined
  vi.mocked(api.streamChat).mockImplementation(
    async function* (_token, _message, _session, requestSignal) {
      signal = requestSignal
      yield { type: 'started', session_id: 'new-session', turn_id: 'turn' }
      yield { type: 'text_delta', delta: '部分回答' }
      await new Promise((_resolve, reject) =>
        requestSignal.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          { once: true },
        ),
      )
    },
  )
  const user = userEvent.setup()
  render(<App />)
  await user.type(screen.getByLabelText('输入消息'), '问题{Enter}')
  await screen.findByText('部分回答')
  expect(screen.getByRole('button', { name: '新建会话' })).toBeDisabled()
  await user.click(screen.getByRole('button', { name: '停止生成' }))
  await screen.findByText('已停止生成')
  expect(signal?.aborted).toBe(true)
  expect(screen.getByText('部分回答')).toBeInTheDocument()
  expect(screen.getByLabelText('输入消息')).toBeEnabled()
})
it('prevents stale history requests from replacing a newer selection', async () => {
  sessionStorage.setItem('zhiye.token', 'token')
  const second = { ...session, id: 'session-b', title: '另一个会话' }
  vi.mocked(api.listSessions).mockResolvedValue({ items: [session, second], next_cursor: null })
  let resolveOld!: (value: Session) => void
  vi.mocked(api.getSession).mockImplementation((_token, id) =>
    id === 'session-a'
      ? new Promise((resolve) => {
          resolveOld = resolve
        })
      : Promise.resolve(second),
  )
  const user = userEvent.setup()
  render(<App />)
  await user.click(await screen.findByRole('button', { name: '历史会话' }))
  await user.click(screen.getByRole('button', { name: '另一个会话' }))
  await screen.findByRole('heading', { name: '另一个会话' })
  resolveOld(session)
  await waitFor(() =>
    expect(screen.getByRole('heading', { name: '另一个会话' })).toBeInTheDocument(),
  )
  expect(sessionStorage.getItem('zhiye.session')).toBe('session-b')
})
it('loads older history and deletes the selected session', async () => {
  sessionStorage.setItem('zhiye.token', 'token')
  sessionStorage.setItem('zhiye.session', session.id)
  vi.mocked(api.listTurns)
    .mockResolvedValueOnce({ items: [history], next_cursor: 0 })
    .mockResolvedValueOnce({
      items: [
        {
          ...history,
          id: 'older',
          items: [
            { type: 'message', id: 'older-u', sequence: 0, role: 'user', text: '更早的问题' },
          ],
        },
      ],
      next_cursor: null,
    })
  vi.stubGlobal('confirm', vi.fn().mockReturnValue(true))
  const user = userEvent.setup()
  render(<App />)
  await screen.findByText('历史回答')
  await user.click(screen.getByRole('button', { name: '加载更早的消息' }))
  await screen.findByText('更早的问题')
  expect(api.listTurns).toHaveBeenLastCalledWith('token', session.id, 0, expect.any(AbortSignal))
  vi.mocked(api.listSessions).mockResolvedValue({ items: [], next_cursor: null })
  await user.click(screen.getByRole('button', { name: '删除会话：历史会话' }))
  await screen.findByText('有什么可以帮你？')
  expect(api.deleteSession).toHaveBeenCalledWith('token', session.id)
  expect(sessionStorage.getItem('zhiye.session')).toBeNull()
})
it('keeps SSE failures visible and restores the input', async () => {
  sessionStorage.setItem('zhiye.token', 'token')
  vi.mocked(api.streamChat).mockImplementation(async function* () {
    yield { type: 'started', session_id: 's', turn_id: 't' }
    yield {
      type: 'error',
      code: 'model_error',
      message: '模型处理失败',
      session_id: 's',
      turn_id: 't',
    } as ChatEvent
  })
  render(<App />)
  fireEvent.change(screen.getByLabelText('输入消息'), { target: { value: '问题' } })
  fireEvent.click(screen.getByLabelText('发送消息'))
  await screen.findByText('模型处理失败')
  await waitFor(() => expect(screen.getByLabelText('输入消息')).toBeEnabled())
})
