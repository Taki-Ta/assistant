import type { ChatEvent, Page, Session, Turn } from './types'
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
  }
}
async function checked(response: Response): Promise<Response> {
  if (response.ok) return response
  const data = await response.json().catch(() => null)
  const message =
    response.status === 401
      ? '登录凭证已失效，请在设置中重新填写。'
      : typeof data?.detail === 'string'
        ? data.detail
        : `请求失败（${response.status}），请稍后重试。`
  throw new ApiError(response.status, message)
}
function headers(token: string) {
  return { Authorization: `Bearer ${token}` }
}
async function get<T>(path: string, token: string, signal?: AbortSignal): Promise<T> {
  const response = await checked(await fetch(`/api/v1${path}`, { headers: headers(token), signal }))
  return response.json()
}
export function listSessions(token: string, cursor?: string, signal?: AbortSignal) {
  const query = new URLSearchParams({ limit: '20' })
  if (cursor) query.set('cursor', cursor)
  return get<Page<Session, string>>(`/sessions?${query}`, token, signal)
}
export function getSession(token: string, id: string, signal?: AbortSignal) {
  return get<Session>(`/sessions/${encodeURIComponent(id)}`, token, signal)
}
export function listTurns(token: string, id: string, cursor?: number, signal?: AbortSignal) {
  const query = new URLSearchParams({ limit: '20' })
  if (cursor !== undefined) query.set('cursor', String(cursor))
  return get<Page<Turn, number>>(
    `/sessions/${encodeURIComponent(id)}/turns?${query}`,
    token,
    signal,
  )
}
export async function deleteSession(token: string, id: string) {
  await checked(
    await fetch(`/api/v1/sessions/${encodeURIComponent(id)}`, {
      method: 'DELETE',
      headers: headers(token),
    }),
  )
}
function parseFrame(frame: string): ChatEvent | null {
  let event = ''
  const data: string[] = []
  for (const line of frame.split('\n')) {
    if (line.startsWith(':')) continue
    const colon = line.indexOf(':')
    const field = colon < 0 ? line : line.slice(0, colon)
    const value = colon < 0 ? '' : line.slice(colon + 1).replace(/^ /, '')
    if (field === 'event') event = value
    if (field === 'data') data.push(value)
  }
  if (
    !data.length ||
    ![
      'started',
      'text_delta',
      'function_call',
      'function_call_output',
      'completed',
      'error',
    ].includes(event)
  )
    return null
  const payload = JSON.parse(data.join('\n')) as ChatEvent
  if (payload.type !== event) throw new Error('收到无法识别的回答事件。')
  return payload
}
// 网络分片与 SSE 帧无对应关系；保留未完成帧和跨分片 UTF-8 字符。
export async function* readEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<ChatEvent> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let pendingCR = false
  const append = (text: string, final = false) => {
    text = (pendingCR ? '\r' : '') + text
    pendingCR = !final && text.endsWith('\r')
    if (pendingCR) text = text.slice(0, -1)
    buffer += text.replace(/\r\n?/g, '\n')
  }
  try {
    while (true) {
      const { value, done } = await reader.read()
      append(done ? decoder.decode() : decoder.decode(value, { stream: true }), done)
      let boundary: number
      while ((boundary = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)
        const event = parseFrame(frame)
        if (event) yield event
      }
      if (done) break
    }
  } finally {
    try {
      await reader.cancel()
    } finally {
      reader.releaseLock()
    }
  }
}
export async function* streamChat(
  token: string,
  message: string,
  sessionId: string | null,
  signal: AbortSignal,
) {
  const response = await checked(
    await fetch('/api/v1/chat/stream', {
      method: 'POST',
      headers: {
        ...headers(token),
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({ message, ...(sessionId ? { session_id: sessionId } : {}) }),
      signal,
    }),
  )
  if (!response.headers.get('content-type')?.includes('text/event-stream') || !response.body)
    throw new Error('服务未返回流式回答。')
  for await (const event of readEvents(response.body)) {
    yield event
    if (event.type === 'completed' || event.type === 'error') return
  }
  throw new Error('连接已中断，回答未完成。你可以继续提问。')
}
