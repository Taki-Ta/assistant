// @vitest-environment node
import { describe, expect, it, vi } from 'vitest'
import { ApiError, listTurns, readEvents, streamChat } from './chat'

function stream(text: string, fragment = false) {
  const bytes = new TextEncoder().encode(text)
  return new ReadableStream<Uint8Array>({
    start(controller) {
      if (fragment) for (const byte of bytes) controller.enqueue(Uint8Array.of(byte))
      else controller.enqueue(bytes)
      controller.close()
    },
  })
}
async function collect<T>(events: AsyncIterable<T>) {
  const result: T[] = []
  for await (const event of events) result.push(event)
  return result
}
const started =
  'event: started\ndata: {"type":"started","session_id":"session","turn_id":"turn"}\n\n'
const completed =
  'event: completed\ndata: {"type":"completed","session_id":"session","turn_id":"turn","answer":"你好","retrieved_sources":[]}\n\n'

describe('SSE', () => {
  it('decodes byte-by-byte UTF-8, CRLF boundaries, heartbeats and multiple frames', async () => {
    const text =
      started +
      ': ping\n\n' +
      'event: text_delta\ndata: {"type":"text_delta","delta":"你好🌱"}\n\n' +
      completed
    const events = await collect(readEvents(stream(text.replaceAll('\n', '\r\n'), true)))
    expect(events.map((event) => event.type)).toEqual(['started', 'text_delta', 'completed'])
    expect(events[1]).toMatchObject({ delta: '你好🌱' })
  })
  it('supports multiple data lines and ignores unknown events', async () => {
    const text =
      'event: ignored\ndata: nope\n\nevent: text_delta\ndata: {"type":"text_delta",\ndata: "delta":"a"}\n\n'
    expect(await collect(readEvents(stream(text)))).toEqual([{ type: 'text_delta', delta: 'a' }])
  })
  it('does not dispatch incomplete frames', async () => {
    expect(await collect(readEvents(stream(started.trimEnd())))).toEqual([])
  })
  it('releases and cancels a reader when the consumer stops early', async () => {
    const cancel = vi.fn()
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(started))
      },
      cancel,
    })
    for await (const _event of readEvents(body)) break
    expect(cancel).toHaveBeenCalledOnce()
    expect(body.locked).toBe(false)
  })
  it('rejects invalid payloads', async () => {
    await expect(
      collect(readEvents(stream('event: started\ndata: {"type":"completed"}\n\n'))),
    ).rejects.toThrow('无法识别')
  })
  it('sends only current message, session ID and bearer authorization', async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(stream(started + completed), {
        headers: { 'Content-Type': 'text/event-stream' },
      }),
    )
    vi.stubGlobal('fetch', fetch)
    const signal = new AbortController().signal
    const events = await collect(streamChat('test-token', '你好', 'session', signal))
    expect(events.at(-1)?.type).toBe('completed')
    const [url, options] = fetch.mock.calls[0]
    expect(url).toBe('/api/v1/chat/stream')
    expect(options.signal).toBe(signal)
    expect(options.headers.Authorization).toBe('Bearer test-token')
    expect(JSON.parse(options.body)).toEqual({ message: '你好', session_id: 'session' })
  })
  it('reports premature EOF instead of treating it as completion', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          new Response(stream(started), { headers: { 'Content-Type': 'text/event-stream' } }),
        ),
    )
    await expect(
      collect(streamChat('token', '问题', null, new AbortController().signal)),
    ).rejects.toThrow('连接已中断')
  })
  it('propagates SSE errors as terminal events', async () => {
    const error =
      'event: error\ndata: {"type":"error","code":"model_error","message":"失败","session_id":"session","turn_id":"turn"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(stream(started + error), {
          headers: { 'Content-Type': 'text/event-stream' },
        }),
      ),
    )
    expect(
      (await collect(streamChat('token', '问题', null, new AbortController().signal))).at(-1),
    ).toMatchObject({ type: 'error', message: '失败' })
  })
  it('reports a safe credential error on HTTP 401', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('{"detail":"private"}', { status: 401 })),
    )
    await expect(listTurns('token', 'session', 0)).rejects.toBeInstanceOf(ApiError)
    await expect(listTurns('token', 'session', 0)).rejects.toThrow('登录凭证已失效')
  })
  it('preserves cursor zero when loading history', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('{"items":[],"next_cursor":null}'))
    vi.stubGlobal('fetch', fetch)
    await listTurns('token', 'session', 0)
    expect(fetch.mock.calls[0][0]).toContain('cursor=0')
  })
})
