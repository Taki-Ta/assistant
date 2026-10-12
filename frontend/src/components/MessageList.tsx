import { useLayoutEffect, useRef, useState } from 'react'
import { ArrowDown, LoaderCircle } from 'lucide-react'
import type { OutputItem, Turn } from '../api/types'
import { MessageItem } from './MessageItem'
import { ToolActivity } from './ToolActivity'
import { Sources, turnSources } from './Sources'
export function MessageList({
  turns,
  sessionId,
  busy,
  hasOlder,
  loadingOlder,
  onOlder,
}: {
  turns: Turn[]
  sessionId: string | null
  busy: boolean
  hasOlder: boolean
  loadingOlder: boolean
  onOlder: () => void
}) {
  const container = useRef<HTMLDivElement>(null)
  const follow = useRef(true)
  const olderHeight = useRef<number | null>(null)
  const [atBottom, setAtBottom] = useState(true)
  useLayoutEffect(() => {
    follow.current = true
    setAtBottom(true)
  }, [sessionId])
  useLayoutEffect(() => {
    const node = container.current
    if (!node) return
    if (olderHeight.current !== null) {
      node.scrollTop += node.scrollHeight - olderHeight.current
      olderHeight.current = null
    } else if (follow.current) node.scrollTop = node.scrollHeight
  }, [turns, sessionId])
  function loadOlder() {
    if (container.current) olderHeight.current = container.current.scrollHeight
    onOlder()
  }
  // 失败时恢复滚动行为，避免下一条消息错误地按“加载历史”处理。
  useLayoutEffect(() => {
    if (!loadingOlder) olderHeight.current = null
  }, [loadingOlder])
  return (
    <div className="relative min-h-0 flex-1">
      <div
        ref={container}
        className="h-full overflow-y-auto overscroll-contain px-5"
        onScroll={(e) => {
          const node = e.currentTarget
          const bottom = node.scrollHeight - node.scrollTop - node.clientHeight < 100
          follow.current = bottom
          setAtBottom(bottom)
        }}
      >
        <div className="mx-auto max-w-3xl pb-8 pt-4">
          {hasOlder && (
            <div className="flex justify-center">
              <button
                className="rounded-full px-4 py-2 text-xs text-stone-500 hover:bg-stone-100 disabled:opacity-50"
                disabled={loadingOlder || busy}
                onClick={loadOlder}
              >
                {loadingOlder ? '正在加载…' : '加载更早的消息'}
              </button>
            </div>
          )}
          {turns.map((turn, index) => {
            const active = busy && index === turns.length - 1
            const outputs = new Map(
              turn.items
                .filter((item): item is OutputItem => item.type === 'function_call_output')
                .map((item) => [item.call_id, item]),
            )
            return (
              <section key={turn.id}>
                {turn.items.map((item) =>
                  item.type === 'message' ? (
                    <MessageItem key={item.id} item={item} />
                  ) : item.type === 'function_call' ? (
                    <ToolActivity
                      key={item.id}
                      call={item}
                      result={outputs.get(item.call_id)}
                      active={active}
                    />
                  ) : null,
                )}
                {active &&
                  !turn.items.some(
                    (item) => item.type === 'message' && item.role === 'assistant',
                  ) && (
                    <div
                      role="status"
                      className="my-6 flex items-center gap-2 text-xs text-stone-400"
                    >
                      <LoaderCircle size={15} className="animate-spin" />
                      正在思考…
                    </div>
                  )}
                {!active && turn.status === 'in_progress' && (
                  <p className="my-4 text-xs text-stone-400">
                    这一轮仍在处理中，稍后重新打开会话查看结果。
                  </p>
                )}
                {turn.error_message && (
                  <p role="status" className="my-4 text-xs text-stone-500">
                    {turn.error_message}
                  </p>
                )}
                <Sources sources={turnSources(turn)} />
              </section>
            )
          })}
        </div>
      </div>
      {!atBottom && (
        <button
          aria-label="回到底部"
          onClick={() => {
            follow.current = true
            setAtBottom(true)
            container.current?.scrollTo({ top: container.current.scrollHeight, behavior: 'smooth' })
          }}
          className="absolute bottom-4 left-1/2 grid size-9 -translate-x-1/2 place-items-center rounded-full border border-stone-200 bg-white shadow-sm"
        >
          <ArrowDown size={16} />
        </button>
      )}
    </div>
  )
}
