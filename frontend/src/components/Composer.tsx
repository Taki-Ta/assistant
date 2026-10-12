import { useLayoutEffect, useRef, useState } from 'react'
import { ArrowUp, Square } from 'lucide-react'
export function Composer({
  busy,
  disabled,
  onSend,
  onStop,
  draft,
  onDraftChange,
}: {
  busy: boolean
  disabled: boolean
  onSend: (text: string) => void
  onStop: () => void
  draft: string
  onDraftChange: (text: string) => void
}) {
  const input = useRef<HTMLTextAreaElement>(null)
  const [composing, setComposing] = useState(false)
  useLayoutEffect(() => {
    if (input.current) {
      input.current.style.height = 'auto'
      input.current.style.height = `${Math.min(input.current.scrollHeight, 180)}px`
    }
  }, [draft])
  function send() {
    if (!busy && !disabled && draft.trim()) {
      onSend(draft)
      onDraftChange('')
      input.current?.focus()
    }
  }
  return (
    <div className="w-full">
      <div className="composer rounded-3xl border border-stone-200 bg-white p-3 shadow-[0_4px_24px_#00000006] focus-within:border-stone-300">
        <textarea
          ref={input}
          aria-label="输入消息"
          placeholder="向知识库提问…"
          className="max-h-44 min-h-12 w-full resize-none bg-transparent px-2 py-2 text-[15px] leading-6 outline-none placeholder:text-stone-400 disabled:opacity-60"
          rows={1}
          value={draft}
          disabled={disabled || busy}
          onChange={(e) => onDraftChange(e.target.value)}
          onCompositionStart={() => setComposing(true)}
          onCompositionEnd={() => setComposing(false)}
          onKeyDown={(e) => {
            if (
              e.key === 'Enter' &&
              !e.shiftKey &&
              !composing &&
              !e.nativeEvent.isComposing &&
              e.keyCode !== 229
            ) {
              e.preventDefault()
              send()
            }
          }}
        />
        <div className="flex items-center justify-between pl-2">
          <span className="text-xs text-stone-400">
            {busy ? '正在生成回答' : 'Enter 发送 · Shift + Enter 换行'}
          </span>
          <button
            type="button"
            aria-label={busy ? '停止生成' : '发送消息'}
            onClick={busy ? onStop : send}
            disabled={!busy && (disabled || !draft.trim())}
            className="grid size-9 place-items-center rounded-full bg-stone-800 text-white transition hover:bg-stone-700 disabled:bg-stone-200 disabled:text-stone-400"
          >
            {busy ? <Square size={14} fill="currentColor" /> : <ArrowUp size={18} />}
          </button>
        </div>
      </div>
      <p className="mt-3 text-center text-[11px] text-stone-400">
        回答由 AI 生成，重要信息请结合引用来源核实。
      </p>
    </div>
  )
}
