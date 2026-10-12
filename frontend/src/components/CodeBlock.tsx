import { useRef, useState, type ReactNode } from 'react'
import { Check, Copy } from 'lucide-react'
export function CodeBlock({ children }: { children?: ReactNode }) {
  const pre = useRef<HTMLPreElement>(null)
  const [copied, setCopied] = useState(false)
  const [failed, setFailed] = useState(false)
  async function copy() {
    try {
      await navigator.clipboard.writeText(pre.current?.textContent || '')
      setCopied(true)
      setFailed(false)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setFailed(true)
    }
  }
  return (
    <div className="relative">
      <button
        aria-label="复制代码"
        onClick={copy}
        className="absolute top-2 right-2 rounded-md bg-[#f0f1ec] p-1.5 text-stone-400 hover:text-stone-700"
        title={failed ? '无法复制，请手动选择代码' : '复制代码'}
      >
        {copied ? <Check size={13} /> : <Copy size={13} />}
      </button>
      <pre ref={pre}>{children}</pre>
      {failed && (
        <p role="status" className="text-xs text-stone-500">
          无法复制，请手动选择代码。
        </p>
      )}
    </div>
  )
}
