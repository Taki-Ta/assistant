import { useState } from 'react'
import { Check, Copy } from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { MessageItem as Message } from '../api/types'
import { CodeBlock } from './CodeBlock'
export function MessageItem({ item }: { item: Message }) {
  const [copied, setCopied] = useState(false)
  const [copyError, setCopyError] = useState(false)
  async function copy() {
    try {
      await navigator.clipboard.writeText(item.text)
      setCopied(true)
      setCopyError(false)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopyError(true)
    }
  }
  if (item.role === 'user')
    return (
      <div className="my-7 flex justify-end">
        <div className="max-w-[85%] rounded-3xl bg-[#eeeee9] px-5 py-3 text-[15px] leading-7 whitespace-pre-wrap break-words">
          {item.text}
        </div>
      </div>
    )
  if (item.role !== 'assistant') return null
  return (
    <article className="group my-5">
      <div className="markdown text-[15px] leading-7">
        <Markdown
          remarkPlugins={[remarkGfm]}
          components={{
            pre: CodeBlock,
            a: ({ children, ...props }) => (
              <a {...props} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            ),
          }}
        >
          {item.text}
        </Markdown>
      </div>
      <div className="mt-3 flex items-center gap-2">
        <button
          aria-label="复制回答"
          title="复制回答"
          className="rounded-md p-1.5 text-stone-400 transition hover:bg-stone-100 hover:text-stone-700"
          onClick={copy}
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
        </button>
        {copied && (
          <span role="status" className="text-xs text-stone-400">
            已复制
          </span>
        )}
        {copyError && (
          <span role="status" className="text-xs text-stone-500">
            无法复制，请手动选择文本。
          </span>
        )}
      </div>
    </article>
  )
}
