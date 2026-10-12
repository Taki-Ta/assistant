import { BookOpen, ChevronDown } from 'lucide-react'
import type { Source, Turn } from '../api/types'
export function turnSources(turn: Turn): Source[] {
  const sources = new Map<string, Source>()
  const all =
    turn.sources ??
    turn.items.flatMap((item) =>
      item.type === 'function_call_output' && item.succeeded ? item.sources : [],
    )
  for (const source of all) {
    const old = sources.get(source.chunk_id)
    if (!old || source.score > old.score) sources.set(source.chunk_id, source)
  }
  return [...sources.values()].sort((a, b) => b.score - a.score)
}
export function Sources({ sources }: { sources: Source[] }) {
  if (!sources.length) return null
  return (
    <details className="mt-5 text-sm">
      <summary className="flex w-fit cursor-pointer list-none items-center gap-2 rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs text-stone-600 hover:bg-stone-50">
        <BookOpen size={13} />
        参考来源 <span className="text-stone-400">{sources.length}</span>
        <ChevronDown size={12} />
      </summary>
      <div className="mt-3 grid gap-2">
        {sources.map((source) => (
          <details
            key={source.chunk_id}
            className="rounded-xl border border-stone-200 bg-white p-3"
          >
            <summary className="cursor-pointer text-stone-700">
              {source.document_name || '知识库文档'}
              {source.headings?.length ? (
                <span className="ml-2 text-xs text-stone-400">{source.headings.join(' / ')}</span>
              ) : null}
            </summary>
            {source.start_line != null && (
              <p className="mt-2 text-xs text-stone-400">
                第 {source.start_line}
                {source.end_line != null && source.end_line !== source.start_line
                  ? `–${source.end_line}`
                  : ''}{' '}
                行
              </p>
            )}
            <p className="mt-2 max-h-64 overflow-y-auto whitespace-pre-wrap break-words text-xs leading-6 text-stone-600">
              {source.content}
            </p>
          </details>
        ))}
      </div>
    </details>
  )
}
