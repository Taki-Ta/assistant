import { Check, ChevronDown, LoaderCircle, Search, TriangleAlert } from 'lucide-react'
import type { CallItem, OutputItem } from '../api/types'
export function ToolActivity({
  call,
  result,
  active,
}: {
  call: CallItem
  result?: OutputItem
  active: boolean
}) {
  const label = call.tool_name === 'search_knowledge' ? '检索知识库' : '执行工具'
  const state = result ? (result.succeeded ? '完成' : '失败') : active ? '进行中' : '未完成'
  return (
    <details className="my-4 max-w-full text-xs text-stone-500">
      <summary className="flex w-fit cursor-pointer list-none items-center gap-2 rounded-lg py-1 hover:text-stone-800">
        {result ? (
          result.succeeded ? (
            <Check size={14} className="text-emerald-700" />
          ) : (
            <TriangleAlert size={14} />
          )
        ) : active ? (
          <LoaderCircle size={14} className="animate-spin" />
        ) : (
          <Search size={14} />
        )}
        {label}
        <span className="text-stone-400">· {state}</span>
        <ChevronDown size={12} />
      </summary>
      <div className="mt-2 rounded-xl border border-stone-200 bg-stone-50 p-3">
        <p className="mb-2 font-medium">{call.tool_name}</p>
        <pre className="max-h-32 overflow-auto whitespace-pre-wrap break-all leading-5">
          {JSON.stringify(call.arguments, null, 2)}
        </pre>
        {result && (
          <>
            <p className="mt-3 mb-1 font-medium">执行结果</p>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-all leading-5">
              {result.output}
            </pre>
          </>
        )}
      </div>
    </details>
  )
}
