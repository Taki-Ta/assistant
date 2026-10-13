export interface Source {
  chunk_id: string
  document_id?: string | null
  document_name?: string | null
  content: string
  score: number
  headings: string[]
  start_line?: number | null
  end_line?: number | null
}
export interface Session {
  id: string
  title: string | null
  created_at: string
  updated_at: string
}
export interface MessageItem {
  type: 'message'
  id: string
  sequence: number
  role: 'user' | 'assistant' | 'system' | 'developer'
  text: string
  streamKey?: string
}
export interface CallItem {
  type: 'function_call'
  id: string
  sequence: number
  call_id: string
  tool_name: string
  arguments: Record<string, unknown>
}
export interface OutputItem {
  type: 'function_call_output'
  id: string
  sequence: number
  call_id: string
  tool_name: string
  output: string
  succeeded: boolean
  sources: Source[]
}
export type Item = MessageItem | CallItem | OutputItem
export interface Turn {
  id: string
  sequence: number
  status: 'in_progress' | 'completed' | 'failed' | 'cancelled' | 'stopped'
  created_at: string
  completed_at: string | null
  error_message: string | null
  items: Item[]
  sources?: Source[]
}
export interface Page<T, C> {
  items: T[]
  next_cursor: C | null
}
export type ChatEvent =
  | { type: 'started'; session_id: string; turn_id: string }
  | {
      type: 'text_delta'
      delta: string
      stream_key?: string | null
      provider_response_id?: string | null
      provider_item_id?: string | null
    }
  | {
      type: 'function_call'
      call_id: string
      tool_name: string
      arguments: Record<string, unknown>
    }
  | {
      type: 'function_call_output'
      call_id: string
      tool_name: string
      output: string
      succeeded: boolean
      sources: Source[]
    }
  | {
      type: 'completed'
      session_id: string
      turn_id: string
      answer: string
      retrieved_sources: Source[]
    }
  | { type: 'error'; code: string; message: string; session_id: string; turn_id: string }
