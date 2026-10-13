import type { ChatEvent, Item, Turn } from '../api/types'
export interface ChatState {
  sessionId: string | null
  turns: Turn[]
  phase: 'idle' | 'waiting' | 'streaming'
  error: string | null
}
export const initialState: ChatState = { sessionId: null, turns: [], phase: 'idle', error: null }
export type Action =
  | { type: 'reset' }
  | { type: 'history'; sessionId: string; turns: Turn[] }
  | { type: 'older'; turns: Turn[] }
  | { type: 'send'; message: string; id: string }
  | { type: 'event'; event: ChatEvent }
  | { type: 'end'; stopped?: boolean; error?: string }
export function chatReducer(state: ChatState, action: Action): ChatState {
  if (action.type === 'reset') return initialState
  if (action.type === 'history')
    return { ...initialState, sessionId: action.sessionId, turns: action.turns }
  if (action.type === 'older')
    return {
      ...state,
      turns: [
        ...action.turns.filter((t) => !state.turns.some((old) => old.id === t.id)),
        ...state.turns,
      ],
    }
  if (action.type === 'send')
    return {
      ...state,
      phase: 'waiting',
      error: null,
      turns: [
        ...state.turns,
        {
          id: action.id,
          sequence: (state.turns.at(-1)?.sequence ?? -1) + 1,
          status: 'in_progress',
          created_at: new Date().toISOString(),
          completed_at: null,
          error_message: null,
          items: [
            {
              type: 'message',
              id: `${action.id}-user`,
              sequence: 0,
              role: 'user',
              text: action.message,
            },
          ],
        },
      ],
    }
  const turn = state.turns.at(-1)
  if (!turn) return state
  const next = { ...turn, items: [...turn.items] }
  let phase = state.phase
  let sessionId = state.sessionId
  let error = state.error
  if (action.type === 'end') {
    phase = 'idle'
    if (next.status === 'in_progress') {
      next.status = action.stopped ? 'stopped' : 'failed'
      next.error_message = action.stopped ? '已停止生成' : action.error || '回答未完成'
    }
    error = action.error ?? null
  } else {
    const event = action.event
    if (event.type === 'started') {
      sessionId = event.session_id
      next.id = event.turn_id
      phase = 'streaming'
    }
    if (event.type === 'text_delta') {
      const key =
        event.stream_key ?? `${event.provider_response_id ?? ''}/${event.provider_item_id ?? ''}`
      const index = next.items.findIndex(
        (i) => i.type === 'message' && i.role === 'assistant' && i.streamKey === key,
      )
      if (index >= 0) {
        const item = next.items[index]
        if (item.type === 'message') next.items[index] = { ...item, text: item.text + event.delta }
      } else
        next.items.push({
          type: 'message',
          role: 'assistant',
          id: `${next.id}-text-${next.items.length}`,
          sequence: next.items.length,
          text: event.delta,
          streamKey: key,
        })
    }
    if (event.type === 'function_call' || event.type === 'function_call_output')
      next.items.push({
        ...event,
        id: `${next.id}-${next.items.length}`,
        sequence: next.items.length,
      } as Item)
    if (event.type === 'completed') {
      let last = -1
      for (let i = next.items.length - 1; i >= 0; i--) {
        if (next.items[i].type === 'function_call' || next.items[i].type === 'function_call_output')
          break
        if (
          next.items[i].type === 'message' &&
          (next.items[i] as { role: string }).role === 'assistant'
        ) {
          last = i
          break
        }
      }
      if (last >= 0) next.items[last] = { ...next.items[last], text: event.answer } as Item
      else
        next.items.push({
          type: 'message',
          role: 'assistant',
          id: `${next.id}-answer`,
          sequence: next.items.length,
          text: event.answer,
        })
      next.status = 'completed'
      next.sources = event.retrieved_sources
      phase = 'idle'
    }
    if (event.type === 'error') {
      next.status = 'failed'
      next.error_message = event.message
      error = event.message
      phase = 'idle'
    }
  }
  return { ...state, sessionId, phase, error, turns: [...state.turns.slice(0, -1), next] }
}
