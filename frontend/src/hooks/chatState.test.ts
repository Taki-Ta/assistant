import { expect, it } from 'vitest'
import { chatReducer, initialState, type ChatState } from './chatState'
import type { ChatEvent } from '../api/types'
function started() {
  let state = chatReducer(initialState, { type: 'send', message: '问题', id: 'local' })
  return chatReducer(state, {
    type: 'event',
    event: { type: 'started', session_id: 'session', turn_id: 'turn' },
  })
}
function event(state: ChatState, event: ChatEvent) {
  return chatReducer(state, { type: 'event', event })
}
it('keeps intermediate model messages separate and calibrates only the final answer', () => {
  let state = started()
  state = event(state, {
    type: 'text_delta',
    delta: '先检索',
    provider_response_id: 'r1',
    provider_item_id: 'i1',
  })
  state = event(state, {
    type: 'function_call',
    call_id: 'c1',
    tool_name: 'search_knowledge',
    arguments: { query: '问题' },
  })
  state = event(state, {
    type: 'function_call_output',
    call_id: 'c1',
    tool_name: 'search_knowledge',
    output: '资料',
    succeeded: true,
    sources: [],
  })
  state = event(state, {
    type: 'text_delta',
    delta: '最终',
    provider_response_id: 'r2',
    provider_item_id: 'i2',
  })
  state = event(state, {
    type: 'text_delta',
    delta: '回答',
    provider_response_id: 'r2',
    provider_item_id: 'i2',
  })
  state = event(state, {
    type: 'completed',
    session_id: 'session',
    turn_id: 'turn',
    answer: '校准后的回答',
    retrieved_sources: [],
  })
  expect(state.sessionId).toBe('session')
  expect(state.turns[0].items.filter((i) => i.type === 'message').map((i) => i.text)).toEqual([
    '问题',
    '先检索',
    '校准后的回答',
  ])
  expect(state.turns[0].status).toBe('completed')
  expect(state.phase).toBe('idle')
})
it('keeps partial text when stopped or failed and can continue in the same session', () => {
  let state = event(started(), { type: 'text_delta', delta: '部分回答' })
  state = chatReducer(state, { type: 'end', stopped: true })
  expect(state.turns[0].status).toBe('stopped')
  expect(state.turns[0].items.at(-1)).toMatchObject({ text: '部分回答' })
  state = chatReducer(state, { type: 'send', message: '继续', id: 'next' })
  expect(state.turns).toHaveLength(2)
  expect(state.sessionId).toBe('session')
  state = event(state, {
    type: 'error',
    code: 'error',
    message: '模型失败',
    session_id: 'session',
    turn_id: 'next',
  })
  expect(state.turns[1].error_message).toBe('模型失败')
})
it('does not replace a completed turn with stopped status', () => {
  const state = event(started(), {
    type: 'completed',
    session_id: 'session',
    turn_id: 'turn',
    answer: '完成',
    retrieved_sources: [],
  })
  expect(chatReducer(state, { type: 'end', stopped: true }).turns[0].status).toBe('completed')
})
it('prepends older turns without duplicates and resets on new conversation', () => {
  const state = started()
  const old = { ...state.turns[0], id: 'old', sequence: 0 }
  const next = chatReducer(state, { type: 'older', turns: [old, state.turns[0]] })
  expect(next.turns.map((t) => t.id)).toEqual(['old', 'turn'])
  expect(chatReducer(next, { type: 'reset' })).toEqual(initialState)
})
it('appends a final answer after tools when the model emits no final delta', () => {
  let state = event(started(), { type: 'text_delta', delta: '让我查一下' })
  state = event(state, {
    type: 'function_call',
    call_id: 'c',
    tool_name: 'search_knowledge',
    arguments: {},
  })
  state = event(state, {
    type: 'function_call_output',
    call_id: 'c',
    tool_name: 'search_knowledge',
    output: '资料',
    succeeded: true,
    sources: [],
  })
  state = event(state, {
    type: 'completed',
    session_id: 'session',
    turn_id: 'turn',
    answer: '最后的回答',
    retrieved_sources: [],
  })
  expect(state.turns[0].items.filter((i) => i.type === 'message').map((i) => i.text)).toEqual([
    '问题',
    '让我查一下',
    '最后的回答',
  ])
})
