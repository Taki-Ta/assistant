import { useEffect, useReducer, useRef, useState } from 'react'
import { chatReducer, initialState } from './chatState'
import { streamChat } from '../api/chat'
export function useChat(token: string, onFinished: () => void) {
  const [busy, setBusy] = useState(false)
  const [state, dispatch] = useReducer(chatReducer, initialState)
  const mounted = useRef(false)
  const controller = useRef<AbortController | null>(null)
  const finish = useRef(onFinished)
  finish.current = onFinished
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      controller.current?.abort()
    }
  }, [])
  async function send(message: string) {
    if (controller.current || !token || !message.trim()) return
    const request = new AbortController()
    controller.current = request
    setBusy(true)
    dispatch({ type: 'send', message: message.trim(), id: crypto.randomUUID() })
    try {
      for await (const event of streamChat(
        token,
        message.trim(),
        state.sessionId,
        request.signal,
      )) {
        if (mounted.current) dispatch({ type: 'event', event })
      }
      if (mounted.current) dispatch({ type: 'end' })
    } catch (error) {
      if (mounted.current)
        dispatch({
          type: 'end',
          stopped: request.signal.aborted,
          error: request.signal.aborted
            ? undefined
            : error instanceof Error
              ? error.message
              : '回答失败，请稍后重试。',
        })
    } finally {
      controller.current = null
      if (mounted.current) {
        setBusy(false)
        finish.current()
      }
    }
  }
  return { state, dispatch, send, busy, stop: () => controller.current?.abort() }
}
