import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowUpRight, BookOpen, FileText, LoaderCircle, Menu, Search, X } from 'lucide-react'
import * as api from './api/chat'
import type { Session } from './api/types'
import { useChat } from './hooks/useChat'
import { Composer } from './components/Composer'
import { Sidebar } from './components/Sidebar'
import { SettingsDialog } from './components/SettingsDialog'
import { MessageList } from './components/MessageList'

function stored(key: string) {
  try {
    return sessionStorage.getItem(key) || ''
  } catch {
    return ''
  }
}
function store(key: string, value: string) {
  try {
    if (value) sessionStorage.setItem(key, value)
    else sessionStorage.removeItem(key)
  } catch {
    /* 存储被禁用时仍可使用内存中的状态。 */
  }
}
const TOKEN_KEY = 'zhiye.token'
const SESSION_KEY = 'zhiye.session'
const suggestions = [
  { icon: Search, title: '找到问题的答案', text: '帮我查找知识库中与这个问题相关的资料' },
  { icon: FileText, title: '整理重点内容', text: '请总结知识库中的核心内容，并列出重点' },
  { icon: BookOpen, title: '把知识讲清楚', text: '请结合知识库，用简单的语言解释一个概念' },
]

export default function App() {
  const [token, setToken] = useState(() => stored(TOKEN_KEY))
  const [sessions, setSessions] = useState<Session[]>([])
  const [sessionCursor, setSessionCursor] = useState<string | null>(null)
  const [turnCursor, setTurnCursor] = useState<number | null>(null)
  const [loadingSessions, setLoadingSessions] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [loadingOlder, setLoadingOlder] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [draft, setDraft] = useState('')
  const [selectedTitle, setSelectedTitle] = useState('')
  const sessionRequest = useRef<AbortController | null>(null)
  const historyRequest = useRef<AbortController | null>(null)

  const refreshSessions = useCallback(
    async (cursor?: string) => {
      if (!token) return
      sessionRequest.current?.abort()
      const request = new AbortController()
      sessionRequest.current = request
      setLoadingSessions(true)
      try {
        const page = await api.listSessions(token, cursor, request.signal)
        if (request.signal.aborted) return
        setSessions((old) =>
          cursor
            ? [
                ...old,
                ...page.items.filter((item) => !old.some((existing) => existing.id === item.id)),
              ]
            : page.items,
        )
        setSessionCursor(page.next_cursor)
      } catch (exc) {
        if (!request.signal.aborted) setError(exc instanceof Error ? exc.message : '会话加载失败。')
      } finally {
        if (!request.signal.aborted) setLoadingSessions(false)
      }
    },
    [token],
  )

  const {
    state,
    dispatch,
    send,
    stop,
    busy: generating,
  } = useChat(token, () => {
    void refreshSessions()
  })
  const busy = generating || loadingHistory || deleting
  const pendingHistoryTurn = !generating && state.turns.at(-1)?.status === 'in_progress'

  async function selectSession(id: string) {
    if (generating || deleting) return
    historyRequest.current?.abort()
    const request = new AbortController()
    historyRequest.current = request
    setLoadingHistory(true)
    setLoadingOlder(false)
    setError(null)
    setSidebarOpen(false)
    try {
      const [session, page] = await Promise.all([
        api.getSession(token, id, request.signal),
        api.listTurns(token, id, undefined, request.signal),
      ])
      if (request.signal.aborted) return
      dispatch({ type: 'history', sessionId: id, turns: page.items })
      setSelectedTitle(session.title || '新会话')
      setTurnCursor(page.next_cursor)
      setDraft('')
      store(SESSION_KEY, id)
    } catch (exc) {
      if (!request.signal.aborted) {
        setError(exc instanceof Error ? exc.message : '历史消息加载失败。')
        if (exc instanceof api.ApiError && exc.status === 404) store(SESSION_KEY, '')
      }
    } finally {
      if (!request.signal.aborted) setLoadingHistory(false)
    }
  }

  useEffect(() => {
    historyRequest.current?.abort()
    sessionRequest.current?.abort()
    dispatch({ type: 'reset' })
    setSessions([])
    setSessionCursor(null)
    setTurnCursor(null)
    setSelectedTitle('')
    setError(null)
    setLoadingHistory(false)
    setLoadingOlder(false)
    setLoadingSessions(false)
    if (token) {
      void refreshSessions()
      const id = stored(SESSION_KEY)
      if (id) void selectSession(id)
    }
    return () => {
      historyRequest.current?.abort()
      sessionRequest.current?.abort()
    }
    // 切换凭证时重载账号数据；selectSession 只在这个 effect 中读取本次 token。
  }, [token, refreshSessions, dispatch])
  useEffect(() => {
    if (state.sessionId) store(SESSION_KEY, state.sessionId)
  }, [state.sessionId])

  function newSession() {
    if (generating || deleting) return
    historyRequest.current?.abort()
    setLoadingHistory(false)
    setLoadingOlder(false)
    dispatch({ type: 'reset' })
    setTurnCursor(null)
    setSelectedTitle('')
    setDraft('')
    setError(null)
    store(SESSION_KEY, '')
    setSidebarOpen(false)
  }
  async function loadOlder() {
    if (!state.sessionId || turnCursor === null || busy || loadingOlder) return
    const request = new AbortController()
    historyRequest.current = request
    setLoadingOlder(true)
    try {
      const page = await api.listTurns(token, state.sessionId, turnCursor, request.signal)
      if (!request.signal.aborted) {
        dispatch({ type: 'older', turns: page.items })
        setTurnCursor(page.next_cursor)
      }
    } catch (exc) {
      if (!request.signal.aborted)
        setError(exc instanceof Error ? exc.message : '历史消息加载失败。')
    } finally {
      if (!request.signal.aborted) setLoadingOlder(false)
    }
  }
  async function removeSession(session: Session) {
    if (busy || !window.confirm(`删除会话“${session.title || '新会话'}”？`)) return
    setDeleting(true)
    setError(null)
    try {
      await api.deleteSession(token, session.id)
      if (state.sessionId === session.id) {
        dispatch({ type: 'reset' })
        store(SESSION_KEY, '')
        setSelectedTitle('')
        setTurnCursor(null)
        setDraft('')
      }
      setSessions((old) => old.filter((item) => item.id !== session.id))
      void refreshSessions()
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '删除会话失败。')
    } finally {
      setDeleting(false)
    }
  }
  function saveToken(value: string) {
    store(TOKEN_KEY, value)
    store(SESSION_KEY, '')
    setToken(value)
  }
  function sendMessage(message: string) {
    if (!token) {
      setSettingsOpen(true)
      return
    }
    if (busy || loadingOlder || pendingHistoryTurn) return
    setError(null)
    if (!state.sessionId) setSelectedTitle(message.trim().replace(/\s+/g, ' ').slice(0, 80))
    void send(message)
  }

  return (
    <div className="flex h-dvh overflow-hidden bg-[#fafaf8]">
      <Sidebar
        sessions={sessions}
        activeId={state.sessionId}
        disabled={generating || deleting || loadingOlder}
        loading={loadingSessions}
        hasMore={sessionCursor !== null}
        token={!!token}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNew={newSession}
        onSelect={(id) => void selectSession(id)}
        onDelete={(session) => void removeSession(session)}
        onSettings={() => setSettingsOpen(true)}
        onMore={() => {
          if (sessionCursor) void refreshSessions(sessionCursor)
        }}
        onRefresh={() => {
          setError(null)
          void refreshSessions()
        }}
      />
      <main className="relative flex min-w-0 flex-1 flex-col">
        <header className="flex h-18 shrink-0 items-center justify-between px-5 md:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <button
              aria-label="打开会话列表"
              className="rounded-lg p-1.5 text-stone-500 hover:bg-stone-100 md:hidden"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu size={20} />
            </button>
            <h1 className="truncate text-sm font-medium text-stone-600">
              {selectedTitle || '知识库助手'}
            </h1>
          </div>
          <div className="hidden items-center gap-1.5 text-[11px] text-stone-400 sm:flex">
            <span className="size-1.5 rounded-full bg-[#92ad97]" />
            基于你的知识库
          </div>
        </header>
        {error && (
          <div
            role="alert"
            className="mx-5 mb-2 flex items-center gap-3 rounded-xl border border-stone-200 bg-white p-3 text-xs text-stone-600"
          >
            <span className="flex-1">{error}</span>
            <button
              disabled={busy}
              onClick={() => setSettingsOpen(true)}
              className="underline underline-offset-2"
            >
              设置凭证
            </button>
            <button aria-label="关闭提示" onClick={() => setError(null)}>
              <X size={14} />
            </button>
          </div>
        )}
        {loadingHistory ? (
          <div className="flex flex-1 items-center justify-center gap-2 text-sm text-stone-400">
            <LoaderCircle size={18} className="animate-spin" />
            正在恢复会话…
          </div>
        ) : state.turns.length ? (
          <MessageList
            turns={state.turns}
            sessionId={state.sessionId}
            busy={generating}
            hasOlder={turnCursor !== null}
            loadingOlder={loadingOlder}
            onOlder={() => void loadOlder()}
          />
        ) : (
          <div className="flex min-h-0 flex-1 flex-col items-center justify-center overflow-y-auto px-6 pb-8">
            <div className="mb-7 grid size-15 place-items-center rounded-2xl border border-[#dce3d8] bg-[#edf1e9] text-[#55725b]">
              <BookOpen size={28} strokeWidth={1.3} />
            </div>
            <h2 className="text-center text-2xl font-medium tracking-tight md:text-3xl">
              有什么可以帮你？
            </h2>
            <p className="mt-3 text-center text-sm text-stone-400">
              从一个问题开始，让知识触手可及。
            </p>
            <div className="mt-10 grid w-full max-w-2xl gap-3 sm:grid-cols-3">
              {suggestions.map(({ icon: Icon, title, text }) => (
                <button
                  key={title}
                  onClick={() => setDraft(text)}
                  className="group rounded-2xl border border-stone-200/80 bg-white/50 p-4 text-left transition hover:border-stone-300 hover:bg-white"
                >
                  <div className="flex items-center justify-between">
                    <Icon size={17} strokeWidth={1.5} className="text-stone-400" />
                    <ArrowUpRight size={13} className="text-stone-300 group-hover:text-stone-500" />
                  </div>
                  <p className="mt-4 text-xs text-stone-600">{title}</p>
                </button>
              ))}
            </div>
            {!token && (
              <button
                onClick={() => setSettingsOpen(true)}
                className="mt-6 text-xs text-[#647c66] underline decoration-[#a4b4a3] underline-offset-4"
              >
                设置登录凭证，连接知识库
              </button>
            )}
          </div>
        )}
        <div className="mx-auto w-full max-w-[808px] shrink-0 px-5 pt-2 pb-5 md:px-6 md:pb-6">
          {pendingHistoryTurn && (
            <p className="mb-2 text-center text-xs text-stone-400">
              上一轮仍在处理中，重新打开当前会话以刷新结果。
            </p>
          )}
          <Composer
            busy={generating}
            disabled={!token || loadingHistory || deleting || loadingOlder || pendingHistoryTurn}
            onSend={sendMessage}
            onStop={stop}
            draft={draft}
            onDraftChange={setDraft}
          />
        </div>
      </main>
      {settingsOpen && (
        <SettingsDialog token={token} onSave={saveToken} onClose={() => setSettingsOpen(false)} />
      )}
    </div>
  )
}
