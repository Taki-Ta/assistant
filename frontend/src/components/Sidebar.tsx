import { BookOpen, MessageSquare, Plus, Settings, Trash2, X } from 'lucide-react'
import type { Session } from '../api/types'
export function Sidebar({
  sessions,
  activeId,
  disabled,
  loading,
  hasMore,
  token,
  open,
  onClose,
  onNew,
  onSelect,
  onDelete,
  onSettings,
  onMore,
  onRefresh,
}: {
  sessions: Session[]
  activeId: string | null
  disabled: boolean
  loading: boolean
  hasMore: boolean
  token: boolean
  open: boolean
  onClose: () => void
  onNew: () => void
  onSelect: (id: string) => void
  onDelete: (session: Session) => void
  onSettings: () => void
  onMore: () => void
  onRefresh: () => void
}) {
  return (
    <>
      {open && (
        <button
          aria-label="关闭会话列表遮罩"
          className="fixed inset-0 z-20 bg-black/20 md:hidden"
          onClick={onClose}
        />
      )}
      <aside
        className={`${open ? 'translate-x-0' : '-translate-x-full'} fixed inset-y-0 left-0 z-30 flex w-64 flex-col border-r border-stone-200/60 bg-[#f2f3ee] px-3 py-5 transition-transform md:static md:translate-x-0`}
      >
        <div className="mb-7 flex items-center gap-2.5 px-3">
          <div className="grid size-8 place-items-center rounded-xl bg-[#dce6dc] text-[#37644d]">
            <BookOpen size={18} strokeWidth={1.6} />
          </div>
          <span className="text-lg font-semibold tracking-tight">
            知页<span className="ml-2 text-xs font-normal text-stone-400">ZHIYE</span>
          </span>
          <button aria-label="关闭会话列表" className="ml-auto md:hidden" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <button
          disabled={disabled}
          onClick={onNew}
          className="mb-7 flex items-center gap-2 rounded-xl border border-stone-200 bg-white/80 px-3 py-3 text-sm shadow-[0_1px_2px_#00000003] hover:bg-white disabled:opacity-40"
        >
          <Plus size={17} />
          新建会话
        </button>
        <div className="mb-2 flex items-center justify-between px-3">
          <span className="text-[11px] text-stone-400">最近会话</span>
          <button
            disabled={!token || loading || disabled}
            onClick={onRefresh}
            className="text-[11px] text-stone-400 hover:text-stone-700 disabled:opacity-40"
          >
            刷新
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          {!sessions.length && (
            <p className="px-3 py-4 text-xs leading-6 text-stone-400">
              {!token
                ? '配置登录凭证后，开始你的第一段对话。'
                : loading
                  ? '正在加载会话…'
                  : '还没有会话，聊点什么吧。'}
            </p>
          )}
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`group mb-1 flex items-center rounded-lg ${activeId === session.id ? 'bg-[#e3e6dd]' : 'hover:bg-[#e8eae3]'}`}
            >
              <button
                disabled={disabled}
                onClick={() => onSelect(session.id)}
                className="flex min-w-0 flex-1 items-center gap-2.5 px-3 py-2.5 text-left text-[13px] disabled:opacity-40"
              >
                <MessageSquare className="shrink-0 text-stone-400" size={14} />
                <span className="truncate">{session.title || '新会话'}</span>
              </button>
              <button
                aria-label={`删除会话：${session.title || '新会话'}`}
                disabled={disabled}
                onClick={() => onDelete(session)}
                className="mr-1 rounded-md p-2 text-stone-400 opacity-100 hover:bg-white/60 hover:text-stone-700 focus:opacity-100 md:opacity-0 md:group-hover:opacity-100 disabled:opacity-30"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
          {hasMore && (
            <button
              disabled={loading || disabled}
              onClick={onMore}
              className="w-full py-3 text-xs text-stone-400 disabled:opacity-40"
            >
              {loading ? '加载中…' : '加载更多会话'}
            </button>
          )}
        </div>
        <button
          disabled={disabled}
          onClick={onSettings}
          className="mt-4 flex items-center gap-3 rounded-xl border-t border-stone-200/70 px-3 pt-5 pb-2 text-left disabled:opacity-40"
        >
          <div className="grid size-8 place-items-center rounded-full bg-stone-200/80 text-xs text-stone-600">
            访
          </div>
          <div className="flex-1">
            <p className="text-xs font-medium">我的工作空间</p>
            <p className="mt-1 text-[10px] text-stone-400">
              {token ? '登录凭证已配置' : '设置登录凭证'}
            </p>
          </div>
          <Settings size={16} className="text-stone-400" />
        </button>
      </aside>
    </>
  )
}
