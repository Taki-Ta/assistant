import { useEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'
export function SettingsDialog({
  token,
  onSave,
  onClose,
}: {
  token: string
  onSave: (token: string) => void
  onClose: () => void
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [value, setValue] = useState(token)
  useEffect(() => {
    dialog.current?.showModal()
    return () => dialog.current?.close()
  }, [])
  return (
    <dialog
      ref={dialog}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
      className="m-auto w-[calc(100%_-_2rem)] max-w-md rounded-2xl border border-stone-200 bg-white p-6 text-stone-800 shadow-xl backdrop:bg-black/25"
    >
      <div className="mb-5 flex items-center justify-between">
        <h2 className="text-lg font-medium">连接你的知识库</h2>
        <button
          aria-label="关闭设置"
          onClick={onClose}
          className="rounded-md p-1 text-stone-400 hover:bg-stone-100"
        >
          <X size={18} />
        </button>
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          onSave(value.trim().replace(/^Bearer\s+/i, ''))
          onClose()
        }}
      >
        <label htmlFor="token" className="mb-2 block text-sm">
          登录凭证
        </label>
        <input
          id="token"
          type="password"
          autoComplete="off"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="粘贴 Bearer Token"
          className="w-full rounded-xl border border-stone-200 px-3 py-2.5 text-sm outline-none focus:border-stone-400"
        />
        <p className="mt-3 text-xs leading-6 text-stone-400">
          凭证仅保存在当前浏览器会话中。保存后会从服务端加载你的会话记录。
        </p>
        <div className="mt-6 flex items-center justify-between">
          <button
            type="button"
            onClick={() => {
              onSave('')
              onClose()
            }}
            className="text-xs text-stone-500 hover:text-stone-800"
          >
            清除凭证
          </button>
          <button
            type="submit"
            disabled={!value.trim()}
            className="rounded-lg bg-stone-800 px-5 py-2.5 text-sm text-white hover:bg-stone-700 disabled:opacity-40"
          >
            保存
          </button>
        </div>
      </form>
    </dialog>
  )
}
