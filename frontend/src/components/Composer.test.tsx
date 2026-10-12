import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { Composer } from './Composer'
it('does not send while Chinese IME composition is active', () => {
  const send = vi.fn()
  const draft = vi.fn()
  render(
    <Composer
      busy={false}
      disabled={false}
      onSend={send}
      onStop={vi.fn()}
      draft="中文问题"
      onDraftChange={draft}
    />,
  )
  const input = screen.getByLabelText('输入消息')
  fireEvent.compositionStart(input)
  fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 })
  expect(send).not.toHaveBeenCalled()
  fireEvent.compositionEnd(input)
  fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })
  expect(send).not.toHaveBeenCalled()
  fireEvent.keyDown(input, { key: 'Enter' })
  expect(send).toHaveBeenCalledWith('中文问题')
  expect(draft).toHaveBeenCalledWith('')
})
it('changes send to stop and prevents typing while generating', () => {
  const stop = vi.fn()
  render(
    <Composer
      busy
      disabled={false}
      onSend={vi.fn()}
      onStop={stop}
      draft=""
      onDraftChange={vi.fn()}
    />,
  )
  expect(screen.getByLabelText('输入消息')).toBeDisabled()
  fireEvent.click(screen.getByRole('button', { name: '停止生成' }))
  expect(stop).toHaveBeenCalledOnce()
})
