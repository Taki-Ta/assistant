import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, vi } from 'vitest'
import { TextDecoder, TextEncoder } from 'node:util'
import { ReadableStream } from 'node:stream/web'
beforeEach(() => {
  vi.stubGlobal('TextEncoder', TextEncoder)
  vi.stubGlobal('TextDecoder', TextDecoder)
  vi.stubGlobal('ReadableStream', ReadableStream)
})
if (typeof HTMLDialogElement !== 'undefined') {
  HTMLDialogElement.prototype.showModal = function () {
    this.setAttribute('open', '')
  }
  HTMLDialogElement.prototype.close = function () {
    this.removeAttribute('open')
  }
}
afterEach(() => {
  cleanup()
  if (typeof sessionStorage !== 'undefined') sessionStorage.clear()
  vi.unstubAllGlobals()
})
