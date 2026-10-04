import { describe, expect, it } from 'vitest';
import {
  estimateTokensPerSecond,
  formatError,
  getStreamToken,
  shouldUseWebGPU,
  toChatMessages,
  type ChatMessage,
} from './app-utils';

describe('toChatMessages', () => {
  it('preserves roles and content for Wllama chat input', () => {
    const messages: ChatMessage[] = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' },
    ];

    expect(toChatMessages(messages)).toEqual(messages);
  });
});

describe('getStreamToken', () => {
  it('returns streamed text from a chunk', () => {
    expect(getStreamToken({ choices: [{ delta: { content: 'Hello' } }] })).toBe('Hello');
  });

  it('returns an empty string when a chunk has no content', () => {
    expect(getStreamToken({ choices: [] })).toBe('');
    expect(getStreamToken({ choices: [{ delta: {} }] })).toBe('');
  });
});

describe('estimateTokensPerSecond', () => {
  it('estimates whitespace-separated output tokens per second', () => {
    expect(estimateTokensPerSecond('one two three four', 2000)).toBe(2);
  });

  it('returns null for zero or negative elapsed time', () => {
    expect(estimateTokensPerSecond('one two', 0)).toBeNull();
    expect(estimateTokensPerSecond('one two', -1)).toBeNull();
  });

  it('returns zero for empty output with positive elapsed time', () => {
    expect(estimateTokensPerSecond('', 1000)).toBe(0);
  });
});

describe('formatError', () => {
  it('formats Error instances', () => {
    expect(formatError('Load failed', new Error('aborted'))).toBe('Load failed: aborted');
  });

  it('formats string errors', () => {
    expect(formatError('Load failed', 'aborted')).toBe('Load failed: aborted');
  });

  it('provides a useful fallback for empty abort errors', () => {
    expect(formatError('Load failed', new Error(''))).toContain('browser/WASM runtime aborted');
  });
});


describe('shouldUseWebGPU', () => {
  it('enables GPU inference only when the browser and Wllama are ready', () => {
    expect(shouldUseWebGPU(
      { api: true, adapter: true, shaderF16: true },
      true,
    )).toBe(true);
  });

  it('rejects WebGPU when shader-f16 is unavailable', () => {
    expect(shouldUseWebGPU(
      { api: true, adapter: true, shaderF16: false },
      true,
    )).toBe(false);
  });

  it('rejects WebGPU when Wllama cannot use it', () => {
    expect(shouldUseWebGPU(
      { api: true, adapter: true, shaderF16: true },
      false,
    )).toBe(false);
  });
});
