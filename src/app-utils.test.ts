import { describe, expect, it } from 'vitest';
import {
  estimateTokensPerSecond,
  formatError,
  getStreamToken,
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
    expect(getStreamToken({
      choices: [{ delta: { content: 'Hello' } }],
    })).toBe('Hello');
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

  it('uses a safe fallback for non-Error values', () => {
    expect(formatError('Load failed', 'aborted')).toBe('Load failed.');
  });
});
