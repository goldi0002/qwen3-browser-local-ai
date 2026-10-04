export type ChatMessage = { role: 'user' | 'assistant'; content: string };

export function toChatMessages(messages: ChatMessage[]) {
  return messages.map((message) => ({
    role: message.role,
    content: message.content,
  }));
}

export function getStreamToken(chunk: {
  choices?: Array<{ delta?: { content?: string | null } }>;
}) {
  return chunk.choices?.[0]?.delta?.content ?? '';
}

export function estimateTokensPerSecond(text: string, elapsedMs: number) {
  if (elapsedMs <= 0) return null;
  const trimmed = text.trim();
  const estimatedTokens = trimmed ? trimmed.split(/\s+/).length : 0;
  return estimatedTokens / (elapsedMs / 1000);
}

export function formatError(prefix: string, error: unknown) {
  const message = error instanceof Error
    ? error.message.trim()
    : typeof error === 'string'
      ? error.trim()
      : '';

  if (message) return `${prefix}: ${message}`;
  return `${prefix}: the browser/WASM runtime aborted without a detailed error message.`;
}


export type WebGpuCapabilities = {
  api: boolean;
  adapter: boolean;
  shaderF16: boolean;
};

export function shouldUseWebGPU(
  capabilities: WebGpuCapabilities,
  wllamaSupportsGpu: boolean,
) {
  return capabilities.api &&
    capabilities.adapter &&
    capabilities.shaderF16 &&
    wllamaSupportsGpu;
}
