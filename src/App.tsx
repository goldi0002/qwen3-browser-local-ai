import { useMemo, useRef, useState } from 'react';
import { Wllama } from '@wllama/wllama';
import wllamaWasmUrl from '@wllama/wllama/esm/wasm/wllama.wasm?url';
import {
  estimateTokensPerSecond,
  formatError,
  getStreamToken,
  toChatMessages,
  type ChatMessage,
} from './app-utils';

type Message = ChatMessage;

const CONFIG = { default: wllamaWasmUrl };
const MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024;
const GGUF_MAGIC = 'GGUF';

async function isGgufFile(file: File) {
  const bytes = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  return String.fromCharCode(...bytes) === GGUF_MAGIC;
}

export default function App() {
  const engine = useRef<Wllama | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('Choose a local Q8 GGUF model to begin.');
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [gpu, setGpu] = useState<boolean | null>(null);
  const [tokensPerSecond, setTokensPerSecond] = useState<number | null>(null);
  const [diagnostics, setDiagnostics] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const canChat = Boolean(engine.current) && !loading && !generating;
  const runtimeLabel = useMemo(
    () => gpu === true ? 'WebGPU available' : gpu === false ? 'WASM / CPU fallback' : 'Runtime not initialized',
    [gpu],
  );

  async function selectModel(file: File) {
    setModelFile(file);
    setStatus('Checking GGUF file…');
    setLoading(true);
    setTokensPerSecond(null);
    setDiagnostics(null);

    try {
      if (!file.name.toLowerCase().endsWith('.gguf')) {
        throw new Error('Please choose a .gguf model file.');
      }
      if (file.size === 0) {
        throw new Error('The selected GGUF file is empty.');
      }
      if (file.size > MAX_MODEL_BYTES) {
        throw new Error('This browser build supports GGUF files up to 2 GB.');
      }
      if (!(await isGgufFile(file))) {
        throw new Error('The selected file is not a valid GGUF file (missing GGUF header).');
      }

      try {
        await engine.current?.exit();
      } catch {
        // A partially initialized Wllama worker may already have aborted.
      }
      engine.current = null;

      const nextEngine = new Wllama(CONFIG, {
        allowOffline: true,
        suppressNativeLog: false,
        logger: {
          debug: () => {},
          log: () => {},
          warn: (...args) => console.warn(...args),
          error: (...args) => console.error(...args),
        },
      });

      const supportsGpu = nextEngine.isSupportWebGPU();
      setGpu(supportsGpu);
      setStatus('Loading locally… CPU/WASM mode');

      // Android/Chromium is deliberately CPU-only here. Wllama 3.8.x uses
      // n_gpu_layers: 0 to skip WebGPU device initialization entirely.
      // Conservative context/batch/sequence settings also reduce mobile RAM use.
      await nextEngine.loadModel([file], {
        n_ctx: 1024,
        n_batch: 16,
        n_ubatch: 16,
        n_parallel: 1,
        n_gpu_layers: 0,
        reasoning: false,
        warmup: false,
      });

      engine.current = nextEngine;
      setStatus(`Ready — WASM/CPU${supportsGpu ? ' (WebGPU available, disabled for stability)' : ''}`);
    } catch (error) {
      try {
        await engine.current?.exit();
      } catch {
        // Ignore cleanup failures after an aborted WASM worker.
      }
      engine.current = null;
      setGpu(null);

      const message = formatError('Load failed', error);
      const runtime = [
        `File: ${file.name} (${(file.size / 1024 / 1024).toFixed(0)} MB)`,
        `Browser: ${navigator.userAgent}`,
        `WebGPU API: ${'gpu' in navigator ? 'available' : 'unavailable'}`,
        `SharedArrayBuffer: ${typeof SharedArrayBuffer !== 'undefined' ? 'available' : 'unavailable'}`,
        `Cross-origin isolated: ${crossOriginIsolated ? 'yes' : 'no'}`,
        message,
      ].join('\n');

      setDiagnostics(runtime);
      setStatus(message);
    } finally {
      setLoading(false);
    }
  }

  async function sendMessage() {
    const prompt = input.trim();
    if (!prompt || !engine.current || generating) return;

    const next = [...messages, { role: 'user' as const, content: prompt }];
    setMessages([...next, { role: 'assistant', content: '' }]);
    setInput('');
    setGenerating(true);
    setStatus('Generating…');

    const started = performance.now();
    let generated = '';

    try {
      await engine.current.createChatCompletion({
        messages: toChatMessages(next),
        max_tokens: 256,
        temperature: 0.7,
        top_p: 0.9,
        stream: true,
        onData: (chunk) => {
          const token = getStreamToken(chunk);
          if (!token) return;
          generated += token;
          setMessages([...next, { role: 'assistant', content: generated }]);
        },
      });

      setTokensPerSecond(estimateTokensPerSecond(generated, performance.now() - started));
      setStatus('Ready');
    } catch (error) {
      const message = formatError('Generation failed', error);
      setMessages([...next, { role: 'assistant', content: message }]);
      setStatus('Generation failed.');
    } finally {
      setGenerating(false);
    }
  }

  async function unload() {
    try {
      await engine.current?.exit();
    } catch {
      // Ignore worker cleanup errors.
    }

    engine.current = null;
    setGpu(null);
    setModelFile(null);
    setMessages([]);
    setTokensPerSecond(null);
    setDiagnostics(null);
    setStatus('Model unloaded. Choose a GGUF file to load again.');

    if (fileInput.current) fileInput.current.value = '';
  }

  return <main className="app">
    <header className="hero">
      <div>
        <p className="eyebrow">100% local inference</p>
        <h1>Qwen3 Browser AI</h1>
        <p className="subtitle">Run Qwen3-0.6B Q8 GGUF directly on your Android browser.</p>
      </div>
      <div className="badge">{runtimeLabel}</div>
    </header>

    <section className="card setup">
      <div className="setup-copy">
        <h2>Local model</h2>
        <p>Select the GGUF file from your phone. It is passed directly to Wllama in the browser and is never uploaded by this app.</p>
        {modelFile && <div className="filename">{modelFile.name} · {(modelFile.size / 1024 / 1024).toFixed(0)} MB</div>}
      </div>
      <div className="actions">
        <button onClick={() => fileInput.current?.click()} disabled={loading || generating}>
          {loading ? 'Loading…' : 'Choose .gguf'}
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".gguf,application/octet-stream"
          hidden
          onChange={e => {
            const f = e.target.files?.[0];
            if (f) void selectModel(f);
          }}
        />
        <button className="secondary" onClick={() => void unload()} disabled={!engine.current || loading || generating}>
          Unload
        </button>
      </div>
    </section>

    <section className="card chat">
      <div className="chat-head">
        <div>
          <h2>Chat</h2>
          <span>{status}</span>
          {diagnostics && <pre className="diagnostics">{diagnostics}</pre>}
        </div>
        {tokensPerSecond !== null && <strong>{tokensPerSecond.toFixed(1)} tok/s</strong>}
      </div>

      <div className="messages" aria-live="polite">
        {messages.length === 0
          ? <div className="empty"><span>⚡</span><p>Load Qwen3 locally, then send a message to benchmark browser inference.</p></div>
          : messages.map((m, i) => (
            <div className={`message ${m.role}`} key={i}>
              <div className="role">{m.role === 'user' ? 'You' : 'Qwen3'}</div>
              <div className="bubble">{m.content || '…'}</div>
            </div>
          ))}
      </div>

      <div className="composer">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void sendMessage();
            }
          }}
          placeholder={canChat ? 'Ask Qwen3 anything…' : 'Load a GGUF model first…'}
          disabled={!canChat}
          rows={2}
        />
        <button onClick={() => void sendMessage()} disabled={!canChat || !input.trim()}>Send</button>
      </div>
    </section>

    <footer>
      <span>Wllama + llama.cpp · no inference server</span>
      <span>{'gpu' in navigator ? 'WebGPU detected (disabled for CPU baseline)' : 'WebGPU unavailable'}</span>
    </footer>
  </main>;
}
