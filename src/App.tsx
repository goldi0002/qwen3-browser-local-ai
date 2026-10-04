import { useMemo, useRef, useState } from 'react';
import { Wllama } from '@wllama/wllama';
import wllamaWasmUrl from '@wllama/wllama/esm/wasm/wllama.wasm?url';
import {
  estimateTokensPerSecond,
  formatError,
  getStreamToken,
  shouldUseWebGPU,
  toChatMessages,
  type ChatMessage,
  type WebGpuCapabilities,
} from './app-utils';

type Message = ChatMessage;

const CONFIG = { default: wllamaWasmUrl };
const MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024;
const GGUF_MAGIC = 'GGUF';

// Qwen3-0.6B is small enough to offload completely on most WebGPU phones.
// Wllama treats 0 as an explicit CPU-only request, so never use it on the
// normal WebGPU path.
const GPU_LAYERS = 99999;
const MODEL_OPTIONS = {
  n_ctx: 1024,
  n_batch: 128,
  n_ubatch: 128,
  n_parallel: 1,
  reasoning: false,
  warmup: false,
};

async function isGgufFile(file: File) {
  const bytes = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  return String.fromCharCode(...bytes) === GGUF_MAGIC;
}

type WebGpuInfo = WebGpuCapabilities;

async function getWebGpuInfo(): Promise<WebGpuInfo> {
  const gpu = (navigator as Navigator & {
    gpu?: {
      requestAdapter: () => Promise<{ features: { has: (name: string) => boolean } } | null>;
    };
  }).gpu;

  if (!gpu) return { api: false, adapter: false, shaderF16: false };

  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { api: true, adapter: false, shaderF16: false };

    return {
      api: true,
      adapter: true,
      shaderF16: adapter.features.has('shader-f16'),
    };
  } catch {
    return { api: true, adapter: false, shaderF16: false };
  }
}

function createEngine() {
  return new Wllama(CONFIG, {
    allowOffline: true,
    suppressNativeLog: true,
    logger: {
      debug: () => {},
      log: () => {},
      warn: (...args) => console.warn(...args),
      error: (...args) => console.error(...args),
    },
  });
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
    () => gpu === true ? 'WebGPU acceleration' : gpu === false ? 'WASM / CPU fallback' : 'Runtime not initialized',
    [gpu],
  );

  async function cleanupEngine() {
    const current = engine.current;
    engine.current = null;

    if (current) {
      try {
        await current.exit();
      } catch {
        // A partially initialized worker may already have aborted.
      }
    }
  }

  async function loadWithBackend(file: File, useGpu: boolean) {
    const nextEngine = createEngine();
    const options = {
      ...MODEL_OPTIONS,
      n_gpu_layers: useGpu ? GPU_LAYERS : 0,
    };

    setStatus(useGpu ? 'Initializing WebGPU…' : 'Initializing WASM/CPU…');

    try {
      await nextEngine.loadModel([file], options);
      engine.current = nextEngine;
      return nextEngine;
    } catch (error) {
      try {
        await nextEngine.exit();
      } catch {
        // Ignore cleanup failures from an aborted worker.
      }
      throw error;
    }
  }

  async function selectModel(file: File) {
    setModelFile(file);
    setStatus('Checking GGUF file…');
    setLoading(true);
    setTokensPerSecond(null);
    setDiagnostics(null);
    setGpu(null);

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

      await cleanupEngine();

      const webGpu = await getWebGpuInfo();
      const wllamaProbe = createEngine();
      const wllamaSupportsGpu = wllamaProbe.isSupportWebGPU();
      await wllamaProbe.exit().catch(() => {});

      // Current Wllama WebGPU builds require shader-f16. Checking the adapter
      // first avoids a known Chromium abort on devices exposing WebGPU without
      // that feature.
      const shouldUseGpu = shouldUseWebGPU(webGpu, wllamaSupportsGpu);

      let activeEngine: Wllama | null = null;

      if (shouldUseGpu) {
        try {
          activeEngine = await loadWithBackend(file, true);
          setGpu(true);
          setStatus('Ready — WebGPU acceleration enabled');
        } catch (gpuError) {
          console.warn('WebGPU model load failed; falling back to CPU.', gpuError);
          await cleanupEngine();
          setStatus('WebGPU failed to initialize — falling back to WASM/CPU…');

          activeEngine = await loadWithBackend(file, false);
          setGpu(false);
          setStatus('Ready — WASM/CPU fallback');
        }
      } else {
        activeEngine = await loadWithBackend(file, false);
        setGpu(false);
        setStatus(
          webGpu.api
            ? 'Ready — WASM/CPU fallback (WebGPU adapter/shader-f16 unavailable)'
            : 'Ready — WASM/CPU fallback',
        );
      }

      if (!activeEngine) throw new Error('The inference engine could not be initialized.');
    } catch (error) {
      await cleanupEngine();
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
    let frame: number | null = null;

    const flush = () => {
      frame = null;
      setMessages([...next, { role: 'assistant', content: generated }]);
    };

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

          // Token callbacks can arrive much faster than React needs to render.
          // Coalesce them to animation frames so streaming does not become the
          // bottleneck on mobile browsers.
          if (frame === null) {
            frame = requestAnimationFrame(flush);
          }
        },
      });

      if (frame !== null) {
        cancelAnimationFrame(frame);
        frame = null;
      }
      setMessages([...next, { role: 'assistant', content: generated }]);
      setTokensPerSecond(estimateTokensPerSecond(generated, performance.now() - started));
      setStatus('Ready');
    } catch (error) {
      if (frame !== null) cancelAnimationFrame(frame);
      const message = formatError('Generation failed', error);
      setMessages([...next, { role: 'assistant', content: message }]);
      setStatus('Generation failed.');
    } finally {
      setGenerating(false);
    }
  }

  async function unload() {
    await cleanupEngine();
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
      <span>
        {'gpu' in navigator
          ? gpu === true
            ? 'WebGPU active'
            : gpu === false
              ? 'WebGPU detected · CPU fallback'
              : 'WebGPU detected'
          : 'WebGPU unavailable'}
      </span>
    </footer>
  </main>;
}
