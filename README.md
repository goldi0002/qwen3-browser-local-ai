# Qwen3 Browser Local AI

A React + Vite app for running Qwen3-0.6B Q8 GGUF entirely in the browser with Wllama.

## What it does

- Select a GGUF file directly from an Android device.
- Loads the model locally with Wllama / llama.cpp WebAssembly.
- Uses a conservative CPU/WASM baseline on Android. WebGPU is detected but intentionally disabled for model loading until it is explicitly enabled as an optimization.
- Streams chat responses locally.
- Shows an approximate generated-tokens-per-second benchmark.
- Lets you unload the model and select another one.
- Does not send the selected GGUF or prompts to an inference server.

## Android stability settings

The app uses Wllama 3.8.1 and deliberately loads the model with:

- `n_gpu_layers: 0` — disables GPU inference and, in current Wllama, skips WebGPU device initialization.
- `n_ctx: 1024` — keeps the KV cache small enough for mobile browsers.
- `n_batch/n_ubatch: 16` — limits peak temporary memory during prompt processing.
- `n_parallel: 1` — avoids allocating multiple simultaneous KV-cache sequences.
- `reasoning: false` and `warmup: false` — reduces unnecessary startup/generation work for this small browser demo.

The selected file is also checked for the GGUF magic header before Wllama starts.

## Run locally

Run:

```bash
npm install
npm test
npm run build
npm run dev
```

Open the Vite URL and choose the Qwen3-0.6B Q8 GGUF file.

## GitHub Pages

The Pages workflow runs tests before the production build and deploys only from `main`.

Expected Pages URL:

https://goldi0002.github.io/qwen3-browser-local-ai/

If a phone still shows an older UI after a deployment, close the tab and reopen the Pages URL in a new tab. The app does not install a service worker, so there is no application-level offline cache to clear.

## Troubleshooting Android

If model loading fails:

1. Use the Qwen3-0.6B Q8 GGUF file (about 610 MB) rather than a larger model.
2. Close other heavy browser tabs/apps to free RAM.
3. Reload the Pages URL after a successful deployment.
4. If the app shows diagnostics, include the Browser, WebGPU, SharedArrayBuffer and Cross-origin-isolated lines when reporting the failure.

The browser still needs enough free memory to hold the GGUF and llama.cpp/WASM runtime. A browser/OS memory limit can prevent a model from loading even when the file itself is valid.

## Benchmark caveat

The displayed tok/s value is intentionally approximate. It estimates tokens using whitespace-separated output and divides by generation time. It is useful for comparing browser configurations, not a precise llama.cpp tokenizer benchmark.
