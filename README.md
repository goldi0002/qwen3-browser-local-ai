# Qwen3 Browser Local AI

A React + Vite experiment for running Qwen3-0.6B Q8 GGUF entirely in the browser with Wllama.

## What it does

- Select a GGUF file directly from an Android device.
- Loads the model locally with Wllama / llama.cpp WebAssembly.
- Uses WebGPU when Wllama reports support, otherwise uses WASM/CPU.
- Provides a conversational chat UI.
- Shows an approximate generated-tokens-per-second benchmark.
- Lets you unload the model and select another one.
- Does not send the selected GGUF or prompts to an inference server.

## Run locally

Run npm install, then npm run dev. Open the Vite URL and choose the Qwen3-0.6B Q8 GGUF file.

## GitHub Pages

The Pages workflow runs only from main. The feature branch is intended to be reviewed and merged by the repository owner.

Expected Pages URL after a successful deployment:

https://goldi0002.github.io/qwen3-browser-local-ai/

That URL is not considered live until GitHub Actions reports a successful Pages deployment.

## Benchmark caveat

The displayed tok/s value is intentionally approximate. It estimates tokens using whitespace-separated output and divides by generation time. It is useful for comparing browser configurations, not a precise llama.cpp tokenizer benchmark.
