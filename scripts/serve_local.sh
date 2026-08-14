#!/bin/bash
# Example: serve a hybrid-thinking model with llama.cpp for actinf-pfc.
# --jinja is required for per-request thinking toggles (enable_thinking).
MODEL="${MODEL:-$HOME/models/Qwen3-4B-Q4_K_M.gguf}"
exec llama-server -m "$MODEL" \
    --jinja -fa 1 -ngl 99 --ctx-size 16384 --parallel 4 \
    --host 127.0.0.1 --port "${PORT:-8085}"
