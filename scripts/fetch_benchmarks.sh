#!/bin/bash
# Fetch the benchmark datasets used by --dataset gsm / gsm-full / math500.
set -e
mkdir -p data
curl -sL -o data/gsm8k_test.jsonl \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
curl -sL -o data/gsmhard.jsonl \
    https://huggingface.co/datasets/reasoning-machines/gsm-hard/resolve/main/gsmhardv2.jsonl
curl -sL -o data/math500.jsonl \
    https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl
wc -l data/*.jsonl
