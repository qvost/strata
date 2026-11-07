#!/usr/bin/env bash
# Build the compiled two-stage traversal kernel.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-g++}"
"$CXX" -O3 -march=native -std=c++20 -shared -fPIC \
    "$ROOT/bench/fast_hybrid.cpp" \
    -o "$ROOT/bench/libfast_hybrid.so"
echo "built $ROOT/bench/libfast_hybrid.so"
