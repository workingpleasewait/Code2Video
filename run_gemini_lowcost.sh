#!/usr/bin/env bash
set -euo pipefail
# Low-cost Gemini runner for Code2Video
# Usage:
#   /Users/mss/Code2Video/run_gemini_lowcost.sh --knowledge_point "Topic" [extra args]

PY="/Users/mss/Code2Video/.venv/bin/python3"
AGENT="/Users/mss/Code2Video/src/agent.py"

# Defaults
KP="Linear transformations and matrices"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --knowledge_point)
      shift
      KP="${1:-$KP}"
      ;;
    *)
      EXTRA_ARGS+=("$1")
      ;;
  esac
  shift || true
done

# Secrets via env/Keychain (no printing)
if command -v security >/dev/null 2>&1; then
  : "${GEMINI_API_KEY:=$(security find-generic-password -s GEMINI_API_KEY -w 2>/dev/null || true)}"
fi
export GEMINI_API_KEY
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-pro-preview-05-06}"

# Runtime env
export PATH="/Users/mss/Code2Video/.venv/bin:$PATH"
export PYTHONPATH="/Users/mss/Code2Video/src:/Users/mss/Code2Video:${PYTHONPATH:-}"

# Low-cost flags
exec "${PY}" "${AGENT}" \
  --API Gemini \
  --folder_prefix TEST-gemini-lowcost \
  --use_feedback --feedback_rounds 1 \
  --max_fix_bug_tries 1 --max_regenerate_tries 1 --max_mllm_fix_bugs_tries 1 \
  --no_assets \
  --knowledge_point "${KP}" "${EXTRA_ARGS[@]}"
