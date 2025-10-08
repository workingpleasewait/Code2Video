#!/usr/bin/env bash
set -euo pipefail
# Code2Video single-topic runner with Infisical (Dev)
# Usage:
#   /Users/mss/Code2Video/run_single_infisical_dev.sh --knowledge_point "Topic here" [extra agent args]
# If --knowledge_point is omitted, a default is used.

PY="/Users/mss/Code2Video/.venv/bin/python3"
AGENT="/Users/mss/Code2Video/src/agent.py"
PROJECT_ID="4ca8b424-7c26-4e1f-9802-bdb61aeb6771"
ENV_NAME="Dev"
export PROJECT_ID ENV_NAME

# Defaults
KP="Linear transformations and matrices"
EXTRA_ARGS=()

# Minimal flag parsing for knowledge point; pass through everything else
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

# Fetch secrets outside of Infisical run and execute directly (no nested shells)
# 1) Anthropic
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  ANTHROPIC_API_KEY="$(infisical secrets get ANTHROPIC_API_KEY --projectId "${PROJECT_ID}" --env "${ENV_NAME}" --plain --silent 2>/dev/null || true)"
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ] && command -v security >/dev/null 2>&1; then
  ANTHROPIC_API_KEY="$(security find-generic-password -s ANTHROPIC_API_KEY -w 2>/dev/null || true)"
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ERROR: ANTHROPIC_API_KEY not available from Infisical or Keychain. Aborting." >&2
  exit 1
fi

# 2) Gemini
if [ -z "${GEMINI_API_KEY:-}" ]; then
  GEMINI_API_KEY="$(infisical secrets get GEMINI_API_KEY --projectId "${PROJECT_ID}" --env "${ENV_NAME}" --plain --silent 2>/dev/null || true)"
fi
if [ -z "${GEMINI_API_KEY:-}" ] && [ -n "${GOOGLE_API_KEY:-}" ]; then GEMINI_API_KEY="$GOOGLE_API_KEY"; fi
if [ -z "${GEMINI_API_KEY:-}" ] && [ -n "${GOOGLE_AI_STUDIO_API_KEY:-}" ]; then GEMINI_API_KEY="$GOOGLE_AI_STUDIO_API_KEY"; fi
# Keychain fallback (secure, no printing)
if [ -z "${GEMINI_API_KEY:-}" ] && command -v security >/dev/null 2>&1; then
  GEMINI_API_KEY="$(security find-generic-password -s GEMINI_API_KEY -w 2>/dev/null || true)"
fi
# Last resort: local file fallback (plaintext). Consider removing after verification.
if [ -z "${GEMINI_API_KEY:-}" ] && [ -f "/Users/mss/Desktop/gemini.txt" ]; then GEMINI_API_KEY="$(tr -d "\r\n" < /Users/mss/Desktop/gemini.txt)"; fi

# 3) Export runtime env (no printing secrets)
export ANTHROPIC_API_KEY GEMINI_API_KEY
export CLAUDE_API_KEY="$ANTHROPIC_API_KEY"
export CLAUDE_MODEL="${CLAUDE_MODEL:-claude-3-opus-20240229}"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-pro-preview-05-06}"
export PATH="/Users/mss/Code2Video/.venv/bin:$PATH"
export PYTHONPATH="/Users/mss/Code2Video/src:/Users/mss/Code2Video:${PYTHONPATH:-}"

# 4) Execute agent directly
exec "${PY}" "${AGENT}" --API Gemini --folder_prefix TEST-single --use_feedback --use_assets --knowledge_point "${KP}" "${EXTRA_ARGS[@]}"
