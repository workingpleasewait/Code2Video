#!/usr/bin/env sh
# POSIX-friendly root markdown whitelist guard
set -eu

WHITELIST="README.md CHANGELOG.md CONTRIBUTING.md LICENSE CODE_OF_CONDUCT.md \
project-setup.md code2video.md code2video-technical.md code2video-system.md \
code2video-roadmap.md WARP.md"

in_whitelist() {
  needle=$1
  for w in $WHITELIST; do
    [ "$w" = "$needle" ] && return 0
  done
  return 1
}

# Get staged files (added/copied/modified/renamed)
STAGED=$(git diff --cached --name-only --diff-filter=ACMR || true)

# Collect root-level .md among staged
OFF=""
printf "%s\n" "$STAGED" | awk -F/ 'NF==1 && $0 ~ /\.md$/' | while IFS= read -r f; do
  if in_whitelist "$f"; then
    :
  else
    if [ -f "$f" ] && grep -qi '^Doc-Placement-Override:[[:space:]]*root' "$f" 2>/dev/null; then
      :
    else
      echo "$f"
    fi
  fi
done | {
  # read all offending files
  read_one=false
  while IFS= read -r line; do
    if [ "$read_one" = false ]; then
      echo "❌ Non-whitelisted root .md files:" >&2
      read_one=true
    fi
    echo " - $line" >&2
    found=1
  done
  exit ${found:-0}
}
