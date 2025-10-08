#!/usr/bin/env bash
set -euo pipefail
WHITELIST=("README.md" "CHANGELOG.md" "CONTRIBUTING.md" "LICENSE" "CODE_OF_CONDUCT.md"
           "project-setup.md" "code2video.md" "code2video-technical.md"
           "code2video-system.md" "code2video-roadmap.md" "WARP.md")

in_array(){ local n=$1; shift; for x in "$@"; do [[ "$x" == "$n" ]] && return 0; done; return 1; }

mapfile -t STAGED < <(git diff --cached --name-only --diff-filter=ACMR)
mapfile -t ROOT_MD < <(printf '%s\n' "${STAGED[@]}" | grep -E '^[^/]+\.md$' || true)

OFF=()
for f in "${ROOT_MD[@]}"; do
  in_array "$f" "${WHITELIST[@]}" && continue
  [[ -f "$f" ]] && grep -qi '^Doc-Placement-Override:[[:space:]]*root' "$f" && continue
  OFF+=("$f")
done

if (( ${#OFF[@]} )); then
  echo "❌ Non-whitelisted root .md files:"; printf ' - %s\n' "${OFF[@]}"; exit 1
fi
