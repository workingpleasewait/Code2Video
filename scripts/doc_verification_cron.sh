#!/usr/bin/env bash
set -e
PROJECT_ROOT="/Users/mss/Code2Video"
cd "$PROJECT_ROOT"
mkdir -p logs
LOG="logs/doc_verification.log"
if python doc_verification_guard.py >>"$LOG" 2>&1; then
  echo "✅ Verification passed"
  rm -f DOCUMENTATION_DRIFT_ALERT.json || true
else
  echo "❌ Drift detected"
  python - <<'PY'
import json, time
alert={
  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "status": "FAILED",
  "type": "documentation_drift"
}
open("DOCUMENTATION_DRIFT_ALERT.json","w").write(json.dumps(alert,indent=2))
PY
  exit 1
fi
