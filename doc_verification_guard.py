import json, os, subprocess, sys, time

RESULTS = {
  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "verified_claims": [],
  "failed_claims": [],
  "truth_source": "GitHub + Local Repo",
  "notes": "bypass caches; prefer fresh data"
}
ROOT = os.getcwd()
PROJECT = "code2video"

def claim(name, ok, detail=""):
    (RESULTS["verified_claims"] if ok else RESULTS["failed_claims"]).append({"claim": name, "detail": detail})

# 1) Baselines exist
for fn in [f"{PROJECT}.md", f"{PROJECT}-technical.md"]:
    claim(f"baseline:{fn} exists", os.path.exists(os.path.join(ROOT, fn)))

# 2) README freshness vs code dirs

def latest_mtime(paths):
    mt = 0
    for p in paths:
        if not os.path.isdir(p):
            continue
        for d, _, fs in os.walk(p):
            for f in fs:
                try:
                    mt = max(mt, os.path.getmtime(os.path.join(d, f)))
                except FileNotFoundError:
                    pass
    return mt

key_dirs = [p for p in ["src", "cli", "scripts"] if os.path.isdir(p)]
if key_dirs:
    readme_m = os.path.getmtime("README.md") if os.path.exists("README.md") else 0
    code_m = latest_mtime(key_dirs)
    claim("readme_freshness", readme_m >= code_m, f"readme_m={readme_m}, code_m={code_m}")

# 3) CLI entrypoint present (best-effort)
claim("cli_entrypoint", any(os.path.exists(p) for p in ["cli/__init__.py", "cli.py", "Makefile"]))

# 4) Tests runnable (smoke; ignore failure if pytest missing)
try:
    rc = subprocess.call(["python", "-m", "pytest", "-q"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    claim("tests_runnable", rc == 0, f"pytest rc={rc}")
except Exception as e:
    claim("tests_runnable", False, str(e))

with open("doc_verification_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

sys.exit(0 if not RESULTS["failed_claims"] else 1)
