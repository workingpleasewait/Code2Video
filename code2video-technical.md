# code2video (technical baseline)

Scope
- Agents: Planner, Coder, Critic integrated in src/agent.py
- LLM clients: src/gpt_request.py (Gemini default; Claude retained; Ollama optional)
- Prompts: prompts/*.py (stage4: Critic JSON-contract strengthened)
- Runners:
  - run_single_infisical_dev.sh — single topic, secret injection, Gemini default
  - run_gemini_lowcost.sh — low-cost Gemini with minimal retries, critic on
- Output structure: src/CASES/{FOLDER_PREFIX}_{Model}/0-<topic>/

Pipeline (high level)
1) Outline generation (Gemini)
2) Storyboard generation (Gemini)
3) Manim code generation (Gemini); base class + grid helpers inserted
4) Debug/fix code (ScopeRefine)
5) Render sections (Manim)
6) Critic loop (Gemini, video+reference image)
7) Optional optimization renders and merge

Security/secrets
- Prefer macOS Keychain and Infisical; do not print secrets
- Env vars: GEMINI_API_KEY, GEMINI_MODEL; Anthropic (legacy) supported but not default

Local helpers
- code2video-gemini, code2video-lowcost shell functions (in ~/.zshrc)

Known issues
- Non-JSON outputs from LLMs mitigated with extract_json_from_markdown and conversion prompts
- Manim code sometimes needs multiple fix attempts (retry flags control cost)
