# code2video (baseline)

Status (UTC): {{timestamp}}

Verified facts
- Default text generation: Gemini (Planner + Coder)
- Critic: Gemini (video + reference image)
- Helpers:
  - /Users/mss/Code2Video/run_single_infisical_dev.sh (now defaults to Gemini)
  - /Users/mss/Code2Video/run_gemini_lowcost.sh (low-cost preset: minimal retries, critic on)
- Convenience commands (load on new terminal):
  - code2video-gemini — force Gemini generation
  - code2video-lowcost — low-cost preset
- Rendering stack: Manim Community v0.19.0
- Example successful output (Gemini gen + critic):
  - /Users/mss/Code2Video/src/CASES/AllGemini-DEFAULT_Gemini/0-Re-anchoring_recommendations_coinbase-trading_Buy_Desk_overview/Re-anchoring_recommendations_coinbase-trading_Buy_Desk_overview.mp4

Policy
- Sync-with-README: README.md and these baselines must reflect current code state.
- External truths: use GitHub repo state + fresh API calls; record UTC timestamps.

Next
- Keep this file updated after functional changes.
- Expand technical/system baselines as features change.
