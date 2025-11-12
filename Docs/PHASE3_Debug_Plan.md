# Phase 3 Walk-Forward Debug Plan

## Why we are doing this
The full walk-forward pipeline (multi-window SAC intraday training + evaluation) currently takes hours and has produced late-stage errors (e.g., agent alias mismatches, evaluation runtime issues). Running the entire flow every time slows iteration and obscures where a failure occurs. We need a fast, structured debug pass that mimics the live job, surfaces the failing stage quickly, and builds confidence before launching the full production-length run.

## Goals
- Reproduce the exact train → evaluate loop with minimal runtime (minutes instead of hours).
- Introduce clear stage checkpoints so we know which part fails (window prep, training, evaluation, artifact save, etc.).
- Capture evidence (logs/artifacts) for every step to narrow down root causes.
- Once green on the abbreviated flow, scale back up to the original configuration and confirm success end-to-end.

## Plan Overview
1. **Prepare short walk-forward inputs**
   - Create a dedicated window JSON (e.g., `tmp/mini_window.json`) with a narrow train span (≈ 3–4 weeks) and a tiny test span (≈ 2–3 days).
   - Copy the current SAC intraday config (or use defaults) and override `total_timesteps`, `learning_rate`, etc., to match the new conservative defaults but with capped training steps (≤ 25k) so a single run completes in ~5 minutes.

2. **Add stage checkpoints in the pipeline**
   - Update `run_walk_forward_training_pipeline` to print stage markers:
     - `[STAGE ✅] Window Setup` after window resolution.
     - `[STAGE ✅/❌] Training` after `train_agent` returns.
     - `[STAGE ✅/❌] Evaluation` after `run_walk_forward_evaluation` completes.
   - Ensure failures exit immediately with an explicit error message so the failing stage is obvious in logs.

3. **Run structured debug passes**
   - **3.1 Window validation:**
     - Command: `python backend/scripts/run_walk_forward.py --symbol TNA --windows tmp/mini_window.json --dry-run`
     - Confirms the JSON parses and windows align with available data.
   - **3.2 Training-only sanity check:**
     - Command: `python backend/scripts/run_walk_forward.py --symbol TNA --windows tmp/mini_window.json --quiet --output-dir backend/evaluation/debug --stochastic`
     - Expect: quick run, model saved, stage logs confirm `Training` success.
   - **3.3 Evaluation-only replay (if needed):**
     - If training succeeds but evaluation fails, rerun `run_walk_forward_evaluation` directly with the saved model to isolate evaluation logic:
       `python -m backend.training.walk_forward_eval --config <mini-config> --model-path <saved-model> --windows tmp/mini_window.json --quiet`

4. **Investigate failures immediately**
   - Use stage logs to identify where it failed.
   - For training issues: inspect `training_progress.json`, SB3 logs, and ensure the mini config was applied.
   - For evaluation issues: inspect generated artifacts under `backend/evaluation/debug`, confirm model metadata, check console for stack traces.

5. **Iterate until the short pipeline is green**
   - Apply fixes (data, config, or code) per failure.
   - Re-run Steps 3.1–3.2 until both `[STAGE ✅] Training` and `[STAGE ✅] Evaluation` appear and artifacts are emitted without exceptions.

6. **Scale back to production settings**
   - Replace the mini window with the actual `tmp/first_window.json` (or auto-generated windows).
   - Restore standard `total_timesteps` and other hyperparameters.
   - Run the full walk-forward CLI once more:
     `python backend/scripts/run_walk_forward.py --symbol TNA --windows tmp/first_window.json --output-dir backend/evaluation/walk_forward_results --quiet`
   - Verify: No errors, summary JSON generated, models saved with expected metadata.

### Mini-run Hyperparameter Variations
- Config files live in `tmp/mini_config_v*.json`; all use `tmp/mini_windows_multi.json` unless stated otherwise.
- Use separate output dirs per run to keep artifacts distinct.
- Suggested commands:
   - `python backend/scripts/run_walk_forward.py --symbol TNA --windows tmp/mini_windows_multi.json --config tmp/mini_config_v1_default.json --output-dir backend/evaluation/debug_v1 --quiet`
   - `python backend/scripts/run_walk_forward.py --symbol TNA --windows tmp/mini_windows_multi.json --config tmp/mini_config_v2_low_lr.json --output-dir backend/evaluation/debug_v2 --quiet`
   - `python backend/scripts/run_walk_forward.py --symbol TNA --windows tmp/mini_windows_multi.json --config tmp/mini_config_v3_low_entropy.json --output-dir backend/evaluation/debug_v3 --quiet`
   - `python backend/scripts/run_walk_forward.py --symbol TNA --windows tmp/mini_windows_multi.json --config tmp/mini_config_v4_high_gamma.json --output-dir backend/evaluation/debug_v4 --quiet`
   - `python backend/scripts/run_walk_forward.py --symbol TNA --windows tmp/mini_windows_multi.json --config tmp/mini_config_v5_return_reward.json --output-dir backend/evaluation/debug_v5 --quiet`
- After each run capture Sharpe/return from the summary JSON to compare.

## Validation Checklist
- [ ] Mini window dry-run completes without exceptions.
- [ ] Mini window training logs show `[STAGE ✅] Training` and produce a model file.
- [ ] Mini window evaluation logs show `[STAGE ✅] Evaluation` and artifact JSON/metrics.
- [ ] Any identified issues fixed and regression-checked with the mini run.
- [ ] Full window run completes successfully with final summary and no alias/runtime errors.
- [ ] Optional: run targeted pytest modules relevant to the changes once pipeline is green.

## Expected Outcome
Following this plan yields a fast feedback loop for diagnosing the pipeline. By the time we execute the full walk-forward schedule again, we’ll already have high confidence that each stage works, cutting down the risk of wasting hours only to discover an avoidable failure at the end.
