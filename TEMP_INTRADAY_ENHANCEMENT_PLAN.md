# Intraday Enhancement Work Plan (Temporary)

## Overview
Structured execution plan to align the intraday SAC + DSR pipeline with the documented requirements. The plan is ordered by dependency so each stage builds on a validated foundation. Remove this file once the implementation is complete.

---

## Phase 0 – Preparation & Baseline Validation
1. **Workspace hygiene**
   - Command: `git status -sb`
   - Purpose: Ensure we understand current diffs before starting.
2. **Smoke tests (current state)**
   - Run: `pytest Test Files/test_training_quick.py`
   - Goal: Capture baseline pass/fail before modifications.
3. **Artifacts backup check**
   - Verify intraday training artifacts under `backend/models/` and `backend/evaluation/walk_forward_results/` are versioned or disposable before regenerating outputs.

---

## Phase 1 – Walk-Forward Orchestrator Alignment
1. **Audit current generator**
   - File: `backend/training/walk_forward.py`
   - Action: Review `generate_walk_forward_windows` and current window logic.
2. **Define canonical window set**
   - Update `generate_walk_forward_windows` to handle explicit, documented sequences:
     - 2018-01-01 → 2020-12-31 train / 2021-01-01 → 2021-12-31 test
     - 2018-01-01 → 2021-12-31 train / 2022-01-01 → 2022-12-31 test
     - 2018-01-01 → 2022-12-31 train / 2023-01-01 → 2023-12-31 test
   - Provide configuration switch to fall back to auto mode.
3. **Update pipeline entry point**
   - File: `backend/training/walk_forward.py`
   - Modify `run_walk_forward_training_pipeline` to consume the canonical window set when `auto_generate=False` and no custom windows provided.
4. **Validation**
   - Unit-style run: call `python -m backend.training.walk_forward --dry-run` (add helper CLI if missing) to print the resolved windows without training.

---

## Phase 2 – Intraday Environment Enhancements
1. **Introduce slippage & closing logic**
   - File: `backend/environments/base_env.py`
     - Extend `_calculate_commission` signature or add `_calculate_transaction_costs` helper.
   - File: `backend/environments/intraday_env.py`
     - Add configurable slippage (bps or cents per share) applied in `_execute_action` flow via the base helper.
     - Enforce forced exit exactly at 15:45 New York: require `time_fraction` or timestamp column to determine cutoff instead of relying on `last step` heuristics.
2. **Commission & Financing**
   - Possibly new file: `backend/environments/slippage.py` (if abstraction needed) or extend existing commission resolver in `backend/training/commission.py` to add intraday-specific fields.
3. **Reward alignment**
   - File: `backend/environments/intraday_env.py`
     - Ensure `_calculate_reward` subtracts total transaction costs (commission + slippage + financing) before computing net return.
   - File: `backend/training/rewards/dsr_wrapper.py`
     - Confirm the wrapper receives the net reward; adjust interface if necessary.
4. **Validation**
   - Add targeted unit tests:
     - `backend/test_intraday_env.py` (new file) covering slippage math, forced exit time, and reward netting logic.
   - Run: `pytest backend/test_intraday_env.py`

---

## Phase 3 – Training Workflow Updates
1. **Training entry adjustments**
   - File: `backend/training/train.py`
     - Update `_train_intraday_agent` to pass new slippage parameters and rely on sequential session sampler for evaluation.
     - Ensure training sampler behavior matches requirement (randomized for training, sequential for evaluation, no replay of future days in live-like evaluation).
2. **Fine-Tune Pipeline**
   - New file: `backend/training/fine_tune.py`
     - Load latest SAC model, refill replay buffer with most recent N days, call `.learn(..., reset_num_timesteps=False)`.
     - Accept CLI args + API wrapper entry point.
   - File: `backend/api/main.py`
     - Add endpoint for fine-tune trigger (protected route).
3. **CLI/Script Support**
   - File: `backend/scripts/run_walk_forward.py` (new helper) to orchestrate the training/evaluation flow using new windows.
4. **Validation**
   - Smoke run: `python scripts/run_walk_forward.py --symbol TQQQ --dry-run`
   - Fine-tune dry run: `python -m backend.training.fine_tune --model-id <id> --days 30 --dry-run`
   - Full tests: `pytest backend/test_agent_manager_scheduler.py backend/test_live_scheduler_bridge.py`

---

## Phase 4 – Frontend & Config Exposure
1. **Preset updates**
   - File: `frontend/src/components/training/ConfigManager.jsx`
     - Add SAC intraday presets aligned with new defaults (slippage aware, forced exit windows, etc.).
2. **Tab training flow**
   - File: `frontend/src/components/training/TabTraining.jsx`
   - File: `frontend/src/hooks/useTrainingState.js`
     - Surface slippage/forced exit knobs where applicable; hide for incompatible agents.
3. **Validation**
   - UI lint/build: `npm --prefix frontend run lint` and `npm --prefix frontend run build`
   - Manual QA: load Training tab, switch to SAC + DSR preset, ensure form populates and API payload includes new fields.

---

## Phase 5 – Documentation & Operational Notes
1. **Docs**
   - File: `Docs/TRAINING_IMPLEMENTATION_PLAN.md`
   - File: `Docs/LIVE_STACK_RUNBOOK.md`
   - File: `WorkPlan.md`
     - Document new workflow, fine-tune cadence, walk-forward schedule, operational guardrails.
2. **Runbooks**
   - Update `LIVE_TRADING_IMPLEMENTATION.md` with fine-tune process and monitoring steps.
3. **Validation**
   - Markdown lint (if configured): `npm --prefix frontend run lint:md` or `markdownlint` CLI.

---

## Phase 6 – Regression & Acceptance
1. **Automated Tests**
   - Full suite: `pytest`
   - Frontend build: `npm --prefix frontend run build`
2. **Manual Acceptance Checklist**
   - Verify walk-forward JSON artifacts exist for each canonical window and include slippage metadata.
   - Confirm fine-tune route returns success payload (use Postman or curl against local API).
   - Inspect training logs to verify forced exit lines (`forced_exit=True`) occur at 15:45 entries.
3. **Performance sanity**
   - Re-run `Test Files/test_training_with_backtest.py` to ensure compatibility with existing regression tests.

---

## Phase 7 – Clean-Up & Delivery
1. **Artifacts pruning**
   - Remove temporary files (including this plan) after approval.
   - Clear transient training artifacts unless needed.
2. **Git workflow**
   - Command: `git status -sb`
   - Stage & commit logical batches (backend env, training workflow, frontend, docs).
   - Final verification: `pytest`; `npm --prefix frontend run build`.
   - Push to remote after user acceptance.
