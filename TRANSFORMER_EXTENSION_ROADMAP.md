# Transformer Feature-Learning Roadmap for YoopRL

This document formalizes the work required to evolve the current SAC + DSR intraday pipeline while keeping the existing baseline stable. Each phase can be executed independently, but the order matters; later phases assume the previous ones are complete and validated.

---

## Phase 0 – Baseline Readiness (Prerequisite)

Goal: Ensure the existing SAC + DSR pipeline is reproducible, normalized, and monitored before adding new representation learning.

1. **Repository hygiene**
   - Command: `git status -sb`
   - Confirm no unstaged diffs.
   - Ensure `python -m backend.scripts.run_walk_forward --dry-run` succeeds (window generation only).
2. **Data verification**
   - Rebuild intraday caches for all supported symbols (TNA, IWM, QQQ, SPY, UPRO, TQQQ) covering 2021-01-04 through latest trading session.
   - Run `scripts/tmp_dataset_check.py` (or equivalent) to print earliest/latest timestamps for each symbol.
   - Required files: `data/intraday/<SYMBOL>/15m/*.csv` (existing).
3. **Feature normalization audit**
   - File update: `backend/environments/intraday_env.py` – confirm observations pass through the normalizer for both training and evaluation.
   - File update: `backend/training/train.py` – enforce saving/loading of standardization parameters (`normalizer_TNA_SAC_INTRADAY_DSR.json`).
   - Add smoke test: `backend/tests/test_intraday_normalization.py` (new) to ensure normalized mean ~0/std ~1 for a sample batch.
4. **Logging & monitoring**
   - File update: `backend/training/train.py` – ensure TensorBoard logging is enabled by default and outputs to `backend/models/sac/tensorboard`.
   - Create helper script `scripts/analyze_training_logs.py` (new) to summarize actor/critic loss trends, entropy, and max drawdown from metadata JSONs.
5. **Baseline metrics capture**
   - Run `tmp/run_short_walkforward.py --mode long --stochastic-eval` once full pipeline is stable.
   - Archive outputs under `backend/evaluation/baseline_runs/<date>/` (new directory) with README summarizing Sharpe, total return, drawdown per window.

*Deliverable:* Stable baseline report stored in `backend/evaluation/baseline_runs/README.md` (new) including run configuration and metric snapshot.

---

## Phase 1 – Efficiency Upgrades on Current Architecture

Goal: Reduce training time/VRAM pressure on RTX 2060 while retaining the manual feature set.

1. **Mixed-precision training (AMP)**
   - File update: `backend/training/train.py`
     - Switch SAC model instantiation to `torch.cuda.amp.autocast` for forward passes and `GradScaler` for backward.
     - Respect CPU fallback (AMP only when CUDA available).
   - Add unit test `backend/tests/test_amp_guardrails.py` (new) to ensure fallback to FP32 on CPU.
2. **Gradient accumulation**
   - File update: `backend/training/train.py`
     - Introduce config field `gradient_accumulation_steps` (default 1) under `TrainingSettings` in `backend/config/training_config.py`.
     - Modify SAC training loop to accumulate gradients before optimizer step.
   - Update API payload documentation in `Docs/TRAINING_IMPLEMENTATION_PLAN.md` (existing).
3. **Feature enrichment (low-cost)**
   - Add VIX/volatility context to state vector:
     - Extend `backend/data_download/intraday_loader.py` to fetch or merge VIX CSVs (new helper `data/intraday/VIX/15m/`).
     - Update `backend/data_download/intraday_features.py` to compute `vix_trend` and `vix_volatility` features.
     - Update `FeatureConfig.for_agent('SAC_INTRADAY_DSR')` in `backend/config/training_config.py` to toggle new features by default.
   - Introduce regression test `backend/tests/test_feature_registry.py` (new) verifying presence/shape of VIX features.
4. **Documentation**
   - Update `Docs/ARCHITECTURE.md` with AMP + gradient accumulation flow.
   - Append “Efficiency checklist” section to `WorkPlan.md`.

*Deliverable:* Benchmark run replicating baseline windows with AMP + accumulation, demonstrating comparable or improved metrics + reduced wall-clock time.

---

## Phase 2 – Representation Learning Pilot (Pre-Transformer)

Goal: Introduce an intermediate automatic feature learner (1D CNN or small LSTM) to validate architecture changes before full Transformer.

1. **Model component design**
   - New module: `backend/models/feature_extractors/cnn_encoder.py`
     - Accepts tensor of shape `(batch, window_length, feature_dim)` and returns embedding vector (e.g., size 64).
     - Configurable window length via `TrainingSettings.sequence_length` (new field).
   - Update `FeatureConfig` to enable toggling between `manual`, `cnn`, `transformer` modes.
2. **Data loader changes**
   - File update: `backend/environments/intraday_env.py`
     - Modify observation builder to emit stacked OHLCV windows when encoder enabled.
     - Maintain backward compatibility via flag `use_sequence_encoder`.
3. **Integration with SAC**
   - File update: `backend/training/train.py`
     - When encoder enabled, wrap policy network input with encoder module before feeding to SAC policy.
     - Save encoder weights with model checkpoints (`model_manager` needs update to include `encoder_state_dict`).
   - Update `backend/models/model_manager.py` to load/save encoder artifacts.
4. **Testing**
   - Unit test `backend/tests/test_cnn_encoder.py` (new) verifying forward pass, gradient flow, and serialization.
   - Integration test `backend/tests/test_sequence_observation.py` (new) ensuring environment outputs correct tensor shapes.
5. **Evaluation**
   - Run `tmp/run_short_walkforward.py --sequence-encoder cnn --mode short` to compare against baseline metrics.
   - Document results in `backend/evaluation/experiments/sequence_encoder/<date>/report.md` (new path).

*Deliverable:* Experimental report showing performance impact of CNN encoder and decision on whether to advance to Transformer phase.

---

## Phase 3 – Transformer Feature Encoder (Experimental)

Goal: Replace manual feature extraction with Transformer-based embeddings while keeping SAC + DSR loop intact.

1. **Architecture design**
   - New file: `backend/models/feature_extractors/transformer_encoder.py`
     - Implement minimal Transformer encoder stack (e.g., 2 layers, 4 heads, embedding dim 64).
     - Support positional encoding (sinusoidal or learned).
     - Parameterize via `TransformerConfig` dataclass (new) defined in `backend/config/training_config.py`.
2. **Configuration plumbing**
   - Update `TrainingSettings` to include:
     - `sequence_length`
     - `transformer_hidden_dim`
     - `transformer_num_layers`
     - `transformer_num_heads`
   - Extend CLI/REST API to accept these fields (`backend/api/live_routes.py`, `frontend` config components if exposed).
3. **Data preparation enhancements**
   - Ensure `intraday_loader` can assemble contiguous OHLCV sequences of desired length; handle gaps via forward-fill or mask tensor.
   - Introduce padding/masking logic in encoder to handle partial windows at episode start.
4. **Training loop adjustments**
   - Integrate Transformer encoder similarly to CNN case, but ensure AMP/gradient accumulation remain compatible.
   - Apply dropout and layer norm to mitigate overfitting.
   - Optionally enable weight tying for output projection to reduce parameter count.
5. **Monitoring & safeguards**
   - Extend TensorBoard logging to include encoder gradient norms and attention entropy (`backend/training/train.py`).
   - Implement early stopping heuristic: if validation Sharpe deteriorates over N checkpoints, abort run (configure via `TrainingSettings.early_stop_window`).
6. **Evaluation plan**
   - First run: `tmp/run_short_walkforward.py --sequence-encoder transformer --mode short --sequence-length 64`.
   - Compare metrics vs. CNN and manual baseline; log in `backend/evaluation/experiments/transformer/<date>/report.md` with tables for Sharpe, Sortino, max DD, and training time.
7. **Documentation updates**
   - Update `Docs/ARCHITECTURE.md` with new data flow diagram.
   - Add section to `Docs/RL_System_Specification.md` describing encoder modes and configuration knobs.

*Deliverable:* Prototype transformer run with documented performance, resource usage, and known issues. Decision gate: proceed only if metrics beat baseline or provide meaningful insights.

---

## Phase 4 – Data Scaling & Regularization

Goal: Address data scarcity and overfitting risks highlighted in the roadmap.

1. **Historical data extension**
   - Integrate pre-2021 intraday data if licensing allows (e.g., the `NEWDATA/*.csv` source).
   - Build conversion script `scripts/convert_newdata_intraday.py` (new) to transform consolidated CSVs into per-session files with timezone metadata.
   - Update `backend/database/db_manager.py` ingestion to accept extended history.
2. **Augmentation strategies**
   - Implement bootstrapped sampling for training (`TrainingSettings.resample_probability`).
   - Add noise-based augmentation (jitter, dropout) to encoder inputs when training transformer to avoid memorization.
3. **Regularization components**
   - Apply weight decay to encoder-only parameters; expose via config.
   - Introduce dropout rate control for transformer/CNN layers.
   - Add validation splitter for walk-forward windows (`backend/training/walk_forward.py`) to hold out final 5% of training window for early-stopping metrics.
4. **Automated experiment tracking**
   - Set up experiment log `backend/evaluation/experiments/README.md` centralizing results (new file).
   - Optionally integrate with lightweight tracking tool (e.g., Weights & Biases) if approved; otherwise extend existing JSON outputs with config hashes.

*Deliverable:* Expanded dataset + documented regularization techniques showing improved generalization for transformer runs.

---

## Phase 5 – LLM Safety Layer Integration

Goal: Use LLMs as supervisory components without interfering with the 15-minute trading loop.

1. **Architecture definition**
   - New service module: `backend/monitoring/llm_supervisor.py`
     - Polls news APIs or processed feeds (existing `data_download/sentiment_service.py`) hourly.
     - Sends summary prompt to LLM provider (Gemini, Perplexity, etc.).
     - Classifies risk level (OK / Caution / Halt) based on keywords.
   - Extend `backend/monitoring/routes.py` to expose supervisor status via REST.
2. **Kill-switch plumbing**
   - Update `backend/execution/live_trader.py` to query supervisor state before entering a new episode/day; if status `Halt`, flatten positions and pause trading.
   - Add audit log entries in `backend/monitoring/service.py` for every supervisor decision.
3. **Post-mortem workflow**
   - Script `scripts/generate_daily_summary.py` (new) to compile PnL + LLM narrative for each session; output to `backend/evaluation/daily_reports/<date>.md`.
4. **Testing**
   - Mock tests for supervisor module in `backend/tests/test_llm_supervisor.py` verifying fallback when API fails and ensuring timeouts do not block core loop.
   - Integration test to simulate HALT signal and confirm live trader exits gracefully.
5. **Documentation**
   - Update `Docs/LIVE_STACK_RUNBOOK.md` with supervisor operational procedures and manual override instructions.

*Deliverable:* Supervisor service deployed in staging mode, proof that HALT signal stops trading without crashing SAC loop.

---

## Phase 6 – Acceptance & Rollout

1. **Regression sweep**
   - Run full walk-forward suites for manual baseline, CNN encoder, and transformer encoder.
   - Ensure `pytest` (backend) and `npm --prefix frontend run build` pass.
2. **Performance comparison**
   - Compile `Docs/Transformer_Experiment_Report.md` (new) summarizing:
     - Metrics per window
     - Training time vs. baseline
     - Resource usage (VRAM, CPU time)
     - Qualitative observations (attention patterns, overfitting signs)
3. **Go/No-Go decision**
   - Hold review meeting (document notes in `Docs/MEETING_NOTES_Transformer_Gate.md` – new) deciding whether to adopt transformer encoder in production preset (`get_sac_intraday_dsr_preset`).
   - If approved, update preset defaults; if not, log findings and maintain CNN/manual as default.
4. **Operational rollout**
   - Update `frontend/src/components/training/ConfigManager.jsx` to expose encoder mode toggle if transformer moves forward.
   - Publish release notes in `Docs/CHANGELOG.md`.

---

## Appendix – Required New Files Overview

```
backend/tests/test_intraday_normalization.py
backend/tests/test_amp_guardrails.py
backend/tests/test_feature_registry.py
backend/tests/test_cnn_encoder.py
backend/tests/test_sequence_observation.py
backend/tests/test_llm_supervisor.py
backend/models/feature_extractors/cnn_encoder.py
backend/models/feature_extractors/transformer_encoder.py
backend/monitoring/llm_supervisor.py
scripts/analyze_training_logs.py
scripts/convert_newdata_intraday.py
scripts/generate_daily_summary.py
backend/evaluation/baseline_runs/README.md
backend/evaluation/experiments/sequence_encoder/<date>/report.md
backend/evaluation/experiments/transformer/<date>/report.md
backend/evaluation/experiments/README.md
Docs/Transformer_Experiment_Report.md
Docs/MEETING_NOTES_Transformer_Gate.md
```

Additional files mentioned (e.g., reports with `<date>`) will be created per run.

---

**Next Steps**
1. Complete Phase 0 tasks and capture baseline metrics.
2. Implement Phase 1 efficiency upgrades and validate identical or better performance.
3. Proceed sequentially through Phases 2–5 only when preceding deliverables are approved.
4. Use Phase 6 checklist before updating production presets.
