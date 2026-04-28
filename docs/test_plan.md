# CineBandit — Test Plan & Test Cases

**DA5402 · MLOps | Chirag · DA25M008**

---

## 1. Overview

This document defines the test strategy and individual test cases for the CineBandit system. All test cases are grounded in the commands and observed outputs from `verification.md`, which serves as the authoritative end-to-end verification record.

The full automated suite collected **102 tests, 102 passed, 0 failed** in 4.21 s (pytest run recorded in `verification.md` §5).

---

## 2. Environment

| Item | Value |
|---|---|
| Python | 3.12.13 (conda env `cinebandit`) |
| pytest | 9.0.3 |
| Key packages | numpy 2.4.2, pandas 2.3.3, fastapi 0.135.2, mlflow 2.22.0, streamlit 1.55.0, dvc 3.51.2 |
| Test data | MovieLens-100K — 100,000 ratings, 1,682 movies, 943 users |
| Docker stack | 8 containers via `docker compose` (`make up`) |

---

## 3. Test Scope

| Module | Test file | Cases |
|---|---|---|
| LinUCB bandit | `tests/test_bandit.py` | 24 |
| Data loading & features | `tests/test_data.py` | 18 |
| Drift detection | `tests/test_drift.py` | 38 |
| FastAPI endpoints | `tests/test_api.py` | 22 |
| **Total** | | **102** |

---

## 4. Code Coverage (from `verification.md` §5)

Coverage was measured with `pytest --cov=src` on the bandit, data, and drift test files.

| Module | Statements | Covered | Coverage |
|---|---|---|---|
| `src/bandit/linucb.py` | 66 | 65 | **98%** |
| `src/drift/detector.py` | 78 | 75 | **96%** |
| `src/data/loader.py` | 90 | 54 | 60% |
| `src/api/` (routes) | ~150 | 0* | 0% |

> *API routes are covered by `test_api.py` using `TestClient` but are excluded from this coverage run. Full API coverage is validated via the integration tests in §8.

---

## 5. Test Cases — LinUCB Bandit (`test_bandit.py`)

### 5.1 `TestLinUCBArm` (12 cases)

| ID | Test name | What it verifies |
|---|---|---|
| TB-01 | `test_initialisation_shape` | `A` matrix shape is `(d, d)` on a fresh arm |
| TB-02 | `test_initialisation_identity` | `A` initialises as identity matrix |
| TB-03 | `test_initialisation_zero_b` | `b` vector initialises as all zeros |
| TB-04 | `test_ucb_returns_scalar` | `ucb_score()` returns a single float |
| TB-05 | `test_ucb_non_negative_on_fresh_arm` | UCB score ≥ 0 before any updates |
| TB-06 | `test_update_increments_pulls` | `arm.pulls` increments by 1 after `update()` |
| TB-07 | `test_update_accumulates_reward` | `arm.total_reward` accumulates correctly |
| TB-08 | `test_empirical_ctr_zero_pulls` | `empirical_ctr` returns 0.0 when `pulls == 0` |
| TB-09 | `test_empirical_ctr_correct` | `empirical_ctr == total_reward / pulls` |
| TB-10 | `test_sherman_morrison_update` | `A_inv` changes after a rank-1 update without full inversion |
| TB-11 | `test_multiple_updates_consistency` | Repeated updates keep `A_inv` positive definite |
| TB-12 | `test_higher_alpha_higher_ucb` | Larger `alpha` produces larger UCB score on the same arm |

### 5.2 `TestLinUCB` (12 cases)

| ID | Test name | What it verifies |
|---|---|---|
| TB-13 | `test_initialisation` | Model initialises with empty arms dict and `timestep == 0` |
| TB-14 | `test_recommend_creates_arm` | Recommending for an unseen movie creates its arm lazily |
| TB-15 | `test_recommend_returns_valid_movie` | Returned `movie_id` is in the candidate set |
| TB-16 | `test_recommend_excludes_seen` | Movies in `exclude` list never appear in recommendations |
| TB-17 | `test_update_increments_timestep` | `model.timestep` increments on every `update()` call |
| TB-18 | `test_update_creates_arm` | Updating an unseen arm creates it lazily |
| TB-19 | `test_history_recorded` | Each recommendation is recorded in `model.history` |
| TB-20 | `test_exploitation_after_training` | After many positive updates, high-reward arm is ranked first |
| TB-21 | `test_set_alpha_updates_all_arms` | `set_alpha()` propagates to every existing arm |
| TB-22 | `test_get_stats_returns_dict` | `get_stats()` returns a dict with required keys |
| TB-23 | `test_recommend_single_candidate` | Works correctly when only one candidate arm exists |
| TB-24 | `test_recommend_empty_after_exclude` | Returns empty list when all candidates are excluded |

---

## 6. Test Cases — Data Loading (`test_data.py`)

### 6.1 `TestGenres` (3 cases)

| ID | Test name | What it verifies |
|---|---|---|
| TD-01 | `test_genres_length` | Genre list contains exactly 19 genres |
| TD-02 | `test_genres_no_duplicates` | No duplicate genre names |
| TD-03 | `test_known_genres_present` | "Drama", "Action", "Comedy" are all in the list |

### 6.2 `TestContextVector` (6 cases)

| ID | Test name | What it verifies |
|---|---|---|
| TD-04 | `test_context_vector_dimension` | Vector has exactly **60** dimensions (19 user + 19 movie + 19 interaction + 3 metadata) |
| TD-05 | `test_context_vector_dtype` | Vector dtype is `float64` |
| TD-06 | `test_context_vector_no_nan` | No NaN values in any position |
| TD-07 | `test_context_vector_changes_with_movie` | Different `movie_id` → different vector |
| TD-08 | `test_context_vector_changes_with_user_profile` | Different user profile → different vector |
| TD-09 | `test_interaction_term_is_elementwise_product` | Interaction sub-vector equals `user_pref * movie_genre` element-wise |

### 6.3 `TestUserProfiles` (4 cases)

| ID | Test name | What it verifies |
|---|---|---|
| TD-10 | `test_profiles_shape` | Profile matrix is `(942, 19)` |
| TD-11 | `test_profiles_normalised` | Each profile row sums to 1.0 (or 0.0 for users with no ratings) |
| TD-12 | `test_profiles_no_nan` | No NaN values |
| TD-13 | `test_profile_covers_all_users` | 942 profiles built for 943 users (1 user has no ratings) |

### 6.4 `TestDataLoading` (5 cases)

| ID | Test name | What it verifies |
|---|---|---|
| TD-14 | `test_load_ratings_shape` | Ratings DataFrame has expected columns |
| TD-15 | `test_load_ratings_count` | Exactly **100,000** rows loaded |
| TD-16 | `test_ratings_value_range` | Raw ratings are integers in range 1–5 |
| TD-17 | `test_reward_is_binary` | `reward` column contains only 0 and 1 (binarised at rating ≥ 4) |
| TD-18 | `test_load_movies_genre_vectors` | Each movie has a 19-element binary genre vector |

---

## 7. Test Cases — Drift Detection (`test_drift.py`)

### 7.1 `TestKLDivergence` (6 cases)

| ID | Test name | What it verifies |
|---|---|---|
| TDR-01 | `test_identical_distributions_zero` | KL(P \|\| P) == 0.0 |
| TDR-02 | `test_non_negative` | KL divergence is always ≥ 0 |
| TDR-03 | `test_asymmetric` | KL(P \|\| Q) ≠ KL(Q \|\| P) in general |
| TDR-04 | `test_larger_divergence_for_different_distributions` | More different distributions → higher score |
| TDR-05 | `test_handles_zeros_with_epsilon` | Zero bins are smoothed with epsilon; no division-by-zero |
| TDR-06 | `test_unit_vectors` | One-hot distributions produce finite, non-NaN divergence |

### 7.2 `TestGenreDistribution` (4 cases)

| ID | Test name | What it verifies |
|---|---|---|
| TDR-07 | `test_distribution_sums_to_one` | Genre distribution sums to 1.0 |
| TDR-08 | `test_distribution_non_negative` | All genre proportions ≥ 0 |
| TDR-09 | `test_distribution_length` | Distribution has exactly 19 entries (one per genre) |
| TDR-10 | `test_empty_liked_returns_uniform` | Empty liked-movie list → uniform distribution |

### 7.3 `TestBaselineComputation` (6 cases)

| ID | Test name | What it verifies |
|---|---|---|
| TDR-11 | `test_baseline_saved` | `compute_baseline()` writes `genre_baseline.json` to disk |
| TDR-12 | `test_baseline_has_required_keys` | Baseline JSON contains `genre_distribution`, `like_rate`, `n_ratings` |
| TDR-13 | `test_baseline_like_rate_valid` | `like_rate` is in range [0, 1]; expected value ~0.554 |
| TDR-14 | `test_baseline_n_ratings_correct` | `n_ratings == 100000` |
| TDR-15 | `test_load_baseline_returns_dict` | `load_baseline()` deserialises JSON to a dict |
| TDR-16 | `test_load_baseline_missing_returns_none` | Returns `None` when baseline file does not exist |

### 7.4 `TestDriftDetection` (6 cases)

| ID | Test name | What it verifies |
|---|---|---|
| TDR-17 | `test_no_drift_same_data` | Comparing baseline against itself returns `alert: False` |
| TDR-18 | `test_report_has_required_keys` | Report dict contains `drift_score`, `alert`, `threshold` |
| TDR-19 | `test_insufficient_samples_returns_no_alert` | Fewer than minimum samples → no alert raised |
| TDR-20 | `test_missing_baseline_returns_no_alert` | No baseline file → no alert raised |
| TDR-21 | `test_simulated_drift_detected` | Injected genre shift (flip 30% labels) raises `alert: True` |
| TDR-22 | `test_drift_score_non_negative` | Drift score is always ≥ 0 |

### 7.5 `TestDriftSimulation` (5 cases)

| ID | Test name | What it verifies |
|---|---|---|
| TDR-23 | `test_simulate_returns_dataframe` | Simulation returns a pandas DataFrame |
| TDR-24 | `test_simulate_has_required_columns` | DataFrame has `user_id`, `movie_id`, `reward`, `phase` columns |
| TDR-25 | `test_simulate_reward_binary` | `reward` column contains only 0 and 1 |
| TDR-26 | `test_simulate_phase_zero_no_drift` | Phase 0 rows match baseline genre distribution |
| TDR-27 | `test_simulate_reproducible` | Same random seed → identical output |

---

## 8. Test Cases — FastAPI (`test_api.py`)

### 8.1 `TestSystemEndpoints` (8 cases)

| ID | Test name | Expected outcome | Verified in verification.md |
|---|---|---|---|
| TA-01 | `test_root_returns_200` | `GET /` → HTTP 200 | §8 |
| TA-02 | `test_root_contains_service_name` | Response body mentions "CineBandit" | §8 |
| TA-03 | `test_health_returns_ok` | `GET /health` → `{"status":"ok"}` | §8 — `curl /health` output |
| TA-04 | `test_health_always_up` | `/health` returns 200 even if model not loaded | §8 |
| TA-05 | `test_drift_status_structure` | `GET /drift/status` returns `drift_score`, `threshold`, `alert` | §8 — `curl /drift/status` |
| TA-06 | `test_drift_status_default_no_alert` | Default state has `alert: false` | §8 — `"alert":false` |
| TA-07 | `test_snapshot_structure` | `GET /snapshot` returns `rolling_ctr`, `model_timestep`, `arms_seen`, `drift_score` | §8, §14 |
| TA-08 | `test_retrain_trigger` | `POST /retrain` returns success status | §8 |

### 8.2 `TestModelEndpoints` (4 cases)

| ID | Test name | Expected outcome | Verified in verification.md |
|---|---|---|---|
| TA-09 | `test_model_info_when_loaded` | `GET /model/info` → `algorithm`, `alpha`, `timesteps`, `arms_seen` | §8 — `curl /model/info` output |
| TA-10 | `test_model_info_503_when_not_loaded` | Returns HTTP 503 if model not in memory | §8 |
| TA-11 | `test_ready_503_when_not_loaded` | `GET /ready` → HTTP 503 before model loads | §8 |
| TA-12 | `test_model_reload_endpoint_exists` | `POST /model/reload` → HTTP 200 | §8 |

### 8.3 `TestRecommendEndpoint` (8 cases)

| ID | Test name | Expected outcome | Verified in verification.md |
|---|---|---|---|
| TA-13 | `test_recommend_503_without_model` | Returns HTTP 503 before model loads | §8 |
| TA-14 | `test_recommend_invalid_user` | Unknown `user_id` → HTTP 422 or graceful fallback | §8 |
| TA-15 | `test_recommend_invalid_payload` | Missing body → HTTP 422 | §8 |
| TA-16 | `test_recommend_n_out_of_range` | `n > 1682` → HTTP 422 | §8 |
| TA-17 | `test_recommend_n_zero` | `n == 0` → HTTP 422 | §8 |
| TA-18 | `test_recommend_response_structure` | Response contains `user_id`, `recommendations` list, `model_timestep` | §8 — `curl /recommend` output |
| TA-19 | `test_recommend_ucb_scores_positive` | All `ucb_score` values are > 0 | §8, §14 — scores observed: 0.7972, 0.7831 … |
| TA-20 | `test_recommend_exclude_works` | Movies in `exclude` list absent from response | §8 |

### 8.4 `TestFeedbackEndpoint` (9 cases)

| ID | Test name | Expected outcome | Verified in verification.md |
|---|---|---|---|
| TA-21 | `test_feedback_503_without_model` | Returns HTTP 503 before model loads | §8 |
| TA-22 | `test_feedback_invalid_reaction` | Unknown `reaction` value → HTTP 422 | §8 |
| TA-23 | `test_feedback_invalid_user` | Unknown `user_id` → HTTP 422 | §8 |
| TA-24 | `test_feedback_invalid_movie` | Unknown `movie_id` → HTTP 422 | §8 |
| TA-25 | `test_feedback_missing_fields` | Missing required field → HTTP 422 | §8 |
| TA-26 | `test_feedback_like_response` | `reaction: "like"` → `reward: 1.0`, HTTP 200 | §8 — `"reward":1.0` |
| TA-27 | `test_feedback_dislike_reward_zero` | `reaction: "dislike"` → `reward: 0.0` | §8 |
| TA-28 | `test_feedback_increments_timestep` | `model_timestep` in response is previous + 1 | §8 — 10822 → 10823; §14 — 10825 → 10826 |
| TA-29 | `test_all_reaction_types` | "like", "dislike", "skip" all return HTTP 200 | §8 |

### 8.5 `TestDriftUpdate` (3 cases)

| ID | Test name | Expected outcome | Verified in verification.md |
|---|---|---|---|
| TA-30 | `test_drift_update_changes_score` | Posting a drift update changes `drift_score` | §8, §11 — score 0.0 → 0.00239 after DAG run |
| TA-31 | `test_drift_alert_triggered_above_threshold` | Score > 0.3 → `alert: true` | §8 |
| TA-32 | `test_drift_no_alert_below_threshold` | Score < 0.3 → `alert: false` | §8 — `"alert":false` |

---

## 9. Integration & End-to-End Test Cases

These are not automated pytest cases but are verified manually via the commands in `verification.md`.

### 9.1 DVC Pipeline (§6)

| ID | Verification step | Expected outcome | Actual (from verification.md) |
|---|---|---|---|
| DVC-01 | `dvc status` | "Data and pipelines are up to date." | ✓ Confirmed |
| DVC-02 | `dvc dag` | Three-stage DAG: prepare → train → evaluate | ✓ Confirmed |
| DVC-03 | `dvc repro` (no changes) | All stages skipped/cached | ✓ "Stage 'X' didn't change, skipping" |
| DVC-04 | `dvc metrics show` | `test_ctr: 0.6334`, `train_ctr: 0.6828`, `arm_coverage: 1.0` | ✓ Confirmed |
| DVC-05 | Change `alpha: 0.1 → 0.5`, `dvc repro` | `prepare` skipped; `train` and `evaluate` re-run | ✓ "Stage 'prepare' didn't change" |
| DVC-06 | `dvc metrics diff` (alpha 0.1 vs 0.5) | `test_ctr` drops from 0.6334 → 0.5636 | ✓ Confirmed |

### 9.2 MLflow (§9)

| ID | Verification step | Expected outcome | Actual (from verification.md) |
|---|---|---|---|
| MLF-01 | `curl /health` on port 5000 | `OK` | ✓ Confirmed |
| MLF-02 | `make train-mlflow` | Run logged with `train_ctr=0.6828`, `test_ctr=0.6334` | ✓ Run ID `e724cd4ab981…` |
| MLF-03 | Model registry search | Model `CineBandit` present, `production` alias → v1 | ✓ Confirmed |
| MLF-04 | Production alias | Version 1, status `READY` | ✓ Confirmed |

### 9.3 Airflow (§10)

| ID | Verification step | Expected outcome | Actual (from verification.md) |
|---|---|---|---|
| AF-01 | `curl /health` on port 8080 | Scheduler `healthy`, metadatabase `healthy` | ✓ Confirmed |
| AF-02 | List DAGs via API | `cinebandit_drift_detection`, `cinebandit_ingestion`, `cinebandit_retraining` | ✓ All 3 present |
| AF-03 | Trigger `cinebandit_ingestion` | DAG run reaches `state: success` within 60 s | ✓ Confirmed |
| AF-04 | Baseline file written | `data/baselines/genre_baseline.json` exists with `n_ratings: 100000`, `like_rate: 0.55375` | ✓ Confirmed |
| AF-05 | Trigger `cinebandit_drift_detection` | DAG queued; drift score updated | ✓ Queued |
| AF-06 | Trigger `cinebandit_retraining` | DAG queued with `reason: manual_test` | ✓ Queued |

### 9.4 Prometheus (§11)

| ID | Verification step | Expected outcome | Actual (from verification.md) |
|---|---|---|---|
| PM-01 | `/-/ready` | "Prometheus Server is Ready." | ✓ Confirmed |
| PM-02 | Active targets | `cinebandit_api` target health `up`, no last error | ✓ Confirmed |
| PM-03 | `cinebandit_recommendation_requests_total` | Count = 2 after 2 `/recommend` calls | ✓ Confirmed |
| PM-04 | `cinebandit_feedback_total` by reaction | `like=2`, `dislike=1` | ✓ Confirmed |
| PM-05 | `cinebandit_rolling_ctr` | 0.6667 after 2 likes, 1 dislike | ✓ Confirmed |
| PM-06 | `cinebandit_drift_score` | 0.00239 after drift detection DAG | ✓ Confirmed |
| PM-07 | p95 latency histogram | 24.2 ms (target < 200 ms) | ✓ **8× under target** |

### 9.5 Grafana (§12)

| ID | Verification step | Expected outcome | Actual (from verification.md) |
|---|---|---|---|
| GR-01 | `GET /api/health` | `"database":"ok"`, version 10.4.0 | ✓ Confirmed |
| GR-02 | Prometheus datasource | Type `prometheus`, URL `http://prometheus:9090`, default: true | ✓ Confirmed |
| GR-03 | Dashboard search | `CineBandit — Live Monitoring` (UID `cinebandit-main`) present | ✓ Confirmed |
| GR-04 | Panel count | 11 panels provisioned | ✓ All 11 listed |
| GR-05 | Live Grafana query | `cinebandit_recommendation_requests_total` returns 2 frames | ✓ Confirmed |

### 9.6 Full End-to-End Flow (§14)

| Step | Action | Expected outcome | Actual (from verification.md) |
|---|---|---|---|
| E2E-01 | `POST /recommend` for user 42, n=5 | 5 movies with UCB scores, `model_timestep: 10825` | ✓ Top pick: American President (0.7972) |
| E2E-02 | `POST /feedback` — like movie 692 | `reward: 1.0`, `rolling_ctr: 0.75`, timestep → 10826 | ✓ Confirmed |
| E2E-03 | `POST /recommend` again for user 42 | New top pick reflecting updated preferences (Cinema Paradiso, 0.8982) | ✓ Recommendations changed |
| E2E-04 | `GET /snapshot` | `model_timestep: 10826`, `rolling_ctr: 0.75` | ✓ Confirmed |

---

## 10. Performance Thresholds

| Metric | Target | Measured (verification.md §11) | Status |
|---|---|---|---|
| p95 recommendation latency | < 200 ms | **24.2 ms** | ✓ Pass |
| Test CTR | ≥ 0.60 | **0.6334** | ✓ Pass |
| Train CTR | ≥ 0.65 | **0.6828** | ✓ Pass |
| Arm coverage | 100% | **100%** | ✓ Pass |
| Learning improvement (Q1 → Q4) | > 5% | **+10.9%** (0.5957 → 0.7054) | ✓ Pass |
| Unit test pass rate | 100% | **102/102** | ✓ Pass |

---

## 11. Summary Checklist (from `verification.md`)

- [x] Python environment with all packages active
- [x] MovieLens-100K loaded (100,000 ratings, 1,682 movies, 943 users)
- [x] LinUCB trained — train CTR ~0.683, test CTR ~0.633
- [x] All 102 unit tests passing, 0 failed
- [x] DVC pipeline: prepare → train → evaluate (all stages cached)
- [x] DVC metrics show `test_ctr` and `genre_diversity`
- [x] All 8 Docker containers running and healthy
- [x] `GET /health` → `{"status":"ok"}`
- [x] `GET /ready` → `model_loaded: true`, timestep 10822
- [x] `/recommend` returns movie titles with UCB scores
- [x] `/feedback` increments `model_timestep`
- [x] MLflow experiment visible at `localhost:5000`
- [x] CineBandit model registered — `production` alias → v1
- [x] All 3 Airflow DAGs present and listed
- [x] `cinebandit_ingestion` DAG succeeds; baseline JSON written
- [x] Prometheus scraping API target (`health: up`)
- [x] Grafana `CineBandit — Live Monitoring` dashboard with 11 panels provisioned
- [x] Streamlit frontend accessible at `localhost:8501`
- [x] End-to-end feedback loop updates model (timestep 10825 → 10826)