# CineBandit — End-to-End Verification Guide

This document walks through every component of the system in order.
Run each command, paste the output into the placeholder, and share it for verification.

---

## 0. Prerequisites

```bash
# Verify conda environment is active
conda activate cinebandit
python --version
```
**Output:**
```
Python 3.12.13
```

```bash
# Verify all key packages are installed
python -c "import numpy, pandas, sklearn, fastapi, mlflow, streamlit, dvc; \
print('numpy:', numpy.__version__); \
print('pandas:', pandas.__version__); \
print('fastapi:', fastapi.__version__); \
print('mlflow:', mlflow.__version__); \
print('streamlit:', streamlit.__version__); \
print('dvc:', dvc.__version__)"
```
**Output:**
```
numpy: 2.4.2
pandas: 2.3.3
fastapi: 0.135.2
mlflow: 2.22.0
streamlit: 1.55.0
dvc: 3.51.2
```

---

## 1. Project Structure

```bash
# Verify folder structure
find . -type f -name "*.py" | grep -v __pycache__ | grep -v .dvc | sort
```
**Output:**
```
./dags/drift_detection_dag.py
./dags/ingestion_dag.py
./dags/retraining_dag.py
./evaluate.py
./src/__Init__.py
./src/api/__init__.py
./src/api/core/__Init__.py
./src/api/core/metrics.py
./src/api/core/model_store.py
./src/api/core/schemas.py
./src/api/main.py
./src/api/routes/__init__.py
./src/api/routes/feedback.py
./src/api/routes/recommend.py
./src/api/routes/system.py
./src/bandit/__init__.py
./src/bandit/linucb.py
./src/bandit/simulator.py
./src/data/__init__.py
./src/data/loader.py
./src/drift/__init__.py
./src/drift/detector.py
./src/evaluation/__init__.py
./src/evaluation/metrics.py
./src/frontend/app.py
./src/pipeline/evaluate_stage.py
./src/pipeline/prepare.py
./src/pipeline/train_stage.py
./tests/test_api.py
./tests/test_bandit.py
./tests/test_data.py
./tests/test_drift.py
./train.py
./train_mlflow.py
```

---

## 2. Data

```bash
# Verify MovieLens data is present
ls -lh data/raw/ml-100k/
```
**Output:**
```
total 16M
-rw-r--r-- 1 chirag chirag 6.6K Apr 17 17:11 README
-rw-r--r-- 1 chirag chirag  716 Apr 17 17:11 allbut.pl
-rw-r--r-- 1 chirag chirag  643 Apr 17 17:11 mku.sh
-rw-r--r-- 1 chirag chirag 1.9M Apr 17 17:11 u.data
-rw-r--r-- 1 chirag chirag  202 Apr 17 17:11 u.genre
-rw-r--r-- 1 chirag chirag   36 Apr 17 17:11 u.info
-rw-r--r-- 1 chirag chirag 231K Apr 17 17:11 u.item
-rw-r--r-- 1 chirag chirag  193 Apr 17 17:11 u.occupation
-rw-r--r-- 1 chirag chirag  23K Apr 17 17:11 u.user
-rw-r--r-- 1 chirag chirag 1.6M Apr 17 17:11 u1.base
-rw-r--r-- 1 chirag chirag 384K Apr 17 17:11 u1.test
-rw-r--r-- 1 chirag chirag 1.6M Apr 17 17:11 u2.base
-rw-r--r-- 1 chirag chirag 386K Apr 17 17:11 u2.test
-rw-r--r-- 1 chirag chirag 1.6M Apr 17 17:11 u3.base
-rw-r--r-- 1 chirag chirag 388K Apr 17 17:11 u3.test
-rw-r--r-- 1 chirag chirag 1.6M Apr 17 17:11 u4.base
-rw-r--r-- 1 chirag chirag 388K Apr 17 17:11 u4.test
-rw-r--r-- 1 chirag chirag 1.6M Apr 17 17:11 u5.base
-rw-r--r-- 1 chirag chirag 389K Apr 17 17:11 u5.test
-rw-r--r-- 1 chirag chirag 1.8M Apr 17 17:11 ua.base
-rw-r--r-- 1 chirag chirag 183K Apr 17 17:11 ua.test
-rw-r--r-- 1 chirag chirag 1.8M Apr 17 17:11 ub.base
-rw-r--r-- 1 chirag chirag 183K Apr 17 17:11 ub.test
```

```bash
# Verify data loads correctly
python -c "
from src.data.loader import prepare_dataset
ratings, movies, users, profiles = prepare_dataset()
print(f'Ratings : {len(ratings):,}')
print(f'Movies  : {len(movies):,}')
print(f'Users   : {len(users):,}')
print(f'Profiles: {len(profiles):,}')
print(f'Like rate: {ratings[\"reward\"].mean():.3f}')
"
```
**Output:**
```
[loader] Dataset already present at /home/chirag/MLOps/cinebandit/data/raw/ml-100k
[loader] Loading ratings ...
         100,000 ratings loaded
[loader] Loading movies ...
         1,682 movies loaded
[loader] Loading users ...
         943 users loaded
[loader] Building user genre profiles ...
         942 user profiles built
Ratings : 100,000
Movies  : 1,682
Users   : 943
Profiles: 942
Like rate: 0.554
```

---

## 3. LinUCB Model — Training

```bash
# Train the model (standard, without MLflow)
python train.py --alpha 0.1 --no-plot
```
**Output:**
```

=======================================================
  CineBandit — LinUCB Training
  alpha=0.1, pool_size=20, seed=42
=======================================================

[loader] Dataset already present at /home/chirag/MLOps/cinebandit/data/raw/ml-100k
[loader] Loading ratings ...
         100,000 ratings loaded
[loader] Loading movies ...
         1,682 movies loaded
[loader] Loading users ...
         943 users loaded
[loader] Building user genre profiles ...
         942 user profiles built

[main] Data loaded in 1.5s
[main] Train: 80,000 | Test: 20,000

[main] Running training simulation ...
Simulating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 80000/80000 [01:57<00:00, 680.94it/s, CTR=0.683, match_rate=1.000, matched=10500]

[sim] Total steps processed : 80,000
[sim] Replay matches        : 10,822 (13.5%)
[sim] Mean CTR              : 0.6828
[main] Training complete in 117.5s

[main] Running test simulation (frozen model) ...
Simulating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 20000/20000 [00:31<00:00, 634.95it/s, CTR=0.614, match_rate=1.000, matched=2000]

[sim] Total steps processed : 20,000
[sim] Replay matches        : 2,444 (12.2%)
[sim] Mean CTR              : 0.6334

--- TRAINING RESULTS ---

=======================================================
  LinUCB Evaluation Summary
=======================================================
  Matched steps (unbiased updates) : 10,822
  Total bandit timesteps           : 10,822
  Arms (movies) seen               : 1,682 / 1,682 (100.0%)

  Final CTR                        : 0.6828
  Cumulative reward                : 7389

  Genre diversity (entropy)        : 3.595 bits (max=4.25)
  Arm coverage                     : 100.0%

  Most pulled movie_id             : 50 (351 pulls)
=======================================================

--- TEST RESULTS ---

=======================================================
  LinUCB Evaluation Summary
=======================================================
  Matched steps (unbiased updates) : 2,444
  Total bandit timesteps           : 10,822
  Arms (movies) seen               : 1,682 / 1,682 (100.0%)

  Final CTR                        : 0.6334
  Cumulative reward                : 1548

  Genre diversity (entropy)        : 3.621 bits (max=4.25)
  Arm coverage                     : 100.0%

  Most pulled movie_id             : 50 (351 pulls)
=======================================================

[main] Model saved to outputs/linucb_alpha0.1.pkl
```

```bash
# Verify model file was saved
ls -lh outputs/
```
**Output:**
```
total 282M
-rw-r--r-- 1 chirag chirag  80K Mar 29 07:58 alpha_comparison.png
-rw-r--r-- 1 chirag chirag  94M Apr 20 11:09 linucb_alpha0.1.pkl
-rw-r--r-- 1 chirag chirag  94M Apr 17 17:32 linucb_alpha0.5.pkl
-rw-r--r-- 1 chirag chirag  94M Mar 29 07:54 linucb_alpha1.0.pkl
-rw-r--r-- 1 chirag chirag 175K Mar 29 08:00 test_results_alpha0.1.png
-rw-r--r-- 1 chirag chirag 165K Mar 29 07:54 test_results_alpha1.0.png
-rw-r--r-- 1 chirag chirag 199K Mar 29 08:00 train_results_alpha0.1.png
-rw-r--r-- 1 chirag chirag 163K Mar 29 07:54 train_results_alpha1.0.png
```

---

## 4. LinUCB Model — Evaluation

```bash
# Evaluate the saved model
python evaluate.py --model outputs/linucb_alpha0.1.pkl
```
**Output:**
```
[loader] Dataset already present at /home/chirag/MLOps/cinebandit/data/raw/ml-100k
[loader] Loading ratings ...
         100,000 ratings loaded
[loader] Loading movies ...
         1,682 movies loaded
[loader] Loading users ...
         943 users loaded
[loader] Building user genre profiles ...
         942 user profiles built
[eval] Test set: 20,000 interactions
[eval] Loading model from outputs/linucb_alpha0.1.pkl
[eval] Model loaded: alpha=0.1, arms=1682, timesteps=10822

[eval] Running test simulation ...
Simulating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 20000/20000 [00:29<00:00, 687.07it/s, CTR=0.614, match_rate=1.000, matched=2000]

[sim] Total steps processed : 20,000
[sim] Replay matches        : 2,444 (12.2%)
[sim] Mean CTR              : 0.6334

=======================================================
  LinUCB Evaluation Summary
=======================================================
  Matched steps (unbiased updates) : 2,444
  Total bandit timesteps           : 10,822
  Arms (movies) seen               : 1,682 / 1,682 (100.0%)

  Final CTR                        : 0.6334
  Cumulative reward                : 1548

  Genre diversity (entropy)        : 3.621 bits (max=4.25)
  Arm coverage                     : 100.0%

  Most pulled movie_id             : 50 (351 pulls)
=======================================================


───────────────────────────────────────────────────────
  Detailed Breakdown
───────────────────────────────────────────────────────

  CTR by time quartile (learning curve):
    Q1 (steps 1–611): CTR = 0.5957
    Q2 (steps 612–1,222): CTR = 0.6056
    Q3 (steps 1,223–1,833): CTR = 0.6268
    Q4 (steps 1,834–2,444): CTR = 0.7054

  Top-10 movies by CTR (min 3 pulls):
  Title                           Pulls    CTR  Rewards
  ────────────────────────────── ────── ────── ────────
  Primal Fear (1996)                  3  1.000        3
  Lone Star (1996)                    3  1.000        3
  Reservoir Dogs (1992)               4  1.000        4
  To Catch a Thief (1955)             5  1.000        5
  Close Shave, A (1995)              15  0.867       13
  Boot, Das (1981)                   43  0.860       37
  Godfather, The (1972)              63  0.857       54
  Crimson Tide (1995)                20  0.850       17
  Apt Pupil (1998)                   26  0.846       22
  As Good As It Gets (1997)          56  0.839       47

  Genre distribution of recommendations:
  Drama           █████                           17.7%
  Action          ████                            14.1%
  Thriller        ███                             11.4%
  Romance         ███                             10.4%
  Comedy          ██                               7.0%
  Adventure       █                                6.5%
  Crime           █                                6.1%
  Sci-Fi          █                                6.1%

  Learning improvement:
    Early CTR (first half) : 0.6007
    Late CTR  (second half): 0.6661
    Improvement            : +10.9%
───────────────────────────────────────────────────────

[eval] Plot saved to outputs/eval_results_alpha0.1.png
```

---

## 5. Unit Tests

```bash
# Run all tests
pytest tests/ -v --tb=short 2>&1
```
**Output:**
```
========================================================================================== test session starts ==========================================================================================
platform linux -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0 -- /home/chirag/miniconda3/envs/cinebandit/bin/python3.12
cachedir: .pytest_cache
rootdir: /home/chirag/MLOps/cinebandit
configfile: pytest.ini
plugins: anyio-4.13.0, hydra-core-1.3.2
collected 102 items                                                                                                                                                                                     

tests/test_api.py::TestSystemEndpoints::test_root_returns_200 PASSED                                                                                                                              [  0%]
tests/test_api.py::TestSystemEndpoints::test_root_contains_service_name PASSED                                                                                                                    [  1%]
tests/test_api.py::TestSystemEndpoints::test_health_returns_ok PASSED                                                                                                                             [  2%]
tests/test_api.py::TestSystemEndpoints::test_health_always_up PASSED                                                                                                                              [  3%]
tests/test_api.py::TestSystemEndpoints::test_drift_status_structure PASSED                                                                                                                        [  4%]
tests/test_api.py::TestSystemEndpoints::test_drift_status_default_no_alert PASSED                                                                                                                 [  5%]
tests/test_api.py::TestSystemEndpoints::test_snapshot_structure PASSED                                                                                                                            [  6%]
tests/test_api.py::TestSystemEndpoints::test_retrain_trigger PASSED                                                                                                                               [  7%]
tests/test_api.py::TestModelEndpoints::test_model_info_when_loaded PASSED                                                                                                                         [  8%]
tests/test_api.py::TestModelEndpoints::test_model_info_503_when_not_loaded FAILED                                                                                                                 [  9%]
tests/test_api.py::TestModelEndpoints::test_ready_503_when_not_loaded FAILED                                                                                                                      [ 10%]
tests/test_api.py::TestModelEndpoints::test_model_reload_endpoint_exists PASSED                                                                                                                   [ 11%]
tests/test_api.py::TestRecommendEndpoint::test_recommend_503_without_model FAILED                                                                                                                 [ 12%]
tests/test_api.py::TestRecommendEndpoint::test_recommend_invalid_user PASSED                                                                                                                      [ 13%]
tests/test_api.py::TestRecommendEndpoint::test_recommend_invalid_payload PASSED                                                                                                                   [ 14%]
tests/test_api.py::TestRecommendEndpoint::test_recommend_n_out_of_range PASSED                                                                                                                    [ 15%]
tests/test_api.py::TestRecommendEndpoint::test_recommend_n_zero PASSED                                                                                                                            [ 16%]
tests/test_api.py::TestRecommendEndpoint::test_recommend_response_structure PASSED                                                                                                                [ 17%]
tests/test_api.py::TestRecommendEndpoint::test_recommend_ucb_scores_positive PASSED                                                                                                               [ 18%]
tests/test_api.py::TestRecommendEndpoint::test_recommend_exclude_works PASSED                                                                                                                     [ 19%]
tests/test_api.py::TestFeedbackEndpoint::test_feedback_503_without_model FAILED                                                                                                                   [ 20%]
tests/test_api.py::TestFeedbackEndpoint::test_feedback_invalid_reaction PASSED                                                                                                                    [ 21%]
tests/test_api.py::TestFeedbackEndpoint::test_feedback_invalid_user PASSED                                                                                                                        [ 22%]
tests/test_api.py::TestFeedbackEndpoint::test_feedback_invalid_movie PASSED                                                                                                                       [ 23%]
tests/test_api.py::TestFeedbackEndpoint::test_feedback_missing_fields PASSED                                                                                                                      [ 24%]
tests/test_api.py::TestFeedbackEndpoint::test_feedback_like_response PASSED                                                                                                                       [ 25%]
tests/test_api.py::TestFeedbackEndpoint::test_feedback_dislike_reward_zero PASSED                                                                                                                 [ 26%]
tests/test_api.py::TestFeedbackEndpoint::test_feedback_increments_timestep PASSED                                                                                                                 [ 27%]
tests/test_api.py::TestFeedbackEndpoint::test_all_reaction_types PASSED                                                                                                                           [ 28%]
tests/test_api.py::TestDriftUpdate::test_drift_update_changes_score PASSED                                                                                                                        [ 29%]
tests/test_api.py::TestDriftUpdate::test_drift_alert_triggered_above_threshold PASSED                                                                                                             [ 30%]
tests/test_api.py::TestDriftUpdate::test_drift_no_alert_below_threshold PASSED                                                                                                                    [ 31%]
tests/test_bandit.py::TestLinUCBArm::test_initialisation_shape PASSED                                                                                                                             [ 32%]
tests/test_bandit.py::TestLinUCBArm::test_initialisation_identity PASSED                                                                                                                          [ 33%]
tests/test_bandit.py::TestLinUCBArm::test_initialisation_zero_b PASSED                                                                                                                            [ 34%]
tests/test_bandit.py::TestLinUCBArm::test_ucb_returns_scalar PASSED                                                                                                                               [ 35%]
tests/test_bandit.py::TestLinUCBArm::test_ucb_non_negative_on_fresh_arm PASSED                                                                                                                    [ 36%]
tests/test_bandit.py::TestLinUCBArm::test_update_increments_pulls PASSED                                                                                                                          [ 37%]
tests/test_bandit.py::TestLinUCBArm::test_update_accumulates_reward PASSED                                                                                                                        [ 38%]
tests/test_bandit.py::TestLinUCBArm::test_empirical_ctr_zero_pulls PASSED                                                                                                                         [ 39%]
tests/test_bandit.py::TestLinUCBArm::test_empirical_ctr_correct PASSED                                                                                                                            [ 40%]
tests/test_bandit.py::TestLinUCBArm::test_sherman_morrison_update PASSED                                                                                                                          [ 41%]
tests/test_bandit.py::TestLinUCBArm::test_multiple_updates_consistency PASSED                                                                                                                     [ 42%]
tests/test_bandit.py::TestLinUCBArm::test_higher_alpha_higher_ucb PASSED                                                                                                                          [ 43%]
tests/test_bandit.py::TestLinUCB::test_initialisation PASSED                                                                                                                                      [ 44%]
tests/test_bandit.py::TestLinUCB::test_recommend_creates_arm PASSED                                                                                                                               [ 45%]
tests/test_bandit.py::TestLinUCB::test_recommend_returns_valid_movie PASSED                                                                                                                       [ 46%]
tests/test_bandit.py::TestLinUCB::test_recommend_excludes_seen PASSED                                                                                                                             [ 47%]
tests/test_bandit.py::TestLinUCB::test_update_increments_timestep PASSED                                                                                                                          [ 48%]
tests/test_bandit.py::TestLinUCB::test_update_creates_arm PASSED                                                                                                                                  [ 49%]
tests/test_bandit.py::TestLinUCB::test_history_recorded PASSED                                                                                                                                    [ 50%]
tests/test_bandit.py::TestLinUCB::test_exploitation_after_training PASSED                                                                                                                         [ 50%]
tests/test_bandit.py::TestLinUCB::test_set_alpha_updates_all_arms PASSED                                                                                                                          [ 51%]
tests/test_bandit.py::TestLinUCB::test_get_stats_returns_dict PASSED                                                                                                                              [ 52%]
tests/test_bandit.py::TestLinUCB::test_recommend_single_candidate PASSED                                                                                                                          [ 53%]
tests/test_bandit.py::TestLinUCB::test_recommend_empty_after_exclude PASSED                                                                                                                       [ 54%]
tests/test_data.py::TestGenres::test_genres_length PASSED                                                                                                                                         [ 55%]
tests/test_data.py::TestGenres::test_genres_no_duplicates PASSED                                                                                                                                  [ 56%]
tests/test_data.py::TestGenres::test_known_genres_present PASSED                                                                                                                                  [ 57%]
tests/test_data.py::TestContextVector::test_context_vector_dimension PASSED                                                                                                                       [ 58%]
tests/test_data.py::TestContextVector::test_context_vector_dtype PASSED                                                                                                                           [ 59%]
tests/test_data.py::TestContextVector::test_context_vector_no_nan PASSED                                                                                                                          [ 60%]
tests/test_data.py::TestContextVector::test_context_vector_changes_with_movie PASSED                                                                                                              [ 61%]
tests/test_data.py::TestContextVector::test_context_vector_changes_with_user_profile PASSED                                                                                                       [ 62%]
tests/test_data.py::TestContextVector::test_interaction_term_is_elementwise_product PASSED                                                                                                        [ 63%]
tests/test_data.py::TestUserProfiles::test_profiles_shape PASSED                                                                                                                                  [ 64%]
tests/test_data.py::TestUserProfiles::test_profiles_normalised PASSED                                                                                                                             [ 65%]
tests/test_data.py::TestUserProfiles::test_profiles_no_nan PASSED                                                                                                                                 [ 66%]
tests/test_data.py::TestUserProfiles::test_profile_covers_all_users PASSED                                                                                                                        [ 67%]
tests/test_data.py::TestDataLoading::test_load_ratings_shape PASSED                                                                                                                               [ 68%]
tests/test_data.py::TestDataLoading::test_load_ratings_count PASSED                                                                                                                               [ 69%]
tests/test_data.py::TestDataLoading::test_ratings_value_range PASSED                                                                                                                              [ 70%]
tests/test_data.py::TestDataLoading::test_reward_is_binary PASSED                                                                                                                                 [ 71%]
tests/test_data.py::TestDataLoading::test_load_movies_genre_vectors PASSED                                                                                                                        [ 72%]
tests/test_data.py::TestDataLoading::test_load_users_normalised_age PASSED                                                                                                                        [ 73%]
tests/test_drift.py::TestKLDivergence::test_identical_distributions_zero PASSED                                                                                                                   [ 74%]
tests/test_drift.py::TestKLDivergence::test_non_negative PASSED                                                                                                                                   [ 75%]
tests/test_drift.py::TestKLDivergence::test_asymmetric PASSED                                                                                                                                     [ 76%]
tests/test_drift.py::TestKLDivergence::test_larger_divergence_for_different_distributions PASSED                                                                                                  [ 77%]
tests/test_drift.py::TestKLDivergence::test_handles_zeros_with_epsilon PASSED                                                                                                                     [ 78%]
tests/test_drift.py::TestKLDivergence::test_unit_vectors PASSED                                                                                                                                   [ 79%]
tests/test_drift.py::TestGenreDistribution::test_distribution_sums_to_one PASSED                                                                                                                  [ 80%]
tests/test_drift.py::TestGenreDistribution::test_distribution_non_negative PASSED                                                                                                                 [ 81%]
tests/test_drift.py::TestGenreDistribution::test_distribution_length PASSED                                                                                                                       [ 82%]
tests/test_drift.py::TestGenreDistribution::test_empty_liked_returns_uniform PASSED                                                                                                               [ 83%]
tests/test_drift.py::TestBaselineComputation::test_baseline_saved PASSED                                                                                                                          [ 84%]
tests/test_drift.py::TestBaselineComputation::test_baseline_has_required_keys PASSED                                                                                                              [ 85%]
tests/test_drift.py::TestBaselineComputation::test_baseline_like_rate_valid PASSED                                                                                                                [ 86%]
tests/test_drift.py::TestBaselineComputation::test_baseline_n_ratings_correct PASSED                                                                                                              [ 87%]
tests/test_drift.py::TestBaselineComputation::test_load_baseline_returns_dict PASSED                                                                                                              [ 88%]
tests/test_drift.py::TestBaselineComputation::test_load_baseline_missing_returns_none PASSED                                                                                                      [ 89%]
tests/test_drift.py::TestDriftDetection::test_no_drift_same_data PASSED                                                                                                                           [ 90%]
tests/test_drift.py::TestDriftDetection::test_report_has_required_keys PASSED                                                                                                                     [ 91%]
tests/test_drift.py::TestDriftDetection::test_insufficient_samples_returns_no_alert PASSED                                                                                                        [ 92%]
tests/test_drift.py::TestDriftDetection::test_missing_baseline_returns_no_alert PASSED                                                                                                            [ 93%]
tests/test_drift.py::TestDriftDetection::test_simulated_drift_detected PASSED                                                                                                                     [ 94%]
tests/test_drift.py::TestDriftDetection::test_drift_score_non_negative PASSED                                                                                                                     [ 95%]
tests/test_drift.py::TestDriftSimulation::test_simulate_returns_dataframe PASSED                                                                                                                  [ 96%]
tests/test_drift.py::TestDriftSimulation::test_simulate_has_required_columns PASSED                                                                                                               [ 97%]
tests/test_drift.py::TestDriftSimulation::test_simulate_reward_binary PASSED                                                                                                                      [ 98%]
tests/test_drift.py::TestDriftSimulation::test_simulate_phase_zero_no_drift PASSED                                                                                                                [ 99%]
tests/test_drift.py::TestDriftSimulation::test_simulate_reproducible PASSED                                                                                                                       [100%]

=============================================================================================== FAILURES ================================================================================================
________________________________________________________________________ TestModelEndpoints.test_model_info_503_when_not_loaded _________________________________________________________________________
tests/test_api.py:124: in test_model_info_503_when_not_loaded
    assert r.status_code == 503
E   assert 200 == 503
E    +  where 200 = <Response [200 OK]>.status_code
___________________________________________________________________________ TestModelEndpoints.test_ready_503_when_not_loaded ___________________________________________________________________________
tests/test_api.py:128: in test_ready_503_when_not_loaded
    assert r.status_code == 503
E   assert 200 == 503
E    +  where 200 = <Response [200 OK]>.status_code
________________________________________________________________________ TestRecommendEndpoint.test_recommend_503_without_model _________________________________________________________________________
tests/test_api.py:143: in test_recommend_503_without_model
    assert r.status_code == 503
E   assert 200 == 503
E    +  where 200 = <Response [200 OK]>.status_code
_________________________________________________________________________ TestFeedbackEndpoint.test_feedback_503_without_model __________________________________________________________________________
tests/test_api.py:221: in test_feedback_503_without_model
    assert r.status_code == 503
E   assert 200 == 503
E    +  where 200 = <Response [200 OK]>.status_code
======================================================================================== short test summary info ========================================================================================
FAILED tests/test_api.py::TestModelEndpoints::test_model_info_503_when_not_loaded - assert 200 == 503
 +  where 200 = <Response [200 OK]>.status_code
FAILED tests/test_api.py::TestModelEndpoints::test_ready_503_when_not_loaded - assert 200 == 503
 +  where 200 = <Response [200 OK]>.status_code
FAILED tests/test_api.py::TestRecommendEndpoint::test_recommend_503_without_model - assert 200 == 503
 +  where 200 = <Response [200 OK]>.status_code
FAILED tests/test_api.py::TestFeedbackEndpoint::test_feedback_503_without_model - assert 200 == 503
 +  where 200 = <Response [200 OK]>.status_code
===================================================================================== 4 failed, 98 passed in 3.04s ======================================================================================
```

```bash
# Run with coverage
pytest tests/test_bandit.py tests/test_data.py tests/test_drift.py \
  --cov=src --cov-report=term-missing 2>&1 | tail -30
```
**Output:**
```
tests/test_drift.py::TestDriftSimulation::test_simulate_returns_dataframe PASSED [ 94%]
tests/test_drift.py::TestDriftSimulation::test_simulate_has_required_columns PASSED [ 95%]
tests/test_drift.py::TestDriftSimulation::test_simulate_reward_binary PASSED [ 97%]
tests/test_drift.py::TestDriftSimulation::test_simulate_phase_zero_no_drift PASSED [ 98%]
tests/test_drift.py::TestDriftSimulation::test_simulate_reproducible PASSED [100%]

================================ tests coverage ================================
_______________ coverage: platform linux, python 3.12.13-final-0 _______________

Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/__Init__.py                   0      0   100%
src/api/__init__.py               0      0   100%
src/api/main.py                  46     46     0%   13-121
src/api/routes/__init__.py        0      0   100%
src/api/routes/feedback.py       27     27     0%   6-68
src/api/routes/recommend.py      29     29     0%   5-55
src/api/routes/system.py         48     48     0%   11-148
src/bandit/__init__.py            0      0   100%
src/bandit/linucb.py             66      1    98%   167
src/bandit/simulator.py          68     68     0%   16-173
src/data/__init__.py              0      0   100%
src/data/loader.py               90     36    60%   27-48, 183-200, 204-211
src/drift/__init__.py             0      0   100%
src/drift/detector.py            78      3    96%   51, 204, 264
src/evaluation/__init__.py        0      0   100%
src/evaluation/metrics.py       131    131     0%   13-265
-----------------------------------------------------------
TOTAL                           583    389    33%
============================== 70 passed in 3.21s ==============================
```

---

## 6. DVC Pipeline

```bash
# Verify DVC is initialized
dvc status
```
**Output:**
```
Data and pipelines are up to date.
```

```bash
# Show pipeline DAG
dvc dag
```
**Output:**
```
        +---------+      
        | prepare |      
        +---------+      
         *         **    
       **            *   
      *               ** 
+-------+               *
| train |             ** 
+-------+            *   
         *         **    
          **     **      
            *   *        
        +----------+     
        | evaluate |     
        +----------+     
```

```bash
# Run the full DVC pipeline
dvc repro 2>&1 | tail -20
```
**Output:**
```
Stage 'prepare' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Data and pipelines are up to date.
```

```bash
# Show metrics after pipeline run
dvc metrics show
```
**Output:**
```
Path                        alpha    arm_coverage    genre_diversity    match_rate    model_timesteps    test_ctr    test_steps    total_reward    train_ctr    train_steps    train_time_s
metrics/train_metrics.json  0.1      1.0             3.5954             0.1353        10822              -           -             -               0.6828       10822          108.61
metrics/eval_metrics.json   -        1.0             3.6213             -             -                  0.6334      2444          1548            -            -             
```

```bash
# Change alpha and show incremental re-run
sed -i 's/alpha: 0.1/alpha: 0.5/' params.yaml
dvc repro 2>&1 | grep -E "Stage|Running|Skipping|cached"
```
**Output:**
```
Stage 'prepare' didn't change, skipping
Stage 'train' is cached - skipping run, checking out outputs
Stage 'evaluate' is cached - skipping run, checking out outputs
```

```bash
# Compare metrics between runs
dvc metrics diff
```
**Output:**
```
DVC failed to load some metrics for following revisions: 'HEAD'.
Path                        Metric           HEAD    workspace    Change
metrics/train_metrics.json  alpha            -       0.5          -
metrics/train_metrics.json  arm_coverage     -       1.0          -
metrics/train_metrics.json  genre_diversity  -       3.5843       -
metrics/train_metrics.json  match_rate       -       0.036        -
metrics/train_metrics.json  model_timesteps  -       2882         -
metrics/train_metrics.json  train_ctr        -       0.6128       -
metrics/train_metrics.json  train_steps      -       2882         -
metrics/train_metrics.json  train_time_s     -       114.34       -
metrics/eval_metrics.json   arm_coverage     -       1.0          -
metrics/eval_metrics.json   genre_diversity  -       3.638        -
metrics/eval_metrics.json   test_ctr         -       0.5636       -
metrics/eval_metrics.json   test_steps       -       1125         -
metrics/eval_metrics.json   total_reward     -       634          -
```

```bash
# Restore best alpha
sed -i 's/alpha: 0.5/alpha: 0.1/' params.yaml
dvc repro 2>&1 | grep -E "Stage|Running|Skipping|cached"
```
**Output:**
```
Stage 'prepare' didn't change, skipping
Stage 'train' is cached - skipping run, checking out outputs
Stage 'evaluate' is cached - skipping run, checking out outputs
```

---

## 7. Docker — All Services

```bash
# Start all services
make up
```
**Output:**
```
Starting CineBandit services...
docker compose up -d
WARN[0000] No services to build                         
[+] up 16/16
 ✔ Network cinebandit_cinebandit          Created                                                                                                                                                   0.0s 
 ✔ Volume cinebandit_model_artifacts      Created                                                                                                                                                   0.0s 
 ✔ Volume cinebandit_grafana_data         Created                                                                                                                                                   0.0s 
 ✔ Volume cinebandit_prometheus_data      Created                                                                                                                                                   0.0s 
 ✔ Volume cinebandit_mlflow_artifacts     Created                                                                                                                                                   0.0s 
 ✔ Volume cinebandit_postgres_data        Created                                                                                                                                                   0.0s 
 ✔ Volume cinebandit_airflow_logs         Created                                                                                                                                                   0.0s 
 ✔ Container cinebandit_prometheus        Created                                                                                                                                                   0.3s 
 ✔ Container cinebandit_postgres          Healthy                                                                                                                                                   6.9s 
 ✔ Container cinebandit_grafana           Created                                                                                                                                                   0.2s 
 ✔ Container cinebandit_mlflow            Created                                                                                                                                                   0.2s 
 ✔ Container cinebandit_airflow_init      Exited                                                                                                                                                   26.6s 
 ✔ Container cinebandit_api               Created                                                                                                                                                   0.2s 
 ✔ Container cinebandit_frontend          Created                                                                                                                                                   0.2s 
 ✔ Container cinebandit_airflow_scheduler Created                                                                                                                                                   0.2s 
 ✔ Container cinebandit_airflow_webserver Created                                                                                                                                                   0.2s 

Waiting for API to be ready...
Successfully copied 98.4MB to cinebandit_api:/app/outputs/linucb_alpha0.1.pkl
Model loaded into API ✅

Services available at:
  API          → http://localhost:8000/docs
  Frontend     → http://localhost:8501
  MLflow       → http://localhost:5000
  Airflow      → http://localhost:8080  (admin/admin)
  Prometheus   → http://localhost:9090
  Grafana      → http://localhost:3000  (admin/admin)
```

```bash
# Verify all containers are running
docker compose ps
```
**Output:**
```
NAME                           IMAGE                           COMMAND                  SERVICE             CREATED              STATUS                                 PORTS
cinebandit_airflow_scheduler   cinebandit-airflow-scheduler    "/usr/bin/dumb-init …"   airflow-scheduler   About a minute ago   Up 52 seconds                          8080/tcp
cinebandit_airflow_webserver   cinebandit-airflow-webserver    "/usr/bin/dumb-init …"   airflow-webserver   About a minute ago   Up 52 seconds                          0.0.0.0:8080->8080/tcp, [::]:8080->8080/tcp
cinebandit_api                 cinebandit-api                  "uvicorn src.api.mai…"   api                 About a minute ago   Up About a minute (healthy)            0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp
cinebandit_frontend            cinebandit-frontend             "streamlit run src/f…"   frontend            About a minute ago   Up About a minute (healthy)            0.0.0.0:8501->8501/tcp, [::]:8501->8501/tcp
cinebandit_grafana             grafana/grafana:10.4.0          "/run.sh"                grafana             About a minute ago   Up About a minute                      0.0.0.0:3000->3000/tcp, [::]:3000->3000/tcp
cinebandit_mlflow              ghcr.io/mlflow/mlflow:v2.22.0   "sh -c 'pip install …"   mlflow              About a minute ago   Up About a minute (health: starting)   0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp
cinebandit_postgres            postgres:15-alpine              "docker-entrypoint.s…"   postgres            About a minute ago   Up About a minute (healthy)            0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp
cinebandit_prometheus          prom/prometheus:v2.51.0         "/bin/prometheus --c…"   prometheus          About a minute ago   Up About a minute                      0.0.0.0:9090->9090/tcp, [::]:9090->9090/tcp
```

---

## 8. FastAPI

```bash
# Health check
curl http://localhost:8000/health
```
**Output:**
```
{"status":"ok"}
```

```bash
# Readiness check (model loaded)
curl http://localhost:8000/ready
```
**Output:**
```
{"status":"ok","model_loaded":true,"model_timestep":10822}
```

```bash
# Model info
curl http://localhost:8000/model/info
```
**Output:**
```
{"algorithm":"LinUCB","alpha":0.1,"timesteps":10822,"arms_seen":1682,"total_movies":1682,"mean_ctr":0.07459146320325148,"model_path":"outputs/linucb_alpha0.1.pkl"}
```

```bash
# Get recommendations for user 1
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n": 3}'
```
**Output:**
```
{"user_id":1,"recommendations":[{"movie_id":112,"title":"Flipper (1996)","genres":["Adventure","Children's"],"year":1996,"ucb_score":0.8768},{"movie_id":523,"title":"Cool Hand Luke (1967)","genres":["Comedy","Drama"],"year":1967,"ucb_score":0.8647},{"movie_id":659,"title":"Arsenic and Old Lace (1944)","genres":["Comedy","Mystery","Thriller"],"year":1944,"ucb_score":0.8344}],"model_timestep":10822}
```

```bash
# Send feedback (like)
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 50, "reaction": "like"}'
```
**Output:**
```
{"status":"ok","user_id":1,"movie_id":50,"reaction":"like","reward":1.0,"rolling_ctr":1.0,"model_timestep":10823}
```

```bash
# Verify model updated (timestep should increment)
curl http://localhost:8000/snapshot
```
**Output:**
```
{"rolling_ctr":1.0,"model_timestep":10823,"arms_seen":1682,"drift_score":0.0}
```

```bash
# Check drift status
curl http://localhost:8000/drift/status
```
**Output:**
```
{"drift_score":0.0,"threshold":0.3,"alert":false,"status":"ok"}
```

```bash
# Check Prometheus metrics
curl -s http://localhost:8000/metrics | grep cinebandit | head -20
```
**Output:**
```
NO OUTPUT RETURNED
```

---

## 9. MLflow

```bash
# Verify MLflow server is up
curl http://localhost:5000/health
```
**Output:**
```
OK
```

```bash
# Run MLflow-tracked training
make train-mlflow
```
**Output:**
```
MLFLOW_TRACKING_URI=http://localhost:5000 python train_mlflow.py --alpha 0.1
2026-04-20 11:24:09,081 | INFO     | [mlflow] Tracking URI: http://localhost:5000
2026-04-20 11:24:09,593 | INFO     | [mlflow] Created experiment 'CineBandit-LinUCB' (id=1)
2026-04-20 11:24:09,594 | INFO     | [train] Loading data...
[loader] Dataset already present at /home/chirag/MLOps/cinebandit/data/raw/ml-100k
[loader] Loading ratings ...
         100,000 ratings loaded
[loader] Loading movies ...
         1,682 movies loaded
[loader] Loading users ...
         943 users loaded
[loader] Building user genre profiles ...
         942 user profiles built
2026-04-20 11:24:11,318 | INFO     | [mlflow] Run ID: e724cd4ab9814780ba5e0cee1cb8644d
2026-04-20 11:24:11,355 | INFO     | [train] Training with alpha=0.1 ...
Simulating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 80000/80000 [02:06<00:00, 630.84it/s, CTR=0.683, match_rate=1.000, matched=10500]

[sim] Total steps processed : 80,000
[sim] Replay matches        : 10,822 (13.5%)
[sim] Mean CTR              : 0.6828
2026-04-20 11:26:18,187 | INFO     | [train] Running test simulation...
Simulating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 20000/20000 [00:32<00:00, 617.17it/s, CTR=0.614, match_rate=1.000, matched=2000]

[sim] Total steps processed : 20,000
[sim] Replay matches        : 2,444 (12.2%)
[sim] Mean CTR              : 0.6334
2026-04-20 11:26:54,712 | INFO     | [mlflow] Model pickle logged as artifact
2026/04/20 11:26:54 INFO mlflow.pyfunc: Validating input example against model signature
2026-04-20 11:26:59,362 | INFO     | [mlflow] PyFunc model logged
[eval] Plot saved to /tmp/tmpg3p1mdl_/train_evaluation.png
[eval] Plot saved to /tmp/tmpg3p1mdl_/test_evaluation.png
2026-04-20 11:27:00,269 | INFO     | [mlflow] Evaluation plots logged
2026-04-20 11:27:00,291 | INFO     | [mlflow] Run complete. train_ctr=0.6828 test_ctr=0.6334

=======================================================
  LinUCB Evaluation Summary
=======================================================
  Matched steps (unbiased updates) : 2,444
  Total bandit timesteps           : 10,822
  Arms (movies) seen               : 1,682 / 1,682 (100.0%)

  Final CTR                        : 0.6334
  Cumulative reward                : 1548

  Genre diversity (entropy)        : 3.621 bits (max=4.25)
  Arm coverage                     : 100.0%

  Most pulled movie_id             : 50 (351 pulls)
=======================================================

🏃 View run loud-sow-976 at: http://localhost:5000/#/experiments/1/runs/e724cd4ab9814780ba5e0cee1cb8644d
🧪 View experiment at: http://localhost:5000/#/experiments/1
Successfully registered model 'CineBandit'.
2026/04/20 11:27:00 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: CineBandit, version 1
Created version '1' of model 'CineBandit'.
2026-04-20 11:27:00,477 | INFO     | [mlflow] Registered model v1 from run e724cd4ab9814780ba5e0cee1cb8644d
2026-04-20 11:27:00,514 | INFO     | [mlflow] Challenger CTR=0.6334 vs Champion CTR=0.0000
2026-04-20 11:27:00,547 | INFO     | [mlflow] ✅ Model v1 set as 'production' (CTR 0.0000 → 0.6334)

Run ID   : e724cd4ab9814780ba5e0cee1cb8644d
Alpha    : 0.1
Train CTR: 0.6828
Test CTR : 0.6334
Version  : 1
Model    : outputs/linucb_alpha0.1.pkl
```

```bash
# List registered models
curl -s http://localhost:5000/api/2.0/mlflow/registered-models/list | \
  python3 -m json.tool | grep -E "name|version|alias"
```
**Output:**
```
Expecting value: line 1 column 1 (char 0)
```

---

## 10. Airflow

```bash
# Verify Airflow is up
curl -s http://localhost:8080/health | python3 -m json.tool
```
**Output:**
```
{
    "dag_processor": {
        "latest_dag_processor_heartbeat": null,
        "status": null
    },
    "metadatabase": {
        "status": "healthy"
    },
    "scheduler": {
        "latest_scheduler_heartbeat": "2026-04-20T11:31:55.718019+00:00",
        "status": "healthy"
    },
    "triggerer": {
        "latest_triggerer_heartbeat": null,
        "status": null
    }
}
```

```bash
# List all DAGs
curl -s -u admin:admin http://localhost:8080/api/v1/dags | \
  python3 -m json.tool | grep "dag_id"
```
**Output:**
```
"dag_id": "cinebandit_drift_detection",
"root_dag_id": null,
"dag_id": "cinebandit_ingestion",
"root_dag_id": null,
"dag_id": "cinebandit_retraining",
"root_dag_id": null,
```

```bash
# Trigger ingestion DAG
curl -s -X POST -u admin:admin \
  http://localhost:8080/api/v1/dags/cinebandit_ingestion/dagRuns \
  -H "Content-Type: application/json" \
  -d '{"conf": {}}' | python3 -m json.tool | grep -E "dag_id|state|run_id"
```
**Output:**
```
"dag_id": "cinebandit_ingestion",
"dag_run_id": "manual__2026-04-20T11:32:45.216025+00:00",
"state": "queued"
```

```bash
# Wait 60 seconds then check ingestion run status
sleep 60
curl -s -u admin:admin \
  "http://localhost:8080/api/v1/dags/cinebandit_ingestion/dagRuns?limit=1&order_by=-execution_date" | \
  python3 -m json.tool | grep -E "state|dag_id|execution_date"
```
**Output:**
```
            "dag_id": "cinebandit_ingestion",
            "execution_date": "2026-04-20T11:32:45.216025+00:00",
            "state": "queued"
```

```bash
# Verify baseline was created
cat data/baselines/genre_baseline.json | python3 -m json.tool
```
**Output:**
```
{
    "genre_distribution": [
        4.196884233145313e-05,
        0.11360126242277734,
        0.0616018667741069,
        0.017022562449637388,
        0.028782232070910557,
        0.1254532634971797,
        0.04027330110126242,
        0.004003827558420629,
        0.20652867311308085,
        0.00469211657265646,
        0.010273972602739725,
        0.020875302175664787,
        0.022470118184260004,
        0.026507520816545796,
        0.09603310502283105,
        0.06035958904109589,
        0.10009568896051571,
        0.05245265914585012,
        0.008930969648133226
    ],
    "genre_names": [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western"
    ],
    "rating_mean": 3.52986,
    "rating_std": 1.125673599144316,
    "like_rate": 0.55375,
    "n_ratings": 100000
}
```

```bash
# Trigger drift detection
curl -s -X POST -u admin:admin \
  http://localhost:8080/api/v1/dags/cinebandit_drift_detection/dagRuns \
  -H "Content-Type: application/json" \
  -d '{"conf": {}}' | python3 -m json.tool | grep -E "dag_id|state|run_id"
```
**Output:**
```
    "dag_id": "cinebandit_drift_detection",
    "dag_run_id": "manual__2026-04-20T11:35:36.599883+00:00",
    "state": "queued"
```

```bash
# Trigger retraining manually
curl -s -X POST -u admin:admin \
  http://localhost:8080/api/v1/dags/cinebandit_retraining/dagRuns \
  -H "Content-Type: application/json" \
  -d '{"conf": {"reason": "manual_test"}}' | \
  python3 -m json.tool | grep -E "dag_id|state|run_id"
```
**Output:**
```
    "dag_id": "cinebandit_retraining",
    "dag_run_id": "manual__2026-04-20T11:36:52.100328+00:00",
    "state": "queued"
```

---

## 11. Prometheus

```bash
# Prometheus health
curl http://localhost:9090/-/ready
```
**Output:**
```
Prometheus Server is Ready.
```

```bash
# Verify Prometheus is scraping the API target successfully
curl -s "http://localhost:9090/api/v1/targets" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
targets = data.get('data', {}).get('activeTargets', [])
for t in targets:
    print('Job     :', t.get('labels', {}).get('job'))
    print('URL     :', t.get('scrapeUrl'))
    print('Health  :', t.get('health'))
    print('Last err:', t.get('lastError') or 'none')
    print('Last scrape:', t.get('lastScrape', '')[:19])
    print()
"
```
**Output:**
```
Job     : cinebandit_api
URL     : http://api:8000/metrics
Health  : up
Last err: none
Last scrape: 2026-04-20T11:37:16

Job     : prometheus
URL     : http://localhost:9090/metrics
Health  : up
Last err: none
Last scrape: 2026-04-20T11:37:10
```

```bash
# Make a few API requests to generate metrics
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" -d '{"user_id": 1, "n": 3}' > /dev/null
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 50, "reaction": "like"}' > /dev/null
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 100, "reaction": "dislike"}' > /dev/null
echo "Requests sent — waiting 15s for Prometheus to scrape..."
sleep 15
```
**Output:**
```
Requests sent — waiting 15s for Prometheus to scrape...
```

```bash
# Query recommendation request counter from Prometheus
curl -s "http://localhost:9090/api/v1/query?query=cinebandit_recommendation_requests_total" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data.get('data', {}).get('result', [])
if results:
    for r in results:
        print(f'Labels : {r["metric"]}')
        print(f'Value  : {r["value"][1]}')
        print()
else:
    print('No data yet — check that API /metrics endpoint is reachable by Prometheus')
"
```
**Output:**
```
Traceback (most recent call last):
  File "<string>", line 7, in <module>
NameError: name 'metric' is not defined
```

```bash
# Query feedback counter broken down by reaction type
curl -s "http://localhost:9090/api/v1/query?query=cinebandit_feedback_total" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data.get('data', {}).get('result', [])
if results:
    for r in results:
        reaction = r['metric'].get('reaction', 'unknown')
        value    = r['value'][1]
        print(f'reaction={reaction:<10} count={value}')
else:
    print('No feedback metrics yet')
"
```
**Output:**
```
reaction=like       count=2
reaction=dislike    count=1
```

```bash
# Query rolling CTR gauge
curl -s "http://localhost:9090/api/v1/query?query=cinebandit_rolling_ctr" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data.get('data', {}).get('result', [])
if results:
    print(f'Rolling CTR: {results[0]["value"][1]}')
else:
    print('No rolling CTR data yet')
"
```
**Output:**
```
Traceback (most recent call last):
  File "<string>", line 6, in <module>
NameError: name 'value' is not defined. Did you mean: 'False'?
```

```bash
# Query drift score gauge
curl -s "http://localhost:9090/api/v1/query?query=cinebandit_drift_score" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data.get('data', {}).get('result', [])
if results:
    print(f'Drift score: {results[0]["value"][1]}')
else:
    print('No drift score data yet — trigger drift detection DAG first')
"
```
**Output:**
```
Traceback (most recent call last):
  File "<string>", line 6, in <module>
NameError: name 'value' is not defined. Did you mean: 'False'?
```

```bash
# Query recommendation latency histogram (p95)
curl -s "http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(cinebandit_recommendation_latency_seconds_bucket[5m]))" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data.get('data', {}).get('result', [])
if results:
    latency_ms = float(results[0]['value'][1]) * 1000
    print(f'p95 latency: {latency_ms:.1f}ms  (target < 200ms)')
else:
    print('No latency data yet')
"
```
**Output:**
```
Traceback (most recent call last):
  File "<string>", line 3, in <module>
  File "/home/chirag/miniconda3/envs/cinebandit/lib/python3.12/json/__init__.py", line 293, in load
    return loads(fp.read(),
           ^^^^^^^^^^^^^^^^
  File "/home/chirag/miniconda3/envs/cinebandit/lib/python3.12/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chirag/miniconda3/envs/cinebandit/lib/python3.12/json/decoder.py", line 338, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chirag/miniconda3/envs/cinebandit/lib/python3.12/json/decoder.py", line 356, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

---

## 12. Grafana

```bash
# Grafana health
curl -s http://localhost:3000/api/health | python3 -m json.tool
```
**Output:**
```
{
    "commit": "03f502a94d17f7dc4e6c34acdf8428aedd986e4c",
    "database": "ok",
    "version": "10.4.0"
}
```

```bash
# Verify Prometheus datasource is configured and working
curl -s -u admin:admin \
  "http://localhost:3000/api/datasources" | \
  python3 -c "
import sys, json
datasources = json.load(sys.stdin)
for ds in datasources:
    print(f'Name   : {ds.get("name")}')
    print(f'Type   : {ds.get("type")}')
    print(f'URL    : {ds.get("url")}')
    print(f'Default: {ds.get("isDefault")}')
    print()
"
```
**Output:**
```
Traceback (most recent call last):
  File "<string>", line 5, in <module>
NameError: name 'name' is not defined
```

```bash
# Verify CineBandit dashboard is provisioned
curl -s -u admin:admin \
  "http://localhost:3000/api/search?query=CineBandit" | \
  python3 -c "
import sys, json
results = json.load(sys.stdin)
if results:
    for r in results:
        print(f'Title  : {r.get("title")}')
        print(f'UID    : {r.get("uid")}')
        print(f'URL    : {r.get("url")}')
        print(f'Type   : {r.get("type")}')
        print()
else:
    print('No dashboards found — check grafana/dashboards/cinebandit.json is mounted')
"
```
**Output:**
```
Traceback (most recent call last):
  File "<string>", line 6, in <module>
NameError: name 'title' is not defined. Did you mean: 'tuple'?
```

```bash
# Verify all dashboard panels are configured (panel count)
curl -s -u admin:admin \
  "http://localhost:3000/api/dashboards/uid/cinebandit-main" | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
dashboard = data.get('dashboard', {})
panels = dashboard.get('panels', [])
print(f'Dashboard : {dashboard.get("title")}')
print(f'UID       : {dashboard.get("uid")}')
print(f'Panels    : {len(panels)}')
print()
print('Panel titles:')
for p in panels:
    print(f'  [{p.get("id"):>2}] {p.get("title")}')
"
```
**Output:**
```
Traceback (most recent call last):
  File "<string>", line 6, in <module>
NameError: name 'title' is not defined. Did you mean: 'tuple'?
```

```bash
# Test a live Grafana query against Prometheus datasource
# (evaluates recommendation requests metric through Grafana)
curl -s -u admin:admin \
  -H "Content-Type: application/json" \
  -X POST "http://localhost:3000/api/ds/query" \
  -d '{
    "queries": [{
      "refId": "A",
      "datasource": {"type": "prometheus"},
      "expr": "cinebandit_recommendation_requests_total",
      "range": true,
      "instant": true
    }],
    "from": "now-5m",
    "to": "now"
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
results = data.get('results', {}).get('A', {})
frames = results.get('frames', [])
if frames:
    print(f'Query returned {len(frames)} frame(s) — Grafana → Prometheus pipeline working ✅')
    for frame in frames:
        schema = frame.get('schema', {})
        fields = schema.get('fields', [])
        print(f'Fields: {[f.get("name") for f in fields]}')
else:
    print('No data returned — check datasource configuration')
" 2>/dev/null || echo "Grafana query API not available in this version"
```
**Output:**
```
No data returned — check datasource configuration
```

---

## 13. Streamlit

```bash
# Verify Streamlit frontend is up
curl -s http://localhost:8501/_stcore/health
```
**Output:**
```
ok
```

---

## 14. Full End-to-End Flow

```bash
# Simulate a full user session:
# 1. Get recommendations
RECS=$(curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "n": 5}')
echo $RECS | python3 -m json.tool
```
**Output:**
```
{
    "user_id": 42,
    "recommendations": [
        {
            "movie_id": 692,
            "title": "American President, The (1995)",
            "genres": [
                "Comedy",
                "Drama",
                "Romance"
            ],
            "year": 1995,
            "ucb_score": 0.7972
        },
        {
            "movie_id": 186,
            "title": "Blues Brothers, The (1980)",
            "genres": [
                "Action",
                "Comedy",
                "Musical"
            ],
            "year": 1980,
            "ucb_score": 0.7831
        },
        {
            "movie_id": 135,
            "title": "2001: A Space Odyssey (1968)",
            "genres": [
                "Drama",
                "Mystery",
                "Sci-Fi",
                "Thriller"
            ],
            "year": 1968,
            "ucb_score": 0.7665
        },
        {
            "movie_id": 420,
            "title": "Alice in Wonderland (1951)",
            "genres": [
                "Animation",
                "Children's",
                "Musical"
            ],
            "year": 1951,
            "ucb_score": 0.6602
        },
        {
            "movie_id": 68,
            "title": "Crow, The (1994)",
            "genres": [
                "Action",
                "Romance",
                "Thriller"
            ],
            "year": 1994,
            "ucb_score": 0.5969
        }
    ],
    "model_timestep": 10825
}
```

```bash
# 2. Extract first movie_id and send like
MOVIE_ID=$(echo $RECS | python3 -c "import sys,json; print(json.load(sys.stdin)['recommendations'][0]['movie_id'])")
echo "Sending like for movie_id: $MOVIE_ID"

curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": 42, \"movie_id\": $MOVIE_ID, \"reaction\": \"like\"}" | \
  python3 -m json.tool
```
**Output:**
```
Sending like for movie_id: 692
{
    "status": "ok",
    "user_id": 42,
    "movie_id": 692,
    "reaction": "like",
    "reward": 1.0,
    "rolling_ctr": 0.75,
    "model_timestep": 10826
}
```

```bash
# 3. Get new recommendations — model has now updated
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "n": 5}' | python3 -m json.tool
```
**Output:**
```
{
    "user_id": 42,
    "recommendations": [
        {
            "movie_id": 170,
            "title": "Cinema Paradiso (1988)",
            "genres": [
                "Comedy",
                "Drama",
                "Romance"
            ],
            "year": 1988,
            "ucb_score": 0.8982
        },
        {
            "movie_id": 510,
            "title": "Magnificent Seven, The (1954)",
            "genres": [
                "Action",
                "Drama",
                "Western"
            ],
            "year": 1954,
            "ucb_score": 0.6623
        },
        {
            "movie_id": 420,
            "title": "Alice in Wonderland (1951)",
            "genres": [
                "Animation",
                "Children's",
                "Musical"
            ],
            "year": 1951,
            "ucb_score": 0.6602
        },
        {
            "movie_id": 521,
            "title": "Deer Hunter, The (1978)",
            "genres": [
                "Drama",
                "War"
            ],
            "year": 1978,
            "ucb_score": 0.6061
        },
        {
            "movie_id": 355,
            "title": "Sphere (1998)",
            "genres": [
                "Adventure",
                "Sci-Fi",
                "Thriller"
            ],
            "year": 1998,
            "ucb_score": 0.2518
        }
    ],
    "model_timestep": 10826
}
```

```bash
# 4. Verify timestep increased
curl -s http://localhost:8000/snapshot | python3 -m json.tool
```
**Output:**
```
{
    "rolling_ctr": 0.75,
    "model_timestep": 10826,
    "arms_seen": 1682,
    "drift_score": 0.0
}
```

---

## Summary Checklist

After running all commands, verify:

- [ ] Python environment with all packages ✓
- [ ] MovieLens-100K data loaded (100,000 ratings) ✓
- [ ] LinUCB trained (train_ctr ~0.68, test_ctr ~0.63) ✓
- [ ] All unit tests passing (64+ passed, 0 failed) ✓
- [ ] DVC pipeline runs (prepare → train → evaluate) ✓
- [ ] DVC metrics show test_ctr and genre_diversity ✓
- [ ] All 9 Docker containers running ✓
- [ ] API /health → {"status":"ok"} ✓
- [ ] API /ready → model_loaded: true ✓
- [ ] /recommend returns movie titles with UCB scores ✓
- [ ] /feedback increments model_timestep ✓
- [ ] MLflow experiment visible at localhost:5000 ✓
- [ ] CineBandit model registered in MLflow registry ✓
- [ ] All 3 Airflow DAGs listed ✓
- [ ] Ingestion DAG runs successfully ✓
- [ ] data/baselines/genre_baseline.json created ✓
- [ ] Prometheus scraping API metrics ✓
- [ ] Grafana CineBandit dashboard provisioned ✓
- [ ] Streamlit frontend accessible at localhost:8501 ✓
- [ ] End-to-end feedback loop updates model ✓