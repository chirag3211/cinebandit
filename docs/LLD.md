# CineBandit
### Low-Level Design Document
*API Endpoint Specifications | Version 1.0 | April 2026*

---

## 1. Overview

This document specifies all FastAPI endpoint definitions including input/output schemas, data types, validation constraints, and example payloads. All endpoints are served at `http://localhost:8000` in local deployment and `http://api:8000` within the Docker network.

- **Base URL:** `http://localhost:8000`
- **Content-Type:** `application/json` (for all POST requests)
- **Authentication:** None (internal service — Docker network isolated)
- **Error format:** `{"detail": "<error message>"}` with appropriate HTTP status code

### 1.1 HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 OK | Request succeeded |
| 422 Unprocessable Entity | Request validation failed (wrong type, out of range, missing field) |
| 404 Not Found | Resource not found (invalid user_id or movie_id) |
| 503 Service Unavailable | Model not loaded — call /model/reload first |
| 500 Internal Server Error | Unexpected server error — check API logs |

---

## 2. Endpoint Specifications

### `GET /health` [System]

Liveness probe. Always returns 200 if the server process is running. Used by Docker healthcheck and orchestration systems.

**Response Body (200 OK)**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| status | string | Always | Always 'ok' |

**Example:**
```json
{"status": "ok"}
```

---

### `GET /ready` [System]

Readiness probe. Returns 200 only when a model is loaded and ready to serve recommendations. Returns 503 if model not loaded.

**Response Body (200 OK)**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| status | string | Always | 'ok' or 'not_ready' |
| model_loaded | boolean | Always | True if model is loaded |
| model_timestep | integer | If loaded | Current model update count |

**Example:**
```json
{"status": "ok", "model_loaded": true, "model_timestep": 10822}
```

---

### `POST /recommend` [Inference]

Get top-N movie recommendations for a user. The LinUCB model scores all candidate movies using UCB formula and returns highest-scoring ones.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_id | integer | Required | User ID (1-943 for ML-100K) |
| n | integer | Optional (default: 5) | Number of recommendations (1-20) |
| exclude | array[int] | Optional | Movie IDs to exclude from recommendations |

**Response Body (200 OK)**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_id | integer | Always | Echo of request user_id |
| recommendations | array | Always | List of recommended movies (see below) |
| recommendations[].movie_id | integer | Always | MovieLens movie ID |
| recommendations[].title | string | Always | Movie title and year |
| recommendations[].genres | array[string] | Always | List of genre labels |
| recommendations[].year | integer | Always | Release year |
| recommendations[].ucb_score | float | Always | UCB score (higher = more confident) |
| model_timestep | integer | Always | Model version at time of request |

**Example:**
```json
{"user_id":1,"recommendations":[{"movie_id":50,"title":"Star Wars (1977)","genres":["Action","Adventure","Sci-Fi"],"year":1977,"ucb_score":0.9234}],"model_timestep":10822}
```

---

### `POST /feedback` [Inference]

Submit user feedback for a recommendation. Triggers an incremental LinUCB parameter update — the model learns from every interaction in real time.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_id | integer | Required | User ID |
| movie_id | integer | Required | Movie ID that was recommended |
| reaction | string (enum) | Required | One of: 'like', 'dislike', 'skip' |

**Response Body (200 OK)**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| status | string | Always | Always 'ok' on success |
| user_id | integer | Always | Echo of request user_id |
| movie_id | integer | Always | Echo of request movie_id |
| reaction | string | Always | Echo of request reaction |
| reward | float | Always | Reward value: like=1.0, dislike=0.0, skip=0.0 |
| rolling_ctr | float | Always | CTR over last 100 feedback events |
| model_timestep | integer | Always | Incremented model update count |

**Example:**
```json
{"status":"ok","user_id":1,"movie_id":50,"reaction":"like","reward":1.0,"rolling_ctr":0.75,"model_timestep":10823}
```

---

### `GET /model/info` [Model]

Return metadata about the currently loaded model. Returns 503 if no model is loaded.

**Response Body (200 OK)**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| algorithm | string | Always | Model algorithm name (LinUCB) |
| alpha | float | Always | Exploration parameter |
| timesteps | integer | Always | Total number of incremental updates |
| arms_seen | integer | Always | Number of unique movies updated on |
| total_movies | integer | Always | Total movies in dataset |
| mean_ctr | float | Always | Mean empirical CTR across all arms |
| model_path | string | Always | Path to loaded model file |

**Example:**
```json
{"algorithm":"LinUCB","alpha":0.1,"timesteps":10822,"arms_seen":1682,"total_movies":1682,"mean_ctr":0.075,"model_path":"outputs/linucb_alpha0.1.pkl"}
```

---

### `GET /drift/status` [Monitoring]

Return the current data drift score and whether the alert threshold has been exceeded. Updated periodically by the Airflow drift detection DAG.

**Response Body (200 OK)**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| drift_score | float | Always | KL divergence score (0 = no drift) |
| threshold | float | Always | Alert threshold (default: 0.3) |
| alert | boolean | Always | True if drift_score > threshold |
| status | string | Always | 'ok' or 'drift_detected' |

**Example:**
```json
{"drift_score":0.0,"threshold":0.3,"alert":false,"status":"ok"}
```

---

### `POST /drift/update` [Monitoring]

Internal endpoint called by the Airflow drift detection DAG to push the latest KL divergence score into the API.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| score | float (query param) | Required | KL divergence score from drift detection |

**Response Body (200 OK)**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| status | string | Always | Always 'ok' |
| drift_score | float | Always | Echo of posted score |

**Example:**
```json
{"status":"ok","drift_score":0.25}
```

---

### `POST /retrain/trigger` [Model]

Manually trigger a retraining pipeline run. Increments the Prometheus retrain counter. In production, also calls the Airflow DAG REST API.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| reason | string | Optional (default: 'manual') | Reason for triggering retraining |

**Response Body (200 OK)**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| status | string | Always | Always 'triggered' |
| reason | string | Always | Echo of request reason |
| message | string | Always | Human-readable status message |

**Example:**
```json
{"status":"triggered","reason":"manual","message":"Retraining pipeline triggered."}
```

---

### `POST /model/reload` [Model]

Hot-reload the model from disk without restarting the server. Called by the Airflow retraining DAG after a new champion model is saved.

**Response Body (200 OK)**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| status | string | Always | Always 'ok' on success |
| message | string | Always | Confirmation message |
| timesteps | integer | Always | Timestep of newly loaded model |

**Example:**
```json
{"status":"ok","message":"Model reloaded successfully","timesteps":10822}
```

---

### `GET /snapshot` [Monitoring]

Quick metrics snapshot for the Streamlit dashboard. Returns the most important live metrics in a single call to minimise round trips.

**Response Body (200 OK)**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| rolling_ctr | float | Always | Rolling CTR over last 100 feedback events |
| model_timestep | integer | Always | Current model update count |
| arms_seen | integer | Always | Unique movies the model has updated on |
| drift_score | float | Always | Current KL divergence drift score |

**Example:**
```json
{"rolling_ctr":0.75,"model_timestep":10823,"arms_seen":1682,"drift_score":0.0}
```

---

### `GET /metrics` [System]

Prometheus metrics endpoint. Returns all instrumented metrics in Prometheus exposition format. Scraped every 15 seconds by the Prometheus container.

**Example:**
```
# HELP cinebandit_rolling_ctr Rolling CTR
# TYPE cinebandit_rolling_ctr gauge
cinebandit_rolling_ctr 0.75
```

---

## 3. Module Structure

| Module | Responsibility |
|--------|---------------|
| `src/api/main.py` | App factory, lifespan startup, CORS middleware, Prometheus /metrics mount, global exception handler |
| `src/api/core/model_store.py` | Thread-safe singleton ModelStore. Holds LinUCB model in memory. Exposes recommend(), update(), reload(), info properties |
| `src/api/core/metrics.py` | All 12 Prometheus metric definitions as module-level singletons. RollingCTRTracker class |
| `src/api/core/schemas.py` | Pydantic request/response models for all endpoints. Reaction enum. REACTION_TO_REWARD mapping |
| `src/api/routes/recommend.py` | POST /recommend handler. Calls store.recommend(), records latency histogram |
| `src/api/routes/feedback.py` | POST /feedback handler. Calls store.update(), increments counters, updates rolling CTR |
| `src/api/routes/system.py` | GET /health, /ready, /model/info, /drift/status, /snapshot. POST /drift/update, /retrain/trigger, /model/reload |

---

## 4. Data Models

### 4.1 MovieRecommendation

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| movie_id | integer | Always | MovieLens movie identifier |
| title | string | Always | Full movie title including year e.g. 'Star Wars (1977)' |
| genres | array[string] | Always | Genre labels from MovieLens e.g. ['Action', 'Adventure'] |
| year | integer | Always | 4-digit release year extracted from title |
| ucb_score | float | Always | Upper Confidence Bound score. Higher = model more confident this is a good recommendation for this user |

### 4.2 Reaction Enum

| Value | Type | Description |
|-------|------|-------------|
| like | string | User liked the movie. Reward = 1.0. Model parameters updated toward this movie for this user context. |
| dislike | string | User disliked the movie. Reward = 0.0. Model learns to avoid recommending similar movies to this user. |
| skip | string | User skipped without rating. Reward = 0.0. Treated as neutral signal. |
