---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
lastStep: 8
status: complete
completedAt: '2026-06-03'
inputDocuments:
  - _bmad-output/planning-artifacts/prds/prd-DNO_advanced-2026-06-03/prd.md
  - _bmad-output/planning-artifacts/prds/prd-DNO_advanced-2026-06-03/addendum.md
workflowType: 'architecture'
project_name: 'DNO_advanced'
user_name: 'Aswath'
date: '2026-06-03'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements (17 total across 5 groups):**
- Data Layer (FR-D1–D4): Hybrid synthetic/real CSV dataset for 38 district facilities and 5 CDST labs; ORS 38×5 travel-time matrix cached locally; reproducible data generator with fixed seed.
- Graph Engine (FR-G1–G2): In-memory NetworkX directed graph, rebuilt each session. 43 nodes, 190 directed edges weighted by composite score.
- Routing Intelligence (FR-R1–R4): Dijkstra static routing + pre-trained tabular Q-learning agent. Both results surfaced when they disagree.
- Anomaly Signal (FR-A1–A3): Pre-trained Isolation Forest on facility features; anomaly score drives map colour-coding and optional routing weight.
- SHAP Explainability (FR-S1–S3): Function-level SHAP wrapper over the edge-weight function (not a fitted model); renders compact bar chart + two-line plain-language output.
- Streamlit UI (FR-U1–U4): Three-panel layout (sidebar, map, results); map-click origin selection; Simulate Stockout climax flow; <3s render NFR.

**Non-Functional Requirements:**
- Local-only runtime: single `streamlit run app.py`, no Docker/cloud
- Reproducibility: fixed random seed across data generator, models, Q-table
- ORS fallback: Haversine estimates when API unavailable; app never crashes
- Zero-build deps: single `pip install -r requirements.txt`
- No page reloads: all interactions via Streamlit session state

**Scale & Complexity:**
- Complexity level: Low (local demo, single user, no concurrency, no auth)
- Primary domain: ML inference + geospatial data visualisation
- Nodes: 43 | Edges: 190 | Data volume: two small CSVs

### Technical Constraints & Dependencies

- Python 3.12, venv-isolated
- OpenRouteService API (external, requires key in .env — cache committed to repo for demo resilience)
- Pre-trained model artefacts (Isolation Forest, Q-table) must be serialised and present at startup — no live training at runtime
- Streamlit process model: single-threaded; session state is the only inter-component communication layer
- scikit-learn version must be pinned — joblib artefacts are version-sensitive

### Cross-Cutting Concerns

- **Startup sequencing:** ORS cache check → graph build → model load must complete before UI renders; each step degrades gracefully with visible feedback via `st.status()`
- **Model serialisation:** Isolation Forest as `seed{N}.joblib`; Q-table as `seed{N}.npy`; load-time assertion validates seed match
- **Session state ownership:** all mutable UI state owned by `st.session_state`; graph treated as immutable after build; stockout simulation parameterises weight function, never mutates graph object
- **Reproducible seed:** single `RANDOM_SEED` constant propagated to numpy, sklearn, and Q-learning trainer
- **Demo narrative:** Stockout simulation is the climax — UI architecture must build toward it, not bury it

### Party Mode Insights (Winston · Amelia · John)

- Pre-commit the ORS travel-time cache to the repo; treat demo as air-gapped
- Add a manifest JSON at training time (data hash + seed) validated at startup
- SHAP background dataset must be ≥50 rows; explicit `(n_samples, n_features) → (n_samples,)` adapter required
- Add a visible "Reset" button — judges reload browser tabs
- The SHAP plain-language explanation must read as clinician language, not feature weights
- The Dijkstra vs Q-learning comparison needs a healthcare framing, not an algorithm label

## Starter Template Evaluation

### Primary Technology Domain

Python ML inference + geospatial data visualisation (local demo). No web framework starter applies. Stack fully settled from PRD and installed packages.

### Project Structure Decision

No CLI-generated starter. Manual `src/` layout — standard Python best practice, enforces module boundaries that match the 7 architectural components identified in Step 2.

**Initialization command:**
```bash
mkdir -p src/data src/graph src/routing src/models \
         src/explainability scripts tests artifacts cache data
touch app.py src/__init__.py
```

**Project layout:**
```
DNO_advanced/
├── app.py                        ← Streamlit entry point + session_state init gate
├── data/                         ← facilities.csv, cdst_labs.csv
├── artifacts/                    ← isolation_forest_seed42.joblib, qtable_seed42.npy, manifest.json
├── cache/                        ← ors_matrix.npy (committed to repo for demo resilience)
├── src/
│   ├── data/
│   │   ├── generator.py          ← Synthetic data generation (fixed seed)
│   │   └── ors_client.py         ← ORS API call + Haversine fallback
│   ├── graph/
│   │   └── builder.py            ← NetworkX graph construction (immutable after build)
│   ├── routing/
│   │   ├── dijkstra.py           ← Dijkstra shortest path
│   │   └── qlearning.py          ← Q-table inference (pre-trained only)
│   ├── models/
│   │   ├── anomaly.py            ← Isolation Forest inference
│   │   └── loader.py             ← Artefact loading + seed validation
│   └── explainability/
│       └── shap_wrapper.py       ← SHAP function wrapper + background dataset
├── scripts/
│   ├── generate_data.py          ← Run once to produce CSVs
│   └── train_models.py           ← Run once to produce artifacts/
├── tests/
├── requirements.txt
├── .env                          ← ORS_API_KEY (gitignored)
└── .gitignore
```

**Architectural decisions this layout enforces:**
- `app.py` owns session_state; no module holds mutable globals
- `artifacts/` and `cache/` committed to repo — demo is air-gapped
- `scripts/` are offline-only; never imported by `app.py`
- `src/models/loader.py` is the single entry point for all artefact loading and seed validation
- `src/routing/` separates Dijkstra and Q-learning — each independently testable

## Architecture Validation Results

### Coherence Validation ✅

All Python packages compatible with 3.12. `@st.cache_data` and session_state gate are complementary. Override dict pattern does not conflict with immutable graph design. Absolute imports align with venv-isolated Streamlit process model.

### Requirements Coverage

| FR | Status | File |
|---|---|---|
| FR-D1–D4 | ✅ | `src/data/loader.py`, `scripts/generate_data.py`, `cache/ors_matrix.npy` |
| FR-G1–G2 | ✅ | `src/graph/builder.py` |
| FR-R1–R4 | ✅ | `src/config.py`, `src/routing/dijkstra.py`, `src/routing/qlearning.py`, `app.py` |
| FR-A1–A2 | ✅ | `src/models/anomaly.py`, `app.py` |
| FR-A3 | ⚠️ | Gap resolved — see below |
| FR-S1–S3 | ✅ | `src/explainability/shap_wrapper.py` |
| FR-U1–U4 | ✅ | `app.py` |
| NFR-1–5 | ✅ | Fully covered |

### Gaps Resolved

**Gap 1 — FR-A3: δ anomaly weight term**
- Add `DELTA = 0.0` to `src/config.py`
- Update `build_graph()` signature:
  ```python
  def build_graph(facilities_df, cdst_df, travel_matrix,
                  anomaly_scores: pd.Series | None = None,
                  overrides: dict[str, int] | None = None,
                  delta: float = 0.0) -> nx.DiGraph:
  ```
- Add δ slider to `app.py` sidebar alongside α/β/γ
- Add `st.session_state.anomaly_scores` to canonical session_state keys

**Gap 2 — FR-U2: Folium map-click pattern**
```python
# app.py — canonical map click handler
map_state = st_folium(m, width=700, height=500)
if map_state["last_object_clicked_tooltip"]:
    facility_id = parse_facility_id_from_tooltip(
        map_state["last_object_clicked_tooltip"]
    )
    st.session_state.selected_origin = facility_id
```
Marker tooltip must encode `facility_id` as the tooltip string.

**Gap 3 — manifest.json schema**
```json
{"random_seed": 42, "data_hash": "<sha256 of facilities.csv+cdst_labs.csv>", "created": "2026-06-03"}
```

### Architecture Completeness Checklist

- [x] Project context analyzed, scale assessed, constraints identified
- [x] All critical decisions documented, tech stack specified
- [x] Naming, structure, communication, and process patterns defined
- [x] Complete directory tree, boundaries, integration points, FR mapping

### Architecture Readiness Assessment

**Overall Status:** READY WITH MINOR GAPS (all gaps resolved above)
**Confidence Level:** High

**Key Strengths:**
- All 5 NFRs fully addressed
- Demo climax (stockout simulation) has explicit state model and data flow
- 6 Amelia-flagged implementation traps resolved in patterns
- Module interface contracts prevent agent divergence on return shapes
- Air-gapped demo resilience built in (cache + artifacts committed)

**Post-MVP enhancements (see addendum.md):** federated learning, Neo4j, Kafka, full SHAP dashboard

## Implementation Patterns & Consistency Rules

### Naming Conventions

**Python code — no exceptions:**
- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants (in `src/config.py`): `UPPER_SNAKE_CASE`
- Module files: `snake_case.py`
- Session state keys: `snake_case` strings

**CSV column names:**
- All lowercase `snake_case`: `facility_id`, `facility_name`, `lat`, `lon`, `tests_per_month`, `cartridge_availability`, `assigned_cdst_id`, `anomaly_score`
- CDST: `cdst_id`, `cdst_name`, `lat`, `lon`, `monthly_capacity`, `current_load`

**Artefact filenames:** always seed-encoded: `isolation_forest_seed{N}.joblib`, `qtable_seed{N}.npy`

**Canonical session state keys (agents must not invent new keys):**
```python
st.session_state.initialised          # bool
st.session_state.travel_matrix        # np.ndarray (38×5)
st.session_state.graph                # nx.DiGraph
st.session_state.iso_forest           # sklearn IsolationForest
st.session_state.qtable               # np.ndarray
st.session_state.shap_explainer       # shap.Explainer
st.session_state.selected_origin      # str | None (facility_id)
st.session_state.stockout_overrides   # dict[str, int]
st.session_state.ors_fallback_active  # bool
```

### Import Pattern

All imports are absolute from project root. Never use relative imports.

```python
# CORRECT
from src.config import RANDOM_SEED, ALPHA, BETA, GAMMA
from src.graph.builder import build_graph

# WRONG
from ..config import RANDOM_SEED
```

### Module Public Interface Contracts

```python
# src/data/ors_client.py
def load_or_fetch_ors_matrix(facilities_df, cdst_df) -> tuple[np.ndarray, bool]:
    # Returns (matrix, is_fallback)

# src/graph/builder.py
def build_graph(facilities_df, cdst_df, travel_matrix,
                overrides: dict[str, int] | None = None) -> nx.DiGraph:

# src/routing/dijkstra.py
def run_dijkstra(graph, origin_id) -> dict:
    # Returns {"cdst_id": str, "score": float, "path": list[str]}

# src/routing/qlearning.py
def run_qlearning(qtable, origin_id, facilities_df, cdst_df) -> dict:
    # Returns {"cdst_id": str, "q_value": float}

# src/explainability/shap_wrapper.py
def explain_routing(origin_id, recommended_cdst_id, graph, background_data) -> dict:
    # Returns {"shap_values": np.ndarray, "feature_names": list[str], "explanation_text": str}

# src/models/anomaly.py
def score_facilities(model, facilities_df) -> pd.Series:
    # Returns Series indexed by facility_id, values 0.0–1.0
```

### Error Handling Patterns

- External I/O: always `try/except` with specific exception types, never bare `Exception`
- Missing artefact files: raise `FileNotFoundError` with clear message; `app.py` catches and calls `st.error()`
- Seed mismatch: raise `ValueError`; never silently continue

### Loading State Pattern

```python
with st.status("Initialising DiagNet...", expanded=True) as status:
    st.write("Loading travel matrix...")
    travel_matrix, fallback = load_or_fetch_ors_matrix(...)
    st.write("Building diagnostic network graph...")
    graph = build_graph(...)
    st.write("Loading AI models...")
    iso_forest, qtable = load_models(...)
    status.update(label="Ready", state="complete")
```

### Enforcement Rules

**All agents MUST:**
- Import constants from `src.config` — never hardcode values inline
- Use only canonical session_state keys
- Return exact dict shapes from interface contracts
- Use `logging.getLogger(__name__)` — never `print()` for debug
- Wrap all file I/O in typed try/except

**Anti-patterns:**
- `except Exception:` anywhere in `src/`
- Hardcoded paths or seeds outside `src/config.py`
- Mutating `graph` after `build_graph()` returns
- Calling ORS API outside `src/data/ors_client.py`

## Project Structure & Boundaries

### Complete Project Directory Tree

```
DNO_advanced/
├── app.py                               # FR-U1–U4: Streamlit entry point, session_state
│                                        #   init gate, all panel layout, stockout button
├── requirements.txt
├── .env                                 # ORS_API_KEY (gitignored)
├── .env.example
├── .gitignore
│
├── data/                                # FR-D1, FR-D2: Source of truth CSVs
│   ├── facilities.csv                   # 38 district facility records
│   └── cdst_labs.csv                    # 5 CDST lab records
│
├── artifacts/                           # Pre-trained model artefacts (committed)
│   ├── isolation_forest_seed42.joblib   # FR-A1: pre-trained anomaly model
│   ├── qtable_seed42.npy                # FR-R3: pre-trained Q-table
│   └── manifest.json                    # data hash + seed — validated at startup
│
├── cache/                               # FR-D3: ORS matrix cache (committed)
│   └── ors_matrix.npy                  # 38×5 travel time matrix (minutes)
│
├── src/
│   ├── config.py                        # ALL constants: RANDOM_SEED, α/β/γ,
│   │                                    #   paths, ORS URL, CDST IDs
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                    # FR-D1, D2: pd.read_csv wrappers
│   │   └── ors_client.py               # FR-D3: ORS API + Haversine fallback
│   ├── graph/
│   │   ├── __init__.py
│   │   └── builder.py                  # FR-G1, G2, R1: build_graph() with overrides
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── dijkstra.py                 # FR-R2: run_dijkstra() → {cdst_id, score, path}
│   │   └── qlearning.py               # FR-R3: run_qlearning() → {cdst_id, q_value}
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py                   # load_models() — validates manifest, raises on mismatch
│   │   └── anomaly.py                  # FR-A1–A3: score_facilities() → pd.Series
│   └── explainability/
│       ├── __init__.py
│       └── shap_wrapper.py             # FR-S1–S3: explain_routing() → {shap_values, text}
│
├── scripts/                            # Offline-only — never imported by app.py
│   ├── generate_data.py                # FR-D4: synthetic data + manifest
│   └── train_models.py                 # FR-A1, R3: trains IF + Q-table → artifacts/
│
└── tests/
    ├── test_ors_client.py              # ORS fallback when network unavailable
    ├── test_graph_builder.py           # shape, weights, overrides
    ├── test_dijkstra.py                # correct CDST for known inputs
    ├── test_qlearning.py               # Q-table lookup returns valid CDST id
    ├── test_model_loader.py            # seed mismatch → ValueError; missing → FileNotFoundError
    ├── test_anomaly.py                 # Series with correct index
    └── test_shap_wrapper.py            # latency <1s; non-zero values; explanation_text non-empty
```

### Functional Requirements → File Mapping

| FR | File |
|---|---|
| FR-D1 (facility data) | `data/facilities.csv`, `src/data/loader.py`, `scripts/generate_data.py` |
| FR-D2 (CDST data) | `data/cdst_labs.csv`, `src/data/loader.py`, `scripts/generate_data.py` |
| FR-D3 (ORS matrix) | `src/data/ors_client.py`, `cache/ors_matrix.npy` |
| FR-D4 (data generator) | `scripts/generate_data.py` |
| FR-G1, G2 (graph) | `src/graph/builder.py` |
| FR-R1 (weight fn) | `src/graph/builder.py`, `src/config.py` |
| FR-R2 (Dijkstra) | `src/routing/dijkstra.py` |
| FR-R3 (Q-learning) | `src/routing/qlearning.py`, `artifacts/qtable_seed42.npy`, `scripts/train_models.py` |
| FR-R4 (comparison) | `app.py` (results panel) |
| FR-A1–A3 (anomaly) | `src/models/anomaly.py`, `artifacts/isolation_forest_seed42.joblib`, `scripts/train_models.py` |
| FR-S1–S3 (SHAP) | `src/explainability/shap_wrapper.py` |
| FR-U1–U4 (UI) | `app.py` |

### Data Flow

```
[scripts/generate_data.py]
        │ writes
        ▼
[data/facilities.csv] [data/cdst_labs.csv] [artifacts/manifest.json]
        │
        ▼ app.py startup (session_state gate)
[src/data/ors_client.py] ──▶ ORS API (or Haversine fallback)
        │ writes ▶ cache/ors_matrix.npy
[src/graph/builder.py]   ──▶ nx.DiGraph → session_state.graph
[src/models/loader.py]   ──▶ models    → session_state.iso_forest / qtable
        │
        ▼ user selects origin
[src/routing/dijkstra.py]       ──▶ {cdst_id, score, path}   ──▶ app.py results panel
[src/routing/qlearning.py]      ──▶ {cdst_id, q_value}
[src/explainability/shap_wrapper] ─▶ {shap_values, text}
        │
        ▼ Simulate Stockout button (demo climax)
[st.session_state.stockout_overrides updated]
[build_graph(..., overrides=...) re-called]
[routing re-run → rerouted recommendation surfaced]
```

### Integration Boundaries

| Boundary | Location | Rule |
|---|---|---|
| ORS API | `src/data/ors_client.py` | Only file that touches external network |
| Artefact I/O | `src/models/loader.py` | Only file that reads from `artifacts/` |
| Session state writes | `app.py` only | No `src/` module writes to session_state |
| Graph mutation | Prohibited post-build | `build_graph()` returns immutable DiGraph |
| Script isolation | `scripts/` only | Never imported by `src/` or `app.py` |

## Core Architectural Decisions

### Already Decided (from PRD + Steps 2–3)
- Language/runtime: Python 3.12, venv-isolated
- UI: Streamlit + Folium + streamlit-folium
- Graph: NetworkX in-memory, immutable after build
- Routing: Dijkstra (networkx.dijkstra_path) + tabular Q-learning (pre-trained)
- Anomaly: Isolation Forest (scikit-learn, pre-trained, .joblib)
- Explainability: SHAP Permutation explainer wrapping edge-weight callable
- Serialisation: .joblib (IF), .npy (Q-table), seed-encoded filenames
- Deployment: local only, `streamlit run app.py`

### Decision 1 — Constants & Configuration
- **Decision:** `src/config.py` as single source of truth
- **Contents:** RANDOM_SEED = 42, α/β/γ weight defaults, ORS base URL, artefact paths, data paths
- **Rationale:** Single import prevents constant drift across modules

### Decision 2 — Startup Caching Strategy
- **Decision:** `@st.cache_data` + `session_state` init gate (complementary)
- `@st.cache_data` on `load_or_fetch_ors_matrix()` — survives reruns
- `session_state` gate in `app.py` — prevents graph/model re-init on widget interaction
- **Rationale:** Each mechanism solves a different rerun problem

### Decision 3 — ORS Cache Format
- **Decision:** `.npy` binary (numpy)
- **Path:** `cache/ors_matrix.npy`
- **Rationale:** Consistent with Q-table format; fast load; committed to repo for air-gapped demo resilience

### Decision 4 — Stockout Simulation State
- **Decision:** Override dict in session_state
- `st.session_state.stockout_overrides: dict[str, int]` — maps CDST ID to cartridge override value
- Weight function accepts optional `overrides` kwarg; graph object never mutated
- Reset: `st.session_state.stockout_overrides = {}` on Reset button click
- **Rationale:** Lightweight, explicit, reproducible mid-session

### Decision 5 — Error Display Strategy
- `st.warning()` — persistent states (ORS fallback active, seed mismatch)
- `st.toast()` — transient events (route computed, stockout applied)
- Python `logging` — debug traces to terminal only, never surfaced in UI
- **Rationale:** Judges see resilience signals without UI clutter

### Deferred (Post-MVP)
- CI/CD, containerisation, authentication, multi-user — all deferred per PRD scope
