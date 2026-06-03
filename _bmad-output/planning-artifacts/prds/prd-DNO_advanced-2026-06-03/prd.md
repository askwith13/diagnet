---
title: DiagNet — MVP Demo PRD
status: draft
version: 2.0
created: 2026-06-03
updated: 2026-06-03
hackathon: AWARE AI Hackathon 2026
---

# DiagNet — Diagnostic Network Intelligence (MVP Demo)

**Theme:** Healthcare Professionals Assistance & Community Engagement — AWARE AI Hackathon 2026  
**Narrative Context:** West Bengal TB Diagnostic Network / NTEP  
**Deployment Target:** Single local machine (laptop demo / video walkthrough)

---

## 1. Vision

India's TB diagnostic cascade loses patients invisibly — not because facilities refuse them, but because no one can see in real time which lab has cartridges, which is overwhelmed, or which is geographically reachable within a day. **DiagNet** makes the network legible and gives frontline workers an AI-powered routing recommendation they can act on immediately.

The MVP demonstrates one high-value capability end-to-end: **intelligent patient routing** — showing a clinician the best CDST lab to refer a patient to, adapting in real time to cartridge availability, queue load, and travel time, with a plain-language AI explanation for every recommendation.

**Demo goal:** A live or recorded workflow where a clinician selects an origin facility, DiagNet computes the optimal referral route across West Bengal's 38 district facilities and 5 CDST labs, and explains *why* in two sentences — all running on a laptop with no internet dependency beyond the ORS API call.

---

## 2. Scope Boundaries

| In scope | Out of scope |
|---|---|
| West Bengal simulated network (38 district labs, 5 CDST labs) | Multi-state / federated deployment |
| Synthetic data generation + real ORS travel times | Live production data ingestion (NIKSHAY, DHIS2, OpenELIS) |
| In-memory graph (NetworkX) | Neo4j, Apache Kafka, Celery, Redis |
| Dijkstra static routing + tabular Q-learning adaptive layer | PyTorch Geometric GNN, LSTM Autoencoder, PyMC Bayesian hierarchy |
| Isolation Forest anomaly signal (routing input) | Full anomaly detection pipeline |
| SHAP explainability on routing decisions | SHAP waterfall dashboard for all models |
| Streamlit UI with interactive map | FastAPI backend, React.js frontend |
| Single-user local session | Multi-user, authentication, audit trails |
| Demo / hackathon context | DPDP Act compliance, TLS, DP guarantees |

---

## 3. Data Layer

### 3.1 Simulated Network — West Bengal

The system ships with a pre-generated synthetic dataset representing a plausible West Bengal TB diagnostic network. All coordinates fall within West Bengal's geographic bounding box; facility names follow district-level naming conventions for narrative realism.

**FR-D1** — District facility records (38 rows):

| Field | Description |
|---|---|
| `facility_id` | Unique identifier |
| `facility_name` | District-level label (e.g., "Murshidabad DHC") |
| `lat`, `lon` | Within West Bengal bounds |
| `tests_per_month` | Integer, 20–300 (realistic volume range) |
| `cartridge_availability` | Integer, 0–150 (units in stock) |
| `assigned_cdst_id` | Currently assigned CDST lab ID |
| `anomaly_score` | Float, pre-computed by Isolation Forest |

**FR-D2** — CDST lab records (5 rows):

| Field | Description |
|---|---|
| `cdst_id` | Unique identifier |
| `cdst_name` | See confirmed labs below |
| `lat`, `lon` | Within West Bengal bounds |
| `monthly_capacity` | Max culture+DST cases per month |
| `current_load` | Cases currently in queue |

**Confirmed CDST labs (real institutions):**

| ID | Name | Region |
|---|---|---|
| CDST-01 | Kolkata CDST Lab | Southern / State Reference |
| CDST-02 | NBMCH CDST Lab | North Bengal (Siliguri) |
| CDST-03 | Midnapore MCH CDST Lab | South-West Bengal |
| CDST-04 | Murshidabad MCH CDST Lab | Central-West Bengal |
| CDST-05 | Bardhamaan MCH CDST Lab | Central Bengal |

> CDST-01 confirmed as **RNTCP State Reference Laboratory, Kolkata**.

**FR-D3** — Travel time matrix: An ORS API call pre-computes a 38×5 matrix of road-network travel times (minutes) at startup. Results are cached locally to avoid re-calling during the demo. If ORS is unreachable, the system falls back to Haversine-derived estimates with a visible warning in the UI.

**FR-D4** — Data generator script: A standalone `generate_data.py` script reproducibly creates the full synthetic dataset from a fixed random seed. Aswath's real sample files can override specific rows; the generator fills the rest.

---

## 4. Functional Requirements

### 4.1 Graph Construction

**FR-G1** — On startup, build a directed weighted graph in NetworkX with:
- **Nodes:** All 38 district facilities + 5 CDST labs
- **Edges:** District facility → CDST lab, weighted by a composite score (see FR-R1)

**FR-G2** — Graph is rebuilt in-memory on each demo session start. No persistent graph store required.

### 4.2 Routing Intelligence

The routing layer is the demo centrepiece. It operates in two complementary modes that together tell the AI story.

**FR-R1** — Static composite edge weight:

```
weight = α × normalised_travel_time
       + β × normalised_queue_load
       + γ × (1 − normalised_cartridge_availability)
```

Default: α=0.5, β=0.3, γ=0.2. Weights are configurable in the UI sidebar for live exploration during the demo.

**FR-R2** — Dijkstra shortest-path over the weighted graph identifies the globally optimal CDST lab for a given origin facility.

**FR-R3** — Tabular Q-learning adaptive agent:
- State: (origin facility, cartridge_availability bucket, queue_load bucket, time_of_week)
- Actions: {refer to CDST-1, CDST-2, CDST-3, CDST-4, CDST-5}
- Reward: inversely proportional to realised composite cost (simulated)
- The agent is pre-trained offline on the synthetic dataset and ships with learned Q-tables. During the demo, the UI shows the agent's recommended action alongside the Dijkstra result — agreement/disagreement is a talking point.

**FR-R4** — If Dijkstra and Q-learning disagree on the top recommendation, the UI surfaces both options with their respective scores and lets the clinician choose.

### 4.3 Anomaly Signal

**FR-A1** — An Isolation Forest model is pre-trained on facility-level features (tests_per_month, cartridge_availability, queue deviation from district mean). It produces a continuous anomaly score per facility.

**FR-A2** — Anomaly score is surfaced as a colour-coded ring on each facility node in the map (green → red). Highly anomalous facilities trigger a soft warning banner if selected as origin.

**FR-A3** — Anomaly score feeds into the routing weight as an optional fourth term (configurable, off by default) to down-rank referrals to anomalous CDST labs.

### 4.4 SHAP Explainability

**FR-S1** — Every routing recommendation is accompanied by a SHAP decomposition explaining which factors drove the choice.

**FR-S2** — The UI renders a compact SHAP bar chart (top 4 features) alongside a generated two-line plain-language explanation:

> *"Kolkata CDST recommended: cartridge availability (+0.41) and queue capacity (+0.29) outweigh the longer travel time (−0.18) compared to the closer Howrah lab."*

**FR-S3** — SHAP values are computed on-the-fly from the composite weight model, not a separate model. The explainer wraps the edge-weight function as a callable and uses `shap.Explainer` for consistency.

### 4.5 Streamlit UI

**FR-U1** — Single-page Streamlit application with three panels:

| Panel | Content |
|---|---|
| **Left sidebar** | Origin facility selector (dropdown), α/β/γ weight sliders, "Run Routing" button |
| **Centre — Map** | Folium map of West Bengal; facilities colour-coded by anomaly score; selected route highlighted with directional arrow; CDST labs marked distinctly |
| **Right — Results** | Recommended CDST lab card (name, travel time, queue load, cartridge stock), Q-learning vs. Dijkstra comparison row, SHAP bar chart, two-line plain-language explanation |

**FR-U2** — Map interactions: clicking a facility on the map selects it as origin and triggers routing automatically.

**FR-U3** — A "Simulate Stockout" button zeroes out cartridge availability at a chosen CDST lab and re-runs routing to show dynamic rerouting — a high-impact demo moment.

**FR-U4** — All rendering must complete within 3 seconds on a standard laptop after the initial ORS matrix call.

---

## 5. Non-Functional Requirements

**NFR-1 — Local-only runtime:** The full system runs with `streamlit run app.py`. No Docker, no cloud services, no background daemons beyond the Streamlit process.

**NFR-2 — Reproducibility:** The synthetic dataset and Q-table are generated from a fixed seed. The demo produces identical results on any machine.

**NFR-3 — Graceful ORS fallback:** If the OpenRouteService API is unavailable, the system uses Haversine estimates and displays a visible disclaimer. The demo must never crash due to API failure.

**NFR-4 — Dependency footprint:** All dependencies installable via a single `pip install -r requirements.txt`. No Conda, no compiled extensions beyond standard scientific Python.

**NFR-5 — Demo resilience:** The "Simulate Stockout" flow and weight slider interactions must never require a page reload. State updates are handled via Streamlit session state.

---

## 6. Open Questions

| # | Question | Owner | Blocking? |
|---|---|---|---|
| ~~OQ-1~~ | ~~Which West Bengal districts should host the 5 CDST labs?~~ | ~~Aswath~~ | Resolved — see FR-D2 |
| ~~OQ-2~~ | ~~Q-learning re-trainable live or pre-trained only?~~ | ~~Aswath~~ | Resolved — pre-trained only |
| ~~OQ-3~~ | ~~Is OpenRouteService API key already available?~~ | ~~Aswath~~ | Resolved — key available, stored in `.env` as `ORS_API_KEY` |

---

## 7. Success Criteria

The MVP is complete when a judge can watch a 3–5 minute demo (live or recorded) that shows:

1. A West Bengal facility map loading with colour-coded anomaly signals
2. A clinician selecting an origin facility and receiving a CDST routing recommendation within 3 seconds
3. A plain-language SHAP explanation accompanying the recommendation
4. The "Simulate Stockout" flow triggering a visible reroute
5. The Q-learning vs. Dijkstra comparison surfaced side-by-side
