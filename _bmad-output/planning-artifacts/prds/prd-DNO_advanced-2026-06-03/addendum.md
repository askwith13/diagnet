# Addendum — DiagNet MVP PRD

Items below were present in the original enterprise PRD (v1.1) and deliberately deferred from MVP scope. They belong in a future production PRD, not the current document.

---

## Deferred: Production Architecture

- **Federated learning:** Flower (`flwr`) + Opacus DP-SGD, state-level gradient exchange, FedAvg central aggregator
- **Production graph store:** Neo4j + Graph Data Science (GDS) library, nightly Cypher-based graph rebuild
- **Streaming pipelines:** Apache Kafka + Celery task queue for real-time HMIS/LIS log ingestion
- **Caching layer:** Redis for MCMC output and matrix pre-computation
- **Production API:** FastAPI backend + React.js + Mapbox GL frontend

## Deferred: Advanced ML Layers

- **Bayesian Hierarchical Stockout Forecaster (B1):** PyMC / CmdStanPy — posterior predictive P(stock_{t+14} ≤ 0 | data) > 0.70 alert trigger
- **Bayesian Network Cascade Dropout (B2):** pgmpy DAG over specimen quality, machine uptime, technician coverage
- **LSTM Autoencoder (M1 Stage 2):** Sequential LOINC-code pathway anomaly detection
- **GNN Facility Embeddings (M3):** PyTorch Geometric, combining static attributes with topology vectors

## Deferred: Data Ingestion

- NIKSHAY API, DHIS2 Registry, OpenELIS, OpenMRS, KoBoToolbox connectors
- SHA-256 pseudonymization pipeline at ingestion edge
- Offline fallback (edge server containers for block-level PHIs without internet)

## Deferred: Compliance & Operational

- DPDP Act (Digital Personal Data Protection) compliance
- TLS 1.3 gradient transmission
- WhatsApp Business API alert delivery
- Multi-user authentication and per-action audit trail

## Options Considered: UI Framework

- **FastAPI + React** (dropped for MVP): production-grade, but requires separate frontend build process
- **Gradio** (considered): simpler than Streamlit but weaker map support
- **Streamlit + Folium** (chosen): zero build step, rich map rendering, sufficient for demo interactions

## Options Considered: Graph Backend

- **Neo4j** (deferred): production-grade Cypher querying, but requires Docker or install; overkill for 43-node demo graph
- **NetworkX** (chosen for MVP): pure Python, in-memory, trivially installable, sufficient for 38+5 node graph
