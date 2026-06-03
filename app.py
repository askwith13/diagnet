import logging

import folium
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

from src.config import ALPHA, BETA, GAMMA, DELTA, CDST_IDS, CDST_NAMES
from src.data.loader import load_facilities, load_cdst_labs
from src.data.ors_client import load_or_fetch_ors_matrix
from src.explainability.shap_wrapper import explain_routing
from src.graph.builder import build_graph
from src.models.anomaly import score_facilities
from src.models.loader import load_models
from src.routing.dijkstra import run_dijkstra
from src.routing.qlearning import run_qlearning

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="DiagNet — TB Diagnostic Network", layout="wide")


def _anomaly_color(score: float) -> str:
    r = int(score * 231 + (1 - score) * 46)
    g = int(score * 76 + (1 - score) * 204)
    b = int(score * 60 + (1 - score) * 113)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_map(facilities_df, cdst_df, anomaly_scores,
               selected_origin=None, graph=None, dijkstra_result=None):
    m = folium.Map(location=[23.5, 87.8], zoom_start=7, tiles="CartoDB positron")

    for _, row in facilities_df.iterrows():
        fid = row["facility_id"]
        score = float(anomaly_scores.get(fid, 0.0))
        color = _anomaly_color(score)
        is_selected = fid == selected_origin
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=11 if is_selected else 7,
            color="#ffffff" if is_selected else "#333333",
            weight=3 if is_selected else 0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=fid,
            popup=folium.Popup(
                f"<b>{row['facility_name']}</b><br>"
                f"Tests/month: {row['tests_per_month']}<br>"
                f"Cartridges: {row['cartridge_availability']}<br>"
                f"Anomaly: {score:.2f}",
                max_width=220,
            ),
        ).add_to(m)

    for _, row in cdst_df.iterrows():
        is_stockout = (
            row["cdst_id"] in st.session_state.stockout_overrides
            and st.session_state.stockout_overrides[row["cdst_id"]] == 0
        )
        folium.Marker(
            location=[row["lat"], row["lon"]],
            tooltip=row["cdst_id"],
            popup=folium.Popup(
                f"<b>{row['cdst_name']}</b><br>"
                f"Capacity: {row['monthly_capacity']}/month<br>"
                f"Load: {row['current_load']}"
                + (" ⚠ STOCKOUT" if is_stockout else ""),
                max_width=220,
            ),
            icon=folium.Icon(
                color="red" if is_stockout else "blue",
                icon="flask",
                prefix="fa",
            ),
        ).add_to(m)

    if dijkstra_result and selected_origin and graph:
        path = dijkstra_result.get("path", [])
        if len(path) >= 2:
            coords = [[graph.nodes[n]["lat"], graph.nodes[n]["lon"]] for n in path]
            folium.PolyLine(
                coords, color="#e74c3c", weight=3.5, opacity=0.85, dash_array="8 4"
            ).add_to(m)

    return m


def _route_comparison_df(graph, origin_id, cdst_df, dijkstra_cdst, ql_cdst):
    rows = []
    for _, cdst_row in cdst_df.iterrows():
        cid = cdst_row["cdst_id"]
        if not graph.has_edge(origin_id, cid):
            continue
        e = graph.edges[origin_id, cid]
        tag = ""
        if cid == dijkstra_cdst and cid == ql_cdst:
            tag = "★ Optimal · Q agrees"
        elif cid == dijkstra_cdst:
            tag = "★ Optimal"
        elif cid == ql_cdst:
            tag = "Q Policy"
        rows.append({
            "Lab": CDST_NAMES[cid],
            "Travel": f"{e['travel_min']:.0f} min",
            "Queue": f"{e['norm_queue']*100:.0f}%",
            "Cartridge": f"{(1 - e['norm_cart_unavail'])*100:.0f}%",
            "Cost": round(e["weight"], 3),
            "": tag,
            "_cid": cid,
        })
    rows.sort(key=lambda r: r["Cost"])
    return rows


def _shap_figure(shap_values, travel_min):
    labels = [
        f"Travel ({travel_min:.0f} min)",
        "Queue load",
        "Cartridge avail.",
        "Anomaly flag",
    ]
    colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in shap_values]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.barh(labels, shap_values, color=colors, height=0.5, edgecolor="none")

    for bar, val in zip(bars, shap_values):
        ax.text(
            val + (0.003 if val >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
            color="#333333",
        )

    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_xlabel(
        "← reduces routing cost (favorable)          increases routing cost (unfavorable) →",
        fontsize=8,
        color="#555555",
    )
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("What drove this recommendation", fontsize=11, pad=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Startup: session_state init gate ─────────────────────────────────────────
if "initialised" not in st.session_state:
    with st.status("Initialising DiagNet...", expanded=True) as status:
        st.write("Loading facility data...")
        facilities_df = load_facilities()
        cdst_df = load_cdst_labs()

        st.write("Loading travel matrix...")
        travel_matrix, fallback = load_or_fetch_ors_matrix(facilities_df, cdst_df)
        st.session_state.travel_matrix = travel_matrix
        st.session_state.ors_fallback_active = fallback

        st.write("Loading AI models...")
        iso_forest, qtable, shap_explainer = load_models(facilities_df, cdst_df, travel_matrix)
        st.session_state.iso_forest = iso_forest
        st.session_state.qtable = qtable
        st.session_state.shap_explainer = shap_explainer

        st.write("Scoring facility anomalies...")
        st.session_state.anomaly_scores = score_facilities(iso_forest, facilities_df)

        st.write("Building diagnostic network graph...")
        st.session_state.graph = build_graph(
            facilities_df, cdst_df, travel_matrix,
            anomaly_scores=st.session_state.anomaly_scores,
        )
        st.session_state.graph_key = (ALPHA, BETA, GAMMA, DELTA, "")

        st.session_state.facilities_df = facilities_df
        st.session_state.cdst_df = cdst_df
        st.session_state.selected_origin = None
        st.session_state.stockout_overrides = {}
        st.session_state.initialised = True
        status.update(label="DiagNet ready", state="complete")

# Convenience aliases
facilities_df = st.session_state.facilities_df
cdst_df = st.session_state.cdst_df
travel_matrix = st.session_state.travel_matrix
anomaly_scores = st.session_state.anomaly_scores
qtable = st.session_state.qtable
shap_explainer = st.session_state.shap_explainer

if st.session_state.ors_fallback_active:
    st.warning("Travel times estimated via Haversine (ORS API unavailable — add key to .env)")

st.title("DiagNet — TB Diagnostic Network Intelligence")

# ── Top row: controls | map | winner cards ────────────────────────────────────
col_sidebar, col_map, col_winner = st.columns([1, 2.5, 1])

with col_sidebar:
    st.subheader("Controls")

    fac_names = ["— select facility —"] + facilities_df["facility_name"].tolist()
    fac_id_to_name = dict(zip(facilities_df["facility_id"], facilities_df["facility_name"]))
    current_origin = st.session_state.selected_origin
    default_idx = (
        fac_names.index(fac_id_to_name[current_origin])
        if current_origin and current_origin in fac_id_to_name
        else 0
    )
    fac_name_sel = st.selectbox("Origin facility", fac_names, index=default_idx)
    if fac_name_sel != "— select facility —":
        sel_fac_id = facilities_df.loc[
            facilities_df["facility_name"] == fac_name_sel, "facility_id"
        ].values[0]
        st.session_state.selected_origin = sel_fac_id
    elif default_idx == 0:
        st.session_state.selected_origin = None

    st.markdown("**Routing weights**")
    alpha = st.slider("Travel time (α)", 0.0, 1.0, float(ALPHA), 0.05)
    beta  = st.slider("Queue load (β)",  0.0, 1.0, float(BETA),  0.05)
    gamma = st.slider("Cartridge (γ)",   0.0, 1.0, float(GAMMA), 0.05)
    delta = st.slider("Anomaly (δ)",     0.0, 1.0, float(DELTA), 0.05)

    st.markdown("---")
    st.markdown("**Simulate Stockout**")
    sim_options = ["— none —"] + CDST_IDS
    sim_cdst = st.selectbox(
        "CDST lab",
        sim_options,
        format_func=lambda x: x if x.startswith("—") else f"{x}: {CDST_NAMES[x]}",
    )
    col_sim, col_rst = st.columns(2)
    with col_sim:
        if st.button("Apply", disabled=(sim_cdst.startswith("—"))):
            st.session_state.stockout_overrides = {sim_cdst: 0}
    with col_rst:
        if st.button("Reset"):
            st.session_state.stockout_overrides = {}

    if st.session_state.stockout_overrides:
        cdst_key = next(iter(st.session_state.stockout_overrides))
        st.error(f"⚠ Stockout: {CDST_NAMES.get(cdst_key, cdst_key)}")

    st.markdown("---")
    st.markdown(
        "<small>🔴 high anomaly &nbsp; 🟢 normal &nbsp; 🔵 CDST lab</small>",
        unsafe_allow_html=True,
    )

# ── Graph rebuild on weight/override changes ──────────────────────────────────
graph_key = (alpha, beta, gamma, delta, str(sorted(st.session_state.stockout_overrides.items())))
if st.session_state.get("graph_key") != graph_key:
    st.session_state.graph = build_graph(
        facilities_df, cdst_df, travel_matrix,
        anomaly_scores=anomaly_scores,
        overrides=st.session_state.stockout_overrides,
        delta=delta,
    )
    st.session_state.graph_key = graph_key

graph = st.session_state.graph

origin_id = st.session_state.selected_origin
dijkstra_result = None
ql_result = None
if origin_id:
    try:
        dijkstra_result = run_dijkstra(graph, origin_id)
        ql_result = run_qlearning(qtable, origin_id, facilities_df, cdst_df)
    except Exception:
        pass

# ── Map ───────────────────────────────────────────────────────────────────────
with col_map:
    st.subheader("West Bengal TB Diagnostic Network")
    m = _build_map(
        facilities_df, cdst_df, anomaly_scores,
        selected_origin=origin_id,
        graph=graph,
        dijkstra_result=dijkstra_result,
    )
    map_data = st_folium(m, key="wb_map", width="100%", height=520)

    if map_data and map_data.get("last_object_clicked_tooltip"):
        tooltip = map_data["last_object_clicked_tooltip"]
        if tooltip in set(facilities_df["facility_id"]) and tooltip != st.session_state.selected_origin:
            st.session_state.selected_origin = tooltip
            st.rerun()

# ── Winner cards (top-right) ──────────────────────────────────────────────────
with col_winner:
    st.subheader("Recommendation")
    if not origin_id:
        st.info("Select a facility on the map or from the sidebar.")
    elif dijkstra_result is None:
        st.error("No path found.")
    else:
        dij_cdst = dijkstra_result["cdst_id"]
        ql_cdst = ql_result["cdst_id"]

        st.markdown(f"**From:** {fac_id_to_name.get(origin_id, origin_id)}")

        st.markdown("**Dijkstra — optimal**")
        st.success(f"→ {CDST_NAMES[dij_cdst]}")
        st.caption(f"Cost: {dijkstra_result['score']:.3f}")

        st.markdown("**Q-Learning — AI policy**")
        if ql_cdst == dij_cdst:
            st.success(f"→ {CDST_NAMES[ql_cdst]} ✓")
        else:
            st.warning(f"→ {CDST_NAMES[ql_cdst]}")
        st.caption(f"Q-value: {ql_result['q_value']:.4f}")

# ── Hero section: SHAP + route table ─────────────────────────────────────────
if origin_id and dijkstra_result:
    st.divider()
    dij_cdst = dijkstra_result["cdst_id"]
    ql_cdst = ql_result["cdst_id"]

    col_shap, col_table = st.columns([3, 2])

    with col_shap:
        st.markdown(f"## Why **{CDST_NAMES[dij_cdst]}**?")

        if shap_explainer is not None:
            try:
                expl = explain_routing(origin_id, dij_cdst, graph, shap_explainer)
                st.info(expl["explanation_text"])

                edge = graph.edges[origin_id, dij_cdst]
                fig = _shap_figure(expl["shap_values"], edge["travel_min"])
                st.pyplot(fig)
                plt.close(fig)

            except Exception as exc:
                st.warning(f"SHAP explanation unavailable: {exc}")
        else:
            st.warning("SHAP explainer not loaded.")

    with col_table:
        st.markdown("## All routes ranked")
        rows = _route_comparison_df(graph, origin_id, cdst_df, dij_cdst, ql_cdst)

        display_rows = [
            {k: v for k, v in r.items() if k != "_cid"}
            for r in rows
        ]
        df = pd.DataFrame(display_rows)

        def _highlight(row):
            is_opt = "★" in str(row.get("", ""))
            is_ql = row.get("", "") == "Q Policy"
            if is_opt:
                return ["background-color: #d4edda; color: black; font-weight: bold"] * len(row)
            if is_ql:
                return ["background-color: #fff3cd; color: black"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df.style.apply(_highlight, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "★ Dijkstra optimal &nbsp;·&nbsp; Q = Q-Learning policy &nbsp;·&nbsp; "
            "Cost = α·travel + β·queue + γ·cartridge",
            unsafe_allow_html=True,
        )
