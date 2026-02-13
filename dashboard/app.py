"""
Streamlit Dashboard for Hospital Strategy--Action Plan Alignment System (ISPS).

A multi-page interactive dashboard that visualises synchronization analysis,
improvement recommendations, knowledge graph structure, ontology mappings,
agentic AI reasoning traces, and evaluation metrics for Nawaloka Hospital
Negombo's strategy--action alignment.

Launch::

    streamlit run dashboard/app.py

Author : ISPS Team
Created: 2025-01
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils import (
    export_data,
    format_llm_response,
    generate_pdf_report,
    load_analysis_results,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Healthcare blue/green theme */
    :root {
        --primary: #1565C0;
        --primary-light: #42A5F5;
        --secondary: #2E7D32;
        --secondary-light: #66BB6A;
        --accent: #FF8F00;
        --bg-card: #F8FAFB;
        --text-dark: #1A237E;
    }
    .main .block-container { max-width: 1200px; padding-top: 1.5rem; }
    h1 { color: var(--primary) !important; }
    h2, h3 { color: var(--text-dark) !important; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #E3F2FD 0%, #F1F8E9 100%);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-card h4 { margin: 0 0 0.3rem 0; color: var(--primary); font-size: 0.85rem; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: var(--text-dark); }
    .metric-card .sub { font-size: 0.75rem; color: #666; }

    /* Issue badges */
    .badge-orphan  { background: #FFCDD2; color: #B71C1C; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-weak    { background: #FFE0B2; color: #E65100; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-gap     { background: #E1BEE7; color: #6A1B9A; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-good    { background: #C8E6C9; color: #1B5E20; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .badge-high    { background: #C8E6C9; color: #1B5E20; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-medium  { background: #FFF9C4; color: #F57F17; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-low     { background: #FFCDD2; color: #B71C1C; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }

    /* Trace step */
    .trace-step {
        border-left: 3px solid var(--primary-light);
        padding: 0.5rem 0 0.5rem 1rem;
        margin-bottom: 0.8rem;
        background: var(--bg-card);
        border-radius: 0 8px 8px 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0D47A1 0%, #1565C0 100%); }
    [data-testid="stSidebar"] .css-1d391kg { color: white; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #E3F2FD !important; }

    /* Loading indicator */
    .stSpinner > div { border-color: var(--primary) !important; }
</style>
"""

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_all_data() -> dict[str, Any]:
    """Load all upstream pipeline data via utils (cached by Streamlit)."""
    return load_analysis_results()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def confidence_badge(conf: str) -> str:
    """Return an HTML badge for a confidence level."""
    cls = conf.lower() if conf.lower() in ("high", "medium", "low") else "medium"
    return f'<span class="badge-{cls}">{conf}</span>'


def classification_badge(cls: str) -> str:
    """Return an HTML badge for alignment classification."""
    mapping = {"Excellent": "good", "Good": "good", "Fair": "weak", "Poor": "orphan"}
    badge_cls = mapping.get(cls, "weak")
    return f'<span class="badge-{badge_cls}">{cls}</span>'


def make_metric_card(title: str, value: str, subtitle: str = "") -> str:
    """Generate HTML for a metric card."""
    sub_html = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="metric-card">
        <h4>{title}</h4>
        <div class="value">{value}</div>
        {sub_html}
    </div>"""


# ═══════════════════════════════════════════════════════════════════════
# Page: Home
# ═══════════════════════════════════════════════════════════════════════

def page_home(data: dict) -> None:
    """Render the Home page."""
    st.title("Hospital Strategy-Action Plan Alignment System")
    st.markdown("**Nawaloka Hospital Negombo** | Strategic Plan 2026-2030 | Action Plan 2025")

    alignment = data["alignment"]
    actions = data["actions"].get("actions", [])
    objectives = data["strategic"].get("objectives", [])

    # Quick stats
    cols = st.columns(4)
    with cols[0]:
        score = alignment.get("overall_score", 0)
        st.markdown(make_metric_card(
            "Overall Sync Score",
            f"{score:.1%}",
            alignment.get("overall_classification", ""),
        ), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(make_metric_card(
            "Strategic Objectives", str(len(objectives)),
            "A through E",
        ), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(make_metric_card(
            "Action Items", str(len(actions)),
            f"Budget: LKR {sum(a.get('budget_lkr_millions', 0) for a in actions):.0f}M",
        ), unsafe_allow_html=True)
    with cols[3]:
        orphans = len(alignment.get("orphan_actions", []))
        st.markdown(make_metric_card(
            "Orphan Actions", str(orphans),
            "Below alignment threshold",
        ), unsafe_allow_html=True)

    st.divider()

    # Distribution overview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Alignment Distribution")
        dist = alignment.get("distribution", {})
        if dist:
            df_dist = pd.DataFrame([
                {"Classification": k, "Count": v} for k, v in dist.items()
            ])
            colours = {"Excellent": "#1B5E20", "Good": "#66BB6A",
                       "Fair": "#FFB300", "Poor": "#E53935"}
            fig = px.bar(df_dist, x="Classification", y="Count",
                         color="Classification",
                         color_discrete_map=colours)
            fig.update_layout(showlegend=False, height=300,
                              margin=dict(t=20, b=20))
            st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Budget by Objective")
        obj_budget: dict[str, float] = {}
        for a in actions:
            code = a.get("strategic_objective_code", "?")
            obj_budget[code] = obj_budget.get(code, 0) + a.get("budget_lkr_millions", 0)
        if obj_budget:
            df_budget = pd.DataFrame([
                {"Objective": k, "Budget (LKR M)": v}
                for k, v in sorted(obj_budget.items())
            ])
            fig = px.pie(df_budget, names="Objective", values="Budget (LKR M)",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig, width='stretch')

    # Project overview
    st.subheader("About This System")
    st.markdown("""
    The **ISPS** (Intelligent Strategic Plan Synchronization) system uses AI to evaluate
    how well a hospital's operational action plan aligns with its strategic plan. The pipeline includes:

    1. **Document Processing** — NLP extraction from markdown plans
    2. **Vector Embeddings** — Semantic similarity via sentence-transformers
    3. **Synchronization Analysis** — Alignment scoring matrix
    4. **Ontology Mapping** — Healthcare concept taxonomy alignment
    5. **Knowledge Graph** — Structural analysis of relationships
    6. **RAG Engine** — LLM-powered improvement recommendations
    7. **Agent Reasoner** — Agentic diagnosis with self-critique
    """)


# ═══════════════════════════════════════════════════════════════════════
# Page: Synchronization Analysis
# ═══════════════════════════════════════════════════════════════════════

def page_sync_analysis(data: dict) -> None:
    """Render the Synchronization Analysis page."""
    st.title("Synchronization Analysis")

    alignment = data["alignment"]

    # Overall gauge
    col1, col2 = st.columns([1, 2])
    with col1:
        score = alignment.get("overall_score", 0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            title={"text": "Overall Synchronization", "font": {"size": 16}},
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1565C0"},
                "steps": [
                    {"range": [0, 45], "color": "#FFCDD2"},
                    {"range": [45, 60], "color": "#FFF9C4"},
                    {"range": [60, 75], "color": "#C8E6C9"},
                    {"range": [75, 100], "color": "#81C784"},
                ],
                "threshold": {
                    "line": {"color": "#B71C1C", "width": 3},
                    "thickness": 0.8,
                    "value": 45,
                },
            },
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=20, l=30, r=30))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Strategy-wise Alignment")
        obj_data = alignment.get("objective_alignments", [])
        if obj_data:
            df_obj = pd.DataFrame([{
                "Objective": f"{o['code']}: {o['name'][:30]}",
                "Mean Similarity": o["mean_similarity"],
                "Max Similarity": o["max_similarity"],
                "Coverage": o["coverage_score"],
            } for o in obj_data])
            fig = px.bar(df_obj, x="Objective",
                         y=["Mean Similarity", "Max Similarity"],
                         barmode="group",
                         color_discrete_sequence=["#42A5F5", "#1565C0"])
            fig.update_layout(height=280, margin=dict(t=20, b=20),
                              yaxis_title="Cosine Similarity",
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width='stretch')

    st.divider()

    # Alignment matrix heatmap
    st.subheader("Alignment Matrix Heatmap")
    matrix = alignment.get("alignment_matrix", [])
    row_labels = alignment.get("matrix_row_labels", [])
    col_labels = alignment.get("matrix_col_labels", [])

    if matrix:
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f"Act {c}" for c in col_labels],
            y=[f"Obj {r}" for r in row_labels],
            colorscale=[
                [0, "#FFCDD2"], [0.45, "#FFE082"],
                [0.6, "#A5D6A7"], [1, "#1B5E20"],
            ],
            zmin=0, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="Objective %{y}<br>Action %{x}<br>Score: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            height=300, margin=dict(t=20, b=20),
            xaxis_title="Action Items",
            yaxis_title="Strategic Objectives",
        )
        st.plotly_chart(fig, width='stretch')

    # Distribution histogram
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Similarity Score Distribution")
        if matrix:
            all_scores = [s for row in matrix for s in row]
            fig = px.histogram(x=all_scores, nbins=25,
                               labels={"x": "Cosine Similarity", "y": "Count"},
                               color_discrete_sequence=["#42A5F5"])
            fig.add_vline(x=0.45, line_dash="dash", line_color="red",
                          annotation_text="Fair threshold")
            fig.add_vline(x=0.60, line_dash="dash", line_color="green",
                          annotation_text="Good threshold")
            fig.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Action Alignment Details")
        action_data = alignment.get("action_alignments", [])
        if action_data:
            df_actions = pd.DataFrame([{
                "Action": f"{a['action_number']}. {a['title'][:40]}",
                "Declared": a["declared_objective"],
                "Best": a["best_objective"],
                "Score": a["best_score"],
                "Class": a["classification"],
                "Orphan": "Yes" if a["is_orphan"] else "",
            } for a in action_data])
            st.dataframe(df_actions, width='stretch', height=300,
                         hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# Page: Improvement Recommendations
# ═══════════════════════════════════════════════════════════════════════

def page_recommendations(data: dict) -> None:
    """Render the Improvement Recommendations page."""
    st.title("Improvement Recommendations")

    rag = data["rag"]
    gaps_data = data["gaps"]

    tab1, tab2, tab3 = st.tabs([
        "Poorly Aligned Actions",
        "Missing Actions",
        "Strategic Gaps",
    ])

    # ── Tab 1: Poorly aligned ──────────────────────────────────────
    with tab1:
        improvements = rag.get("improvements", [])
        if not improvements:
            st.info("No improvement recommendations available.")
        else:
            st.markdown(f"**{len(improvements)} actions** identified for improvement")
            for imp in improvements:
                with st.expander(
                    f"Action {imp['action_number']}: {imp['action_title'][:55]} "
                    f"— Score: {imp['alignment_score']:.3f}",
                    expanded=False,
                ):
                    cols = st.columns([1, 1, 1])
                    with cols[0]:
                        st.metric("Alignment Score", f"{imp['alignment_score']:.3f}")
                    with cols[1]:
                        st.metric("Declared Objective", imp["declared_objective"])
                    with cols[2]:
                        st.markdown(f"Confidence: {confidence_badge(imp.get('confidence', 'MEDIUM'))}",
                                    unsafe_allow_html=True)

                    if imp.get("modified_description"):
                        st.markdown("**Suggested Description:**")
                        desc = format_llm_response(imp["modified_description"], max_length=500)
                        st.markdown(f"> {desc}")

                    if imp.get("strategic_linkage"):
                        linkage = format_llm_response(imp["strategic_linkage"], max_length=300)
                        st.markdown(f"**Strategic Linkage:** {linkage}")

    # ── Tab 2: Missing actions ─────────────────────────────────────
    with tab2:
        suggestions = rag.get("new_action_suggestions", [])
        if not suggestions:
            st.info("No new action suggestions available.")
        else:
            st.markdown(f"**{len(suggestions)} new actions** suggested to fill gaps")
            for sug in suggestions:
                with st.expander(
                    f"[Obj {sug['objective_code']}] {sug.get('title', 'Untitled')[:55]}",
                    expanded=False,
                ):
                    if sug.get("description"):
                        st.markdown(f"**Description:** {sug['description'][:400]}")
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown(f"**Owner:** {sug.get('owner', 'N/A')}")
                    with cols[1]:
                        st.markdown(f"**Timeline:** {sug.get('timeline', 'N/A')}")
                    with cols[2]:
                        st.markdown(f"**Budget:** {sug.get('budget_estimate', 'N/A')}")
                    if sug.get("kpis"):
                        st.markdown("**KPIs:**")
                        for kpi in sug["kpis"][:4]:
                            st.markdown(f"- {kpi}")

    # ── Tab 3: Strategic gaps ──────────────────────────────────────
    with tab3:
        uncovered = gaps_data.get("uncovered_strategy_concepts", [])
        weak = gaps_data.get("weak_actions", [])

        if uncovered:
            st.subheader(f"Uncovered Strategy Concepts ({len(uncovered)})")
            for gap in uncovered:
                st.markdown(
                    f'<span class="badge-gap">{gap["concept_id"]}</span> '
                    f'— Goals: {", ".join(gap.get("related_strategy_goals", []))}',
                    unsafe_allow_html=True,
                )
                st.caption(gap.get("note", ""))

        if weak:
            st.subheader(f"Weakly Aligned Actions ({len(weak)})")
            df_weak = pd.DataFrame([{
                "Action": w["action_id"],
                "Best Concept": w.get("best_concept", ""),
                "Score": w.get("best_score", 0),
                "Note": w.get("note", "")[:60],
            } for w in weak])
            st.dataframe(df_weak, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# Page: Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════

def page_knowledge_graph(data: dict) -> None:
    """Render the Knowledge Graph visualization page."""
    st.title("Knowledge Graph Explorer")

    kg = data["kg"]
    if not kg:
        st.warning("Knowledge graph data not available. Run `python -m src.knowledge_graph` first.")
        return

    insights = kg.get("insights", {})

    # Stats row
    cols = st.columns(4)
    with cols[0]:
        st.metric("Nodes", insights.get("node_count", len(kg.get("nodes", []))))
    with cols[1]:
        st.metric("Edges", insights.get("edge_count", len(kg.get("edges", kg.get("links", [])))))
    with cols[2]:
        st.metric("Communities", insights.get("community_count", 0))
    with cols[3]:
        st.metric("Isolated Actions", len(insights.get("isolated_actions", [])))

    st.divider()

    # Threshold filter
    threshold = st.slider("Minimum edge weight to display", 0.0, 1.0, 0.45, 0.05)

    # Build interactive Plotly network
    nodes = kg.get("nodes", [])
    links = kg.get("edges", kg.get("links", []))

    if not nodes:
        st.info("No graph nodes found.")
        return

    # Filter edges by weight
    filtered_links = [
        l for l in links
        if l.get("weight", 1.0) >= threshold
    ]

    # Build node index
    node_map = {n["id"]: i for i, n in enumerate(nodes)}

    # Spring layout simulation (simple force-directed)
    np.random.seed(42)
    n = len(nodes)
    pos = np.random.randn(n, 2) * 2

    # Simple iterative layout
    for _ in range(50):
        for link in filtered_links:
            src_idx = node_map.get(link.get("source"))
            tgt_idx = node_map.get(link.get("target"))
            if src_idx is not None and tgt_idx is not None:
                diff = pos[tgt_idx] - pos[src_idx]
                dist = max(np.linalg.norm(diff), 0.1)
                force = diff / dist * 0.05
                pos[src_idx] += force
                pos[tgt_idx] -= force

    # Color mapping
    color_map = {
        "StrategyObjective": "#4285F4",
        "StrategyGoal": "#81D4FA",
        "OntologyConcept": "#9C27B0",
        "Action": "#4CAF50",
        "KPI": "#FFEB3B",
        "Stakeholder": "#FF9800",
        "TimelineQuarter": "#9E9E9E",
    }

    # Only show key node types
    show_types = st.multiselect(
        "Node types to display",
        list(color_map.keys()),
        default=["StrategyObjective", "Action", "OntologyConcept"],
    )

    visible = set()
    for i, node in enumerate(nodes):
        if node.get("node_type") in show_types:
            visible.add(i)

    # Also include linked nodes
    for link in filtered_links:
        src = node_map.get(link.get("source"))
        tgt = node_map.get(link.get("target"))
        if src in visible or tgt in visible:
            if src is not None:
                visible.add(src)
            if tgt is not None:
                visible.add(tgt)

    # Edge traces
    edge_x, edge_y = [], []
    for link in filtered_links:
        src = node_map.get(link.get("source"))
        tgt = node_map.get(link.get("target"))
        if src in visible and tgt in visible:
            edge_x += [pos[src][0], pos[tgt][0], None]
            edge_y += [pos[src][1], pos[tgt][1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#CCCCCC"),
        hoverinfo="none",
    ))

    # Node traces by type
    for ntype in show_types:
        idx_list = [i for i, nd in enumerate(nodes)
                    if nd.get("node_type") == ntype and i in visible]
        if not idx_list:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in idx_list],
            y=[pos[i][1] for i in idx_list],
            mode="markers+text",
            marker=dict(
                size=[min(nodes[i].get("size", 10) * 1.5, 40) for i in idx_list],
                color=color_map.get(ntype, "#999"),
                line=dict(width=1, color="white"),
            ),
            text=[nodes[i].get("label", "")[:20] for i in idx_list],
            textposition="top center",
            textfont=dict(size=7),
            name=ntype,
            hovertext=[
                f"{nodes[i].get('label', '')}<br>"
                f"Type: {ntype}<br>"
                f"Score: {nodes[i].get('alignment_score', 'N/A')}"
                for i in idx_list
            ],
            hoverinfo="text",
        ))

    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=20, b=40, l=20, r=20),
    )
    st.plotly_chart(fig, width='stretch')

    # Bridge nodes and suggestions
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bridge Nodes")
        bridge = insights.get("bridge_nodes", [])
        if bridge:
            for bn in bridge[:5]:
                st.markdown(
                    f"**{bn.get('label', '')[:40]}** "
                    f"({bn.get('node_type', '')}) — "
                    f"betweenness: {bn.get('betweenness', 0):.4f}"
                )
        else:
            st.caption("No significant bridge nodes detected.")

    with col2:
        st.subheader("Suggested New Connections")
        suggestions = insights.get("new_connections", [])
        if suggestions:
            for sug in suggestions[:5]:
                st.markdown(
                    f"**{sug.get('concept', '')}** ← "
                    f"{sug.get('suggested_action', '')} "
                    f"(conf: {sug.get('confidence', 0):.3f})"
                )
        else:
            st.caption("No connection suggestions available.")


# ═══════════════════════════════════════════════════════════════════════
# Page: Ontology Browser
# ═══════════════════════════════════════════════════════════════════════

def page_ontology(data: dict) -> None:
    """Render the Ontology Browser page."""
    st.title("Healthcare Strategy Ontology")

    mappings = data["mappings"]
    gaps_data = data["gaps"]

    if not mappings:
        st.warning("Ontology mappings not available. Run `python -m src.ontology_mapper` first.")
        return

    meta = mappings.get("metadata", {})
    st.markdown(
        f"**Model:** {meta.get('embedding_model', 'N/A')} | "
        f"**Concepts:** {meta.get('total_concepts', 0)} | "
        f"**Threshold:** {meta.get('mapping_threshold', 0)}"
    )

    # Build concept tree from mappings
    concept_tree: dict[str, list[str]] = {}
    concept_labels: dict[str, str] = {}
    concept_action_count: dict[str, int] = {}

    for section in ("action_mappings", "strategy_mappings"):
        for item in mappings.get(section, []):
            for m in item.get("mappings", []):
                parent = m.get("parent_concept", "")
                cid = m["concept_id"]
                concept_labels[cid] = m.get("concept_label", cid)
                if parent and parent != cid:
                    concept_tree.setdefault(parent, [])
                    if cid not in concept_tree[parent]:
                        concept_tree[parent].append(cid)
                if section == "action_mappings":
                    concept_action_count[cid] = concept_action_count.get(cid, 0) + 1

    # Coverage statistics
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Concept Hierarchy")
        # Uncovered concepts
        uncovered_ids = {g["concept_id"] for g in gaps_data.get("uncovered_strategy_concepts", [])}

        for parent in sorted(concept_tree.keys()):
            count = concept_action_count.get(parent, 0)
            icon = "+" if count > 0 else "-"
            st.markdown(f"### {concept_labels.get(parent, parent)} ({count} action links)")
            for child in sorted(concept_tree[parent]):
                child_count = concept_action_count.get(child, 0)
                if child in uncovered_ids:
                    badge = '<span class="badge-orphan">UNCOVERED</span>'
                elif child_count > 0:
                    badge = f'<span class="badge-good">{child_count} actions</span>'
                else:
                    badge = '<span class="badge-weak">0 actions</span>'
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;{concept_labels.get(child, child)} {badge}",
                    unsafe_allow_html=True,
                )

    with col2:
        st.subheader("Coverage Summary")
        total_concepts = len(concept_labels)
        covered = sum(1 for c in concept_labels if concept_action_count.get(c, 0) > 0)
        coverage_pct = covered / max(total_concepts, 1)

        st.metric("Total Concepts", total_concepts)
        st.metric("Covered by Actions", covered)
        st.metric("Coverage Rate", f"{coverage_pct:.0%}")
        st.metric("Uncovered (Strategy)", len(uncovered_ids))

        # Sunburst of concept coverage
        sunburst_data = []
        for parent in concept_tree:
            for child in concept_tree[parent]:
                sunburst_data.append({
                    "parent": concept_labels.get(parent, parent),
                    "child": concept_labels.get(child, child),
                    "count": concept_action_count.get(child, 0),
                })
        if sunburst_data:
            df_sun = pd.DataFrame(sunburst_data)
            fig = px.sunburst(
                df_sun, path=["parent", "child"], values="count",
                color="count",
                color_continuous_scale=["#FFCDD2", "#C8E6C9", "#1B5E20"],
            )
            fig.update_layout(height=400, margin=dict(t=10, b=10))
            st.plotly_chart(fig, width='stretch')


# ═══════════════════════════════════════════════════════════════════════
# Page: Agent Insights
# ═══════════════════════════════════════════════════════════════════════

def page_agent_insights(data: dict) -> None:
    """Render the Agent Insights page."""
    st.title("Agentic AI Reasoning Insights")

    agent_recs = data["agent_recs"]
    agent_trace = data["agent_trace"]

    if not agent_recs:
        st.warning("Agent recommendations not available. Run `python -m src.agent_reasoner` first.")
        return

    meta = agent_recs.get("metadata", {})

    # Stats
    cols = st.columns(4)
    with cols[0]:
        st.metric("Issues Diagnosed", meta.get("total_issues_diagnosed", 0))
    with cols[1]:
        st.metric("Recommendations", meta.get("total_recommendations", 0))
    with cols[2]:
        st.metric("Total Impact", f"{meta.get('total_impact_score', 0):.3f}")
    with cols[3]:
        st.metric("Iterations Run", meta.get("max_iterations", 0))

    st.divider()

    # Critical issues ranked
    st.subheader("Diagnosed Issues (by priority)")
    issues = agent_recs.get("diagnosed_issues_summary", [])
    if issues:
        df_issues = pd.DataFrame([{
            "Issue ID": i["issue_id"],
            "Type": i["type"],
            "Priority": i["priority_score"],
            "Objectives": ", ".join(i.get("affected_objectives", [])),
            "Addressed": "Yes" if i.get("addressed") else "",
        } for i in issues])
        st.dataframe(df_issues, width='stretch', hide_index=True)

    st.divider()

    # Recommendations
    st.subheader("Agent Recommendations")
    for rec in agent_recs.get("recommendations", []):
        with st.expander(
            f"{rec['rec_id']} — {rec['issue_type']} | "
            f"Impact: {rec['impact_score']:.3f} | "
            f"{rec.get('confidence', 'MEDIUM')}",
            expanded=False,
        ):
            st.markdown(f"**Root Cause:** {format_llm_response(rec.get('what_to_change', ''), max_length=300)}")
            st.markdown(f"**Expected Outcome:** {format_llm_response(rec.get('why', ''), max_length=300)}")

            st.markdown("**Recommended Actions:**")
            for a in rec.get("actions", []):
                st.markdown(f"- {format_llm_response(a, max_length=200)}")

            st.markdown("**Proposed KPIs:**")
            for k in rec.get("kpis", []):
                st.markdown(f"- {format_llm_response(k, max_length=150)}")

            cols = st.columns(3)
            with cols[0]:
                st.markdown(f"**Budget:** LKR {rec.get('estimated_budget', 0):.0f}M")
            with cols[1]:
                st.markdown(f"**Timeline:** {rec.get('estimated_timeline', 'N/A')[:60]}")
            with cols[2]:
                st.markdown(f"**Evidence:** {', '.join(rec.get('evidence_ids', [])[:5])}")

    # Reasoning trace
    st.divider()
    st.subheader("Reasoning Trace")
    traces = agent_trace.get("traces", [])
    for trace in traces:
        issue_id = trace.get("issue_detected", {}).get("issue_id", "?")
        status = trace.get("final_decision", {}).get("status", "?")
        status_icon = "+" if status == "ACCEPTED" else "-"

        st.markdown(
            f'<div class="trace-step">'
            f'<strong>Iteration {trace["iteration"]}</strong> — '
            f'{issue_id} → <strong>{status}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander(f"Details: Iteration {trace['iteration']}", expanded=False):
            st.markdown("**Investigation Plan:**")
            for step in trace.get("investigation_plan", []):
                st.markdown(f"- {step}")

            st.markdown("**Reasoning Summary:**")
            for bullet in trace.get("reasoning_summary", []):
                st.markdown(f"- {bullet}")

            st.markdown("**Tool Calls:**")
            for tc in trace.get("tool_calls", []):
                st.markdown(f"- `{tc.get('tool', '?')}` — {tc.get('purpose', tc.get('query', '')[:50])}")

            critique = trace.get("critique_summary", {})
            if critique:
                st.markdown("**Critique:**")
                for key in ("feasibility", "alignment_check", "risks", "missing_info"):
                    val = critique.get(key, "")
                    if val:
                        st.markdown(f"- **{key.replace('_', ' ').title()}:** {val[:150]}")


# ═══════════════════════════════════════════════════════════════════════
# Page: Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════

def page_evaluation(data: dict) -> None:
    """Render the Evaluation Metrics page."""
    st.title("Evaluation Metrics")

    alignment = data["alignment"]

    # Ground truth: intentionally misaligned actions
    MISALIGNED_GROUND_TRUTH = {8, 19, 24, 25}
    ALL_ACTIONS = set(range(1, 26))

    orphan_detected = set(alignment.get("orphan_actions", []))
    poorly_aligned = set(alignment.get("poorly_aligned_actions", []))

    st.subheader("Misalignment Detection Performance")
    st.markdown(
        "The action plan contains **4 intentionally misaligned actions** "
        "(8, 19, 24, 25) embedded for evaluation. Below shows how well "
        "each detection method identifies them."
    )

    # Compute metrics for different methods
    methods = {
        "Embedding Similarity (Orphan Detection)": orphan_detected,
        "Embedding Similarity (Poor Alignment)": poorly_aligned,
    }

    # Add ontology-based detection
    gaps_data = data["gaps"]
    weak_actions = set()
    for w in gaps_data.get("weak_actions", []):
        act_id = w.get("action_id", "")
        num_str = act_id.replace("action_", "")
        if num_str.isdigit():
            weak_actions.add(int(num_str))
    methods["Ontology Mapping (Weak Actions)"] = weak_actions

    # Agent-based detection
    agent_recs = data["agent_recs"]
    agent_detected = set()
    for issue in agent_recs.get("diagnosed_issues_summary", []):
        if issue.get("type") == "orphan_action":
            num_str = issue["issue_id"].replace("orphan_action_", "")
            if num_str.isdigit():
                agent_detected.add(int(num_str))
    methods["Agent Reasoner (Orphan Detection)"] = agent_detected

    # Combined
    combined = orphan_detected | weak_actions
    methods["Combined (Embedding + Ontology)"] = combined

    metrics_rows = []
    for method_name, detected in methods.items():
        tp = len(detected & MISALIGNED_GROUND_TRUTH)
        fp = len(detected - MISALIGNED_GROUND_TRUTH)
        fn = len(MISALIGNED_GROUND_TRUTH - detected)
        tn = len(ALL_ACTIONS - detected - MISALIGNED_GROUND_TRUTH)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)
        accuracy = (tp + tn) / len(ALL_ACTIONS)

        metrics_rows.append({
            "Method": method_name,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
        })

    df_metrics = pd.DataFrame(metrics_rows)

    # Display metrics table
    st.dataframe(
        df_metrics.style.format({
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1": "{:.2%}",
            "Accuracy": "{:.2%}",
        }),
        width='stretch',
        hide_index=True,
    )

    # Bar chart comparison
    col1, col2 = st.columns(2)
    with col1:
        df_plot = df_metrics.melt(
            id_vars="Method",
            value_vars=["Precision", "Recall", "F1"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(df_plot, x="Method", y="Score", color="Metric",
                     barmode="group",
                     color_discrete_sequence=["#1565C0", "#2E7D32", "#FF8F00"])
        fig.update_layout(
            height=350, margin=dict(t=20, b=80),
            xaxis_tickangle=-30,
            yaxis_title="Score",
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Confusion matrix for best method
        best_idx = df_metrics["F1"].idxmax()
        best = df_metrics.iloc[best_idx]
        st.subheader(f"Confusion Matrix: {best['Method'][:35]}")
        cm = [[int(best["TP"]), int(best["FP"])],
              [int(best["FN"]), int(best["TN"])]]
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Misaligned", "Predicted Aligned"],
            y=["Actually Misaligned", "Actually Aligned"],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale=[[0, "#E3F2FD"], [1, "#1565C0"]],
            showscale=False,
        ))
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, width='stretch')

    # Detection details
    st.divider()
    st.subheader("Detection Details")
    st.markdown(f"**Ground truth misaligned:** {sorted(MISALIGNED_GROUND_TRUTH)}")
    st.markdown(f"**Orphan actions detected:** {sorted(orphan_detected)}")
    st.markdown(f"**Ontology weak actions:** {sorted(weak_actions)}")

    misaligned_details = alignment.get("mismatched_actions", [])
    if misaligned_details:
        st.subheader("Declared vs Best Objective Mismatches")
        df_mis = pd.DataFrame([{
            "Action": m["action_number"],
            "Title": m["title"][:45],
            "Declared": m["declared_objective"],
            "Best Match": m["best_objective"],
            "Best Score": f"{m['best_score']:.3f}",
        } for m in misaligned_details])
        st.dataframe(df_mis, width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# Export Report
# ═══════════════════════════════════════════════════════════════════════

def generate_report_text(data: dict) -> str:
    """Generate a text summary report for export."""
    alignment = data["alignment"]
    rag = data["rag"]
    agent = data["agent_recs"]
    gaps = data["gaps"]

    lines = [
        "=" * 70,
        "HOSPITAL STRATEGY-ACTION PLAN ALIGNMENT REPORT",
        "Nawaloka Hospital Negombo",
        "=" * 70,
        "",
        "1. OVERALL SYNCHRONIZATION",
        f"   Score: {alignment.get('overall_score', 0):.1%} ({alignment.get('overall_classification', '')})",
        f"   Mean similarity: {alignment.get('mean_similarity', 0):.4f}",
        f"   Distribution: {alignment.get('distribution', {})}",
        "",
        "2. OBJECTIVE ALIGNMENT",
    ]
    for oa in alignment.get("objective_alignments", []):
        lines.append(
            f"   {oa['code']}: {oa['name']:<40} "
            f"mean={oa['mean_similarity']:.3f}  "
            f"coverage={oa['coverage_score']:.0%}"
        )

    lines += ["", "3. ORPHAN ACTIONS"]
    for num in alignment.get("orphan_actions", []):
        aa = next((a for a in alignment.get("action_alignments", [])
                    if a["action_number"] == num), {})
        lines.append(f"   Action {num}: {aa.get('title', '?')[:50]} — score={aa.get('best_score', 0):.3f}")

    lines += ["", "4. RAG RECOMMENDATIONS"]
    lines.append(f"   Improvements: {len(rag.get('improvements', []))}")
    lines.append(f"   New actions suggested: {len(rag.get('new_action_suggestions', []))}")

    lines += ["", "5. AGENT RECOMMENDATIONS"]
    for rec in (agent or {}).get("recommendations", []):
        lines.append(f"   [{rec['rec_id']}] {rec['issue_type']} — impact={rec['impact_score']:.3f}")

    lines += ["", "6. STRATEGIC GAPS"]
    lines.append(f"   Uncovered concepts: {len(gaps.get('uncovered_strategy_concepts', []))}")
    lines.append(f"   Weak actions: {len(gaps.get('weak_actions', []))}")

    lines += ["", "=" * 70, "Generated by ISPS Dashboard", "=" * 70]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="ISPS Dashboard — Nawaloka Hospital",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## ISPS Dashboard")
        st.markdown("**Nawaloka Hospital Negombo**")
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "Home",
                "Synchronization Analysis",
                "Improvement Recommendations",
                "Knowledge Graph",
                "Ontology Browser",
                "Agent Insights",
                "Evaluation Metrics",
                "Upload & Analyze",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        # Export button
        st.markdown("### Export")

    # Load data
    with st.spinner("Loading pipeline data..."):
        data = load_all_data()

    # Export in sidebar (after data is loaded)
    with st.sidebar:
        # PDF report
        pdf_bytes = generate_pdf_report(data)
        if pdf_bytes:
            st.download_button(
                label="Download Report (PDF)",
                data=pdf_bytes,
                file_name="isps_alignment_report.pdf",
                mime="application/pdf",
            )

        # TXT report
        report_text = generate_report_text(data)
        st.download_button(
            label="Download Report (TXT)",
            data=report_text,
            file_name="isps_alignment_report.txt",
            mime="text/plain",
        )

        # CSV / JSON / Excel via export_data()
        export_fmt = st.selectbox(
            "Data export format",
            ["csv", "json", "excel"],
            label_visibility="collapsed",
        )
        raw_bytes, filename, mime = export_data(data, export_fmt)
        st.download_button(
            label=f"Export Data ({export_fmt.upper()})",
            data=raw_bytes,
            file_name=filename,
            mime=mime,
        )

    # Route to selected page
    if page == "Home":
        page_home(data)
    elif page == "Synchronization Analysis":
        page_sync_analysis(data)
    elif page == "Improvement Recommendations":
        page_recommendations(data)
    elif page == "Knowledge Graph":
        page_knowledge_graph(data)
    elif page == "Ontology Browser":
        page_ontology(data)
    elif page == "Agent Insights":
        page_agent_insights(data)
    elif page == "Evaluation Metrics":
        page_evaluation(data)
    elif page == "Upload & Analyze":
        page_upload_analyze()


# ═══════════════════════════════════════════════════════════════════════
# Page: Upload & Analyze
# ═══════════════════════════════════════════════════════════════════════

def page_upload_analyze() -> None:
    """Upload PDF documents and run dynamic alignment analysis."""
    st.title("Upload & Analyze")
    st.markdown(
        "Upload your own **Strategic Plan** and **Action Plan** as PDF files "
        "to get alignment scores and insights. The system uses AI to extract "
        "objectives and actions, then computes semantic alignment."
    )

    # Check Ollama availability
    from src.pdf_processor import check_ollama_available
    ollama_ok = check_ollama_available()

    if not ollama_ok:
        st.error(
            "Ollama is not running or `llama3.1:8b` model is not available. "
            "Start Ollama with `ollama serve` and pull the model with "
            "`ollama pull llama3.1:8b` before uploading files."
        )
        return

    st.success("Ollama is running and ready.")

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Strategic Plan")
        strategic_file = st.file_uploader(
            "Upload Strategic Plan (PDF)",
            type=["pdf"],
            key="strategic_pdf",
            help="The multi-year strategic plan with objectives, goals, and KPIs.",
        )
        if strategic_file:
            st.caption(f"Uploaded: {strategic_file.name} ({strategic_file.size / 1024:.0f} KB)")

    with col2:
        st.subheader("Action Plan")
        action_file = st.file_uploader(
            "Upload Action Plan (PDF)",
            type=["pdf"],
            key="action_pdf",
            help="The annual action plan with specific actions mapped to objectives.",
        )
        if action_file:
            st.caption(f"Uploaded: {action_file.name} ({action_file.size / 1024:.0f} KB)")

    # Analyze button
    if strategic_file and action_file:
        if st.button("Run Alignment Analysis", type="primary", width='stretch'):
            _run_upload_analysis(strategic_file, action_file)

    # Display results if available in session state
    if "upload_report" in st.session_state:
        _display_upload_results(st.session_state["upload_report"])


def _run_upload_analysis(strategic_file, action_file) -> None:
    """Process uploaded PDFs and run alignment analysis."""
    from src.pdf_processor import (
        extract_strategic_plan_from_pdf,
        extract_action_plan_from_pdf,
    )
    from src.dynamic_analyzer import run_dynamic_analysis

    progress = st.progress(0, text="Reading PDFs...")

    try:
        # Step 1: Extract text and parse strategic plan
        progress.progress(10, text="Extracting strategic plan with AI...")
        strategic_bytes = strategic_file.read()
        strategic_data = extract_strategic_plan_from_pdf(strategic_bytes)
        n_obj = len(strategic_data.get("objectives", []))
        progress.progress(35, text=f"Found {n_obj} strategic objectives.")

        # Step 2: Extract text and parse action plan
        progress.progress(40, text="Extracting action plan with AI...")
        action_bytes = action_file.read()
        action_data = extract_action_plan_from_pdf(action_bytes)
        n_act = len(action_data.get("actions", []))
        progress.progress(65, text=f"Found {n_act} action items.")

        # Step 3: Run alignment analysis
        progress.progress(70, text="Computing semantic embeddings...")
        report = run_dynamic_analysis(strategic_data, action_data)
        progress.progress(100, text="Analysis complete!")

        # Store in session state
        st.session_state["upload_report"] = report
        st.rerun()

    except ValueError as e:
        st.error(f"Parsing error: {e}")
    except ConnectionError:
        st.error("Cannot connect to Ollama. Make sure it is running.")
    except Exception as e:
        st.error(f"Analysis failed: {e}")


def _display_upload_results(report: dict) -> None:
    """Render the alignment analysis results from uploaded documents."""
    st.divider()
    st.header("Analysis Results")

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        st.metric("Overall Score", f"{report['overall_score']:.3f}")
    with cols[1]:
        st.metric("Classification", report["overall_classification"])
    with cols[2]:
        st.metric("Objectives", len(report["objective_alignments"]))
    with cols[3]:
        st.metric("Actions", len(report["action_alignments"]))

    # Distribution
    dist = report["distribution"]
    dist_cols = st.columns(4)
    colours = {"Excellent": "#2ecc71", "Good": "#27ae60", "Fair": "#f39c12", "Poor": "#e74c3c"}
    for i, (cls, count) in enumerate(dist.items()):
        with dist_cols[i]:
            st.markdown(
                f"<div style='text-align:center; padding:8px; "
                f"background:{colours[cls]}22; border-radius:8px;'>"
                f"<b>{cls}</b><br><span style='font-size:24px;'>{count}</span></div>",
                unsafe_allow_html=True,
            )

    # Alignment heatmap
    st.subheader("Alignment Heatmap")
    matrix = report["alignment_matrix"]
    row_labels = [
        f"{code}: {oa['name']}"
        for code, oa in zip(report["matrix_row_labels"], report["objective_alignments"])
    ]
    col_labels = [f"Action {n}" for n in report["matrix_col_labels"]]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=col_labels,
        y=row_labels,
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate="Objective: %{y}<br>Action: %{x}<br>Similarity: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(300, len(row_labels) * 60),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Actions",
        yaxis_title="Strategic Objectives",
    )
    st.plotly_chart(fig, width='stretch')

    # Per-objective breakdown
    st.subheader("Per-Objective Alignment")
    for oa in report["objective_alignments"]:
        with st.expander(
            f"[{oa['code']}] {oa['name']} — "
            f"Mean: {oa['mean_similarity']:.3f} | "
            f"Max: {oa['max_similarity']:.3f} | "
            f"Coverage: {oa['coverage_score']:.0%}",
            expanded=False,
        ):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Aligned Actions", oa["aligned_action_count"])
            with c2:
                st.metric("Declared Actions", oa["declared_action_count"])
            with c3:
                st.metric("Gap Actions", len(oa["gap_actions"]))

            if oa["top_actions"]:
                st.markdown("**Top-5 Actions:**")
                for act_num, sim in oa["top_actions"]:
                    act_info = next(
                        (a for a in report["action_alignments"]
                         if a["action_number"] == act_num), {}
                    )
                    title = act_info.get("title", f"Action {act_num}")
                    st.markdown(f"- Action {act_num}: {title} (sim={sim:.3f})")

    # Per-action table
    st.subheader("Per-Action Alignment")
    action_rows = []
    for a in report["action_alignments"]:
        action_rows.append({
            "#": a["action_number"],
            "Title": a["title"][:50],
            "Declared": a["declared_objective"],
            "Best Fit": a["best_objective"],
            "Score": a["best_score"],
            "Class": a["classification"],
            "Match": "Yes" if a["declared_match"] else "No",
            "Orphan": "Yes" if a["is_orphan"] else "",
        })

    df = pd.DataFrame(action_rows)
    st.dataframe(
        df.style.apply(
            lambda row: [
                "background-color: #ffcccc" if row["Class"] == "Poor"
                else "background-color: #ccffcc" if row["Class"] in ("Good", "Excellent")
                else ""
            ] * len(row),
            axis=1,
        ),
        width='stretch',
        hide_index=True,
    )

    # Orphan and mismatch warnings
    if report["orphan_actions"]:
        st.warning(
            f"**Orphan Actions** (no strategic alignment): "
            f"{report['orphan_actions']}"
        )

    if report["mismatched_actions"]:
        st.info(
            f"**Declaration Mismatches** ({len(report['mismatched_actions'])} actions): "
            "These actions are semantically better aligned to a different objective "
            "than their declared one."
        )
        for m in report["mismatched_actions"]:
            st.markdown(
                f"- **Action {m['action_number']}** ({m['title'][:40]}): "
                f"Declared={m['declared_objective']}, "
                f"Best fit={m['best_objective']} (score={m['best_score']:.3f})"
            )

    # Download results
    st.divider()
    results_json = {k: v for k, v in report.items()
                    if k not in ("strategic_data", "action_data")}
    st.download_button(
        label="Download Results (JSON)",
        data=json.dumps(results_json, indent=2),
        file_name="upload_alignment_report.json",
        mime="application/json",
    )

    # Clear button
    if st.button("Clear Results"):
        del st.session_state["upload_report"]
        st.rerun()


if __name__ == "__main__":
    main()
