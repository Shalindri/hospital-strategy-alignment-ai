"""
Healthcare Strategy Aligner — Single-Page Streamlit App.

Upload Strategic Plan and Action Plan PDFs, run the full analysis pipeline,
and view all results on one page — no sidebar navigation needed.

Launch::

    streamlit run dashboard/app.py

Author : shalindri20@gmail.com
Created: 2025-01
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils import format_llm_response, generate_pdf_report

# ---------------------------------------------------------------------------
# Custom CSS — hide sidebar completely
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Hide sidebar entirely */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

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

    /* Loading indicator */
    .stSpinner > div { border-color: var(--primary) !important; }
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def confidence_badge(conf: str) -> str:
    """Return an HTML badge for a confidence level."""
    cls = conf.lower() if conf.lower() in ("high", "medium", "low") else "medium"
    return f'<span class="badge-{cls}">{conf}</span>'


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
# Render: Sync Analysis
# ═══════════════════════════════════════════════════════════════════════

def _render_sync_analysis(data: dict) -> None:
    """Render synchronization analysis content."""
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
        st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

    # Distribution histogram + action table
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
            st.plotly_chart(fig, use_container_width=True)

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
            st.dataframe(df_actions, use_container_width=True, height=300,
                         hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# Render: Recommendations
# ═══════════════════════════════════════════════════════════════════════

def _render_recommendations(data: dict) -> None:
    """Render improvement recommendations content."""
    rag = data["rag"]
    gaps_data = data["gaps"]

    if not rag:
        st.info("RAG recommendations were not generated. Run the full pipeline to see results here.")
        return

    tab1, tab2, tab3 = st.tabs([
        "Poorly Aligned Actions",
        "Missing Actions",
        "Strategic Gaps",
    ])

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
            st.dataframe(df_weak, use_container_width=True, hide_index=True)

        if not uncovered and not weak:
            st.info("No strategic gaps detected.")


# ═══════════════════════════════════════════════════════════════════════
# Render: Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════

def _render_knowledge_graph(data: dict) -> None:
    """Render knowledge graph visualization."""
    kg = data["kg"]
    if not kg:
        st.info("Knowledge graph was not built. Run the full pipeline to see results here.")
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
    threshold = st.slider("Minimum edge weight to display", 0.0, 1.0, 0.45, 0.05,
                           key="kg_threshold")

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

    show_types = st.multiselect(
        "Node types to display",
        list(color_map.keys()),
        default=["StrategyObjective", "Action", "OntologyConcept"],
        key="kg_node_types",
    )

    visible = set()
    for i, node in enumerate(nodes):
        if node.get("node_type") in show_types:
            visible.add(i)

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
    st.plotly_chart(fig, use_container_width=True)

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
                    f"**{sug.get('concept', '')}** <- "
                    f"{sug.get('suggested_action', '')} "
                    f"(conf: {sug.get('confidence', 0):.3f})"
                )
        else:
            st.caption("No connection suggestions available.")


# ═══════════════════════════════════════════════════════════════════════
# Render: Ontology Browser
# ═══════════════════════════════════════════════════════════════════════

def _render_ontology(data: dict) -> None:
    """Render ontology browser content."""
    mappings = data["mappings"]
    gaps_data = data["gaps"]

    if not mappings:
        st.info("Ontology mappings were not computed. Run the full pipeline to see results here.")
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
        uncovered_ids = {g["concept_id"] for g in gaps_data.get("uncovered_strategy_concepts", [])}

        for parent in sorted(concept_tree.keys()):
            count = concept_action_count.get(parent, 0)
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
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# Render: Agent Insights
# ═══════════════════════════════════════════════════════════════════════

def _render_agent_insights(data: dict) -> None:
    """Render agent insights content."""
    agent_recs = data["agent_recs"]
    agent_trace = data["agent_trace"]

    if not agent_recs:
        st.info("Agent reasoning was not run. Run the full pipeline to see results here.")
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
        st.dataframe(df_issues, use_container_width=True, hide_index=True)

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
        trace_status = trace.get("final_decision", {}).get("status", "?")

        st.markdown(
            f'<div class="trace-step">'
            f'<strong>Iteration {trace["iteration"]}</strong> — '
            f'{issue_id} -> <strong>{trace_status}</strong>'
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
# Render: Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════

def _render_evaluation(data: dict) -> None:
    """Render evaluation metrics content."""
    alignment = data["alignment"]

    orphan_detected = set(alignment.get("orphan_actions", []))
    poorly_aligned = set(alignment.get("poorly_aligned_actions", []))

    st.subheader("Alignment Detection Summary")

    # Compute metrics for different methods
    methods = {
        "Embedding Similarity (Orphan Detection)": orphan_detected,
        "Embedding Similarity (Poor Alignment)": poorly_aligned,
    }

    # Add ontology-based detection (if available)
    gaps_data = data["gaps"]
    weak_actions = set()
    for w in gaps_data.get("weak_actions", []):
        act_id = w.get("action_id", "")
        num_str = act_id.replace("action_", "")
        if num_str.isdigit():
            weak_actions.add(int(num_str))
    if weak_actions:
        methods["Ontology Mapping (Weak Actions)"] = weak_actions

    # Agent-based detection (if available)
    agent_recs = data["agent_recs"]
    agent_detected = set()
    for issue in agent_recs.get("diagnosed_issues_summary", []):
        if issue.get("type") == "orphan_action":
            num_str = issue["issue_id"].replace("orphan_action_", "")
            if num_str.isdigit():
                agent_detected.add(int(num_str))
    if agent_detected:
        methods["Agent Reasoner (Orphan Detection)"] = agent_detected

    if weak_actions:
        combined = orphan_detected | weak_actions
        methods["Combined (Embedding + Ontology)"] = combined

    # Detection counts
    det_rows = []
    for method_name, detected in methods.items():
        det_rows.append({
            "Method": method_name,
            "Flagged Actions": len(detected),
            "Actions": ", ".join(str(a) for a in sorted(detected)) if detected else "None",
        })
    df_det = pd.DataFrame(det_rows)
    st.dataframe(df_det, use_container_width=True, hide_index=True)

    # Cross-method agreement
    st.subheader("Cross-Method Agreement")
    all_flagged = set()
    for detected in methods.values():
        all_flagged |= detected
    if all_flagged:
        agreement_rows = []
        for act_num in sorted(all_flagged):
            flagged_by = [name for name, det in methods.items() if act_num in det]
            agreement_rows.append({
                "Action": act_num,
                "Flagged By": len(flagged_by),
                "Methods": ", ".join(m.split("(")[0].strip() for m in flagged_by),
            })
        df_agree = pd.DataFrame(agreement_rows)
        st.dataframe(df_agree, use_container_width=True, hide_index=True)
    else:
        st.success("No misalignment detected by any method.")

    # Mismatch details
    misaligned_details = alignment.get("mismatched_actions", [])
    if misaligned_details:
        st.divider()
        st.subheader("Declared vs Best Objective Mismatches")
        df_mis = pd.DataFrame([{
            "Action": m["action_number"],
            "Title": m["title"][:45],
            "Declared": m["declared_objective"],
            "Best Match": m["best_objective"],
            "Best Score": f"{m['best_score']:.3f}",
        } for m in misaligned_details])
        st.dataframe(df_mis, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════════

def _run_full_pipeline_with_progress(report: dict, progress, start_pct: int = 0) -> None:
    """Run the 4-step pipeline with a shared progress bar."""
    from dashboard.pipeline_runner import (
        run_dynamic_ontology, run_dynamic_kg,
        run_dynamic_rag, run_dynamic_agent,
    )

    remaining = 100 - start_pct
    step_size = remaining // 4

    # Step 1/4
    progress.progress(start_pct + 5, text="Step 1/4: Ontology mapping...")
    mappings, gaps = run_dynamic_ontology(report)
    st.session_state["dynamic_mappings"] = mappings
    st.session_state["dynamic_gaps"] = gaps
    progress.progress(start_pct + step_size, text="Step 1/4: Ontology mapping... done")

    # Step 2/4
    progress.progress(start_pct + step_size + 5, text="Step 2/4: Knowledge graph...")
    kg = run_dynamic_kg(report, mappings)
    st.session_state["dynamic_kg"] = kg
    progress.progress(start_pct + step_size * 2, text="Step 2/4: Knowledge graph... done")

    # Step 3/4
    progress.progress(start_pct + step_size * 2 + 5, text="Step 3/4: RAG recommendations (LLM)...")
    rag = run_dynamic_rag(report)
    st.session_state["dynamic_rag"] = rag
    progress.progress(start_pct + step_size * 3, text="Step 3/4: RAG recommendations... done")

    # Step 4/4
    progress.progress(start_pct + step_size * 3 + 5, text="Step 4/4: Agent reasoning (LLM)...")
    agent_recs, agent_trace = run_dynamic_agent(report, mappings, kg)
    st.session_state["dynamic_agent_recs"] = agent_recs
    st.session_state["dynamic_agent_trace"] = agent_trace
    progress.progress(100, text="Pipeline complete!")


def _run_upload_analysis(strategic_file, action_file) -> None:
    """Process uploaded PDFs and run alignment + full pipeline."""
    from src.pdf_processor import (
        extract_strategic_plan_from_pdf,
        extract_action_plan_from_pdf,
    )
    from src.dynamic_analyzer import run_dynamic_analysis

    progress = st.progress(0, text="Starting analysis...")

    try:
        # PDF extraction
        progress.progress(5, text="Extracting strategic plan with AI...")
        strategic_bytes = strategic_file.read()
        strategic_data = extract_strategic_plan_from_pdf(strategic_bytes)
        n_obj = len(strategic_data.get("objectives", []))
        progress.progress(15, text=f"Found {n_obj} strategic objectives. Extracting action plan...")

        action_bytes = action_file.read()
        action_data = extract_action_plan_from_pdf(action_bytes)
        n_act = len(action_data.get("actions", []))
        progress.progress(30, text=f"Found {n_act} actions. Computing alignment...")

        # Alignment analysis
        report = run_dynamic_analysis(strategic_data, action_data)
        st.session_state["upload_report"] = report
        progress.progress(40, text="Alignment complete. Running full pipeline...")

        # Full pipeline (ontology, KG, RAG, agent)
        _run_full_pipeline_with_progress(report, progress, start_pct=40)

        st.session_state["pipeline_done"] = True
        st.session_state.pop("pipeline_running", None)
        st.rerun()

    except ValueError as e:
        st.session_state.pop("pipeline_running", None)
        st.error(f"Parsing error: {e}")
    except ConnectionError:
        st.session_state.pop("pipeline_running", None)
        st.error("Cannot connect to Ollama. Make sure it is running.")
    except Exception as e:
        st.session_state.pop("pipeline_running", None)
        st.error(f"Analysis failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Main app — single page
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Single-page Streamlit application entry point."""
    st.set_page_config(
        page_title="Healthcare Strategy Aligner",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────
    st.title("Healthcare Strategy Aligner")

    pipeline_done = st.session_state.get("pipeline_done", False)
    has_report = "upload_report" in st.session_state

    # ── Upload section ────────────────────────────────────────────
    if not pipeline_done:
        _show_upload_form()
    else:
        # Show summary bar + export + clear
        _show_summary_bar()

    # ── Results (only after pipeline completes) ───────────────────
    if pipeline_done:
        st.divider()
        _show_results()


def _show_upload_form() -> None:
    """Render the upload form and trigger analysis."""
    st.markdown(
        "Upload your **Strategic Plan** and **Action Plan** PDFs. "
        "The system will extract content with AI, compute alignment, "
        "and run the full analysis pipeline automatically."
    )

    # Check Ollama availability
    @st.cache_data(ttl=60, show_spinner=False)
    def _check_ollama():
        from src.pdf_processor import check_ollama_available
        return check_ollama_available()

    ollama_ok = _check_ollama()

    if not ollama_ok:
        st.error(
            "Ollama not running. Start with `ollama serve` and "
            "`ollama pull llama3.1:8b`."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        strategic_file = st.file_uploader(
            "Strategic Plan (PDF)",
            type=["pdf"],
            key="strategic_pdf",
        )
    with col2:
        action_file = st.file_uploader(
            "Action Plan (PDF)",
            type=["pdf"],
            key="action_pdf",
        )

    if strategic_file and action_file:
        if st.session_state.get("pipeline_running"):
            st.info("Analysis running... please wait.")
        elif st.button("Run Analysis", type="primary", use_container_width=True):
            st.session_state["pipeline_running"] = True
            _run_upload_analysis(strategic_file, action_file)
    else:
        st.caption("Upload both PDFs to begin.")


def _show_summary_bar() -> None:
    """Show summary metrics, export buttons, and clear button."""
    from dashboard.data_adapter import build_data_dict

    report = st.session_state["upload_report"]
    data = build_data_dict(report, dict(st.session_state))

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(make_metric_card(
            "Overall Score",
            f"{report['overall_score']:.1%}",
            report["overall_classification"],
        ), unsafe_allow_html=True)
    with m2:
        st.markdown(make_metric_card(
            "Strategic Objectives",
            str(len(report["objective_alignments"])),
        ), unsafe_allow_html=True)
    with m3:
        st.markdown(make_metric_card(
            "Action Items",
            str(len(report["action_alignments"])),
        ), unsafe_allow_html=True)
    with m4:
        st.markdown(make_metric_card(
            "Orphan Actions",
            str(len(report.get("orphan_actions", []))),
            "Below alignment threshold",
        ), unsafe_allow_html=True)

    # Export + clear buttons
    b1, b2, b3 = st.columns(3)
    with b1:
        pdf_bytes = generate_pdf_report(data)
        if pdf_bytes:
            st.download_button(
                label="Download Report (PDF)",
                data=pdf_bytes,
                file_name="isps_alignment_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    with b2:
        results_json = {k: v for k, v in report.items()
                        if k not in ("strategic_data", "action_data")}
        st.download_button(
            label="Download Results (JSON)",
            data=json.dumps(results_json, indent=2),
            file_name="alignment_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with b3:
        if st.button("Clear / Upload New", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key in ("upload_report", "pipeline_done", "pipeline_running") \
                        or key.startswith("dynamic_"):
                    del st.session_state[key]
            st.rerun()


def _show_results() -> None:
    """Show all analysis results in tabs."""
    from dashboard.data_adapter import build_data_dict

    report = st.session_state["upload_report"]
    data = build_data_dict(report, dict(st.session_state))

    tabs = st.tabs([
        "Synchronization Analysis",
        "Improvement Recommendations",
        "Knowledge Graph",
        "Ontology Browser",
        "Agent Insights",
        "Evaluation Metrics",
    ])

    with tabs[0]:
        _render_sync_analysis(data)

    with tabs[1]:
        _render_recommendations(data)

    with tabs[2]:
        _render_knowledge_graph(data)

    with tabs[3]:
        _render_ontology(data)

    with tabs[4]:
        _render_agent_insights(data)

    with tabs[5]:
        _render_evaluation(data)


if __name__ == "__main__":
    main()
