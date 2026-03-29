"""
AI HR Agent - Streamlit UI
Replaces the Gradio interface with a Streamlit frontend.
All multi-agent logic lives in hr_agent_core.py
"""

import asyncio
import json
import uuid
import streamlit as st
from datetime import datetime

# ── Import core agent logic ──────────────────────────────────────────────────
from hr_agent_core import (
    sync_process_enhanced_ui,
    export_results_json,
    export_results_markdown,
    format_conversational_response,
    format_clarification_response,
    format_success_response,
    active_sessions,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI HR Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 Enhanced AI HR Agent v2.0")
st.markdown(
    "**Powered by LangGraph · Intent Classification · Advanced NLP · Tavily Search**"
)
st.divider()

# ── Sidebar – Company context ─────────────────────────────────────────────────
with st.sidebar:
    st.header("🏢 Company Context")
    st.caption("Fill in for hiring-related requests")

    company_name = st.text_input("Company Name", placeholder="e.g. Acme AI")
    industry = st.selectbox(
        "Industry",
        ["", "Technology", "Healthcare", "Finance", "E-commerce", "Education", "Other"],
    )
    stage = st.selectbox(
        "Company Stage",
        ["", "Pre-seed", "Seed", "Series A", "Series B", "Series C+"],
    )
    team_size = st.number_input("Current Team Size", min_value=0, value=0, step=1)
    budget = st.text_input("Hiring Budget", placeholder="e.g. $150k/year")

    st.divider()

    # ── Session info ──────────────────────────────────────────────────────────
    st.subheader("Session")
    if st.session_state.session_id:
        st.code(st.session_state.session_id[:16] + "...", language=None)
    else:
        st.caption("No active session yet")

    if st.button("New Session", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.last_result = None
        st.rerun()

    st.divider()

    # ── Quick examples ────────────────────────────────────────────────────────
    st.subheader("Quick Examples")
    examples = [
        "Hi! How can you help me?",
        "What is a competitive salary for a ML engineer?",
        "I need to hire a founding engineer for my AI startup",
        "We need a GenAI intern and a senior backend engineer",
        "Help me build a team for a Series A fintech startup",
        "I need to hire a product manager and a data scientist",
    ]
    for example in examples:
        if st.button(example, use_container_width=True, key=f"ex_{example[:20]}"):
            st.session_state["prefill_input"] = example
            st.rerun()

# ── Main input area ───────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_input", "")
user_input = st.text_area(
    "Your message",
    value=prefill,
    placeholder="Try: 'Hi, how can you help?' or 'I need to hire a founding engineer and GenAI intern'",
    height=100,
)

col_submit, col_clear = st.columns([1, 5])
with col_submit:
    submit = st.button("Submit", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.last_result = None
        st.rerun()

# ── Process request ───────────────────────────────────────────────────────────
if submit and user_input.strip():
    with st.spinner("Processing your request through the multi-agent pipeline..."):
        try:
            status, roles, market, workflow, content = sync_process_enhanced_ui(
                user_message=user_input,
                company_name=company_name,
                industry=industry,
                stage=stage,
                team_size=team_size,
                budget=budget,
                session_id=st.session_state.session_id,
            )

            # Persist the session id returned from core
            if st.session_state.session_id is None and active_sessions:
                st.session_state.session_id = list(active_sessions.keys())[-1]

            st.session_state.last_result = {
                "status": status,
                "roles": roles,
                "market": market,
                "workflow": workflow,
                "content": content,
            }

        except Exception as e:
            st.error(f"Error processing request: {e}")

elif submit and not user_input.strip():
    st.warning("Please enter a message before submitting.")

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.last_result:
    r = st.session_state.last_result
    st.divider()
    st.subheader("Results")

    tab_status, tab_roles, tab_market, tab_workflow, tab_content = st.tabs(
        ["Status", "Roles", "Market Research", "Hiring Workflow", "Generated Content"]
    )

    with tab_status:
        st.markdown(r["status"])

    with tab_roles:
        st.markdown(r["roles"])

    with tab_market:
        st.markdown(r["market"])

    with tab_workflow:
        st.markdown(r["workflow"])

    with tab_content:
        st.markdown(r["content"])

    # ── Export ────────────────────────────────────────────────────────────────
    if st.session_state.session_id and st.session_state.session_id in active_sessions:
        st.divider()
        st.subheader("Export Results")
        col_json, col_md = st.columns(2)

        with col_json:
            json_data = export_results_json(st.session_state.session_id)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"hr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_md:
            md_data = export_results_markdown(st.session_state.session_id)
            st.download_button(
                label="Download Markdown",
                data=md_data,
                file_name=f"hr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
