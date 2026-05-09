import streamlit as st
import sys
import json
import subprocess
import time
from pathlib import Path

# Add project root so src.* imports resolve
sys.path.insert(0, str(Path(__file__).parent))

from src import config

# ============================================================================
# PAGE CONFIG — must be the first Streamlit call
# ============================================================================
st.set_page_config(
    page_title="F-16 RAG Pipeline",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — professional dark panel style
# ============================================================================
st.markdown("""
<style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #1a1f2e 0%, #22293b 100%);
    border: 1px solid #2d3548;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    border-color: #3b82f6;
}
[data-testid="metric-container"] label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7c8db5;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px;
    font-weight: 700;
    color: #e8edf8;
}

/* Section headers */
.section-header {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #ffb300;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #2d3548;
}

/* Answer panel */
.answer-panel {
    background: linear-gradient(to right, #111827, #1a2333);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 24px;
    margin: 12px 0;
    font-size: 16px;
    line-height: 1.8;
    color: #f1f5f9;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* Source chunk card */
.chunk-card {
    background: #1a1f2e;
    border: 1px solid #2d3548;
    border-left: 3px solid #10b981;
    border-radius: 8px;
    padding: 18px 24px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.7;
    color: #f1f5f9;
    transition: all 0.2s ease;
}
.chunk-card:hover {
    border-color: #10b981;
    background: #1e2538;
}
.chunk-meta {
    font-size: 11px;
    color: #5a6a8a;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #2d3548;
    font-family: 'Courier New', monospace;
}

/* Status badge */
.status-ok {
    display: inline-block;
    background: #0f2a1a;
    color: #34d399;
    border: 1px solid #065f46;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}
.status-err {
    display: inline-block;
    background: #2a0f0f;
    color: #f87171;
    border: 1px solid #7f1d1d;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}

/* Latency bar */
.latency-bar {
    height: 6px;
    border-radius: 3px;
    margin: 4px 0 12px 0;
}

/* Command block */
.cmd-block {
    background: #0d1117;
    border: 1px solid #2d3548;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #a5b4fc;
    margin: 6px 0;
}

/* Eval metric card */
.eval-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #252b3d 100%);
    border: 1px solid #2d3548;
    border-radius: 10px;
    padding: 20px;
    margin: 8px 0;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;
}
.eval-card:hover {
    transform: scale(1.02);
    border-color: #6366f1;
}
.eval-card .metric-name {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #93c5fd;
    margin-bottom: 8px;
}
.eval-card .metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
}
.eval-card .metric-desc {
    font-size: 11px;
    color: #4a5a78;
    margin-top: 4px;
}

/* Gradient text for headers */
.gradient-text {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

/* Tabs text size */
button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
    font-size: 18px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INIT
# ============================================================================
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown('<h2 style="font-size:30px;text-align: center"><span class="gradient-text">Defence RAG System </span></h3>',unsafe_allow_html=True)
    st.markdown("---")

    # System health
    st.markdown('<div class="section-header">System Health</div>', unsafe_allow_html=True)

    # Qdrant status
    try:
        import requests
        r = requests.get(f"{config.QDRANT_URL}/collections", timeout=3)
        if r.status_code == 200:
            st.markdown('<span class="status-ok">Qdrant Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">Qdrant Error</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="status-err">Qdrant Offline</span>', unsafe_allow_html=True)

    # Ollama status
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            st.markdown('<span class="status-ok">Ollama Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">Ollama Error</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="status-err">Ollama Offline</span>', unsafe_allow_html=True)

    if st.session_state.pipeline:
        st.markdown('<span class="status-ok">Pipeline Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-err">Pipeline Not Loaded</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Initialize pipeline button
    col_btn, _ = st.columns([2, 1])
    with col_btn:
        if st.button("Load Pipeline", use_container_width=True):
            with st.spinner("Loading models..."):
                try:
                    from src.orchestrator import RAGPipeline
                    st.session_state.pipeline = RAGPipeline()
                    st.success("Pipeline loaded")
                    st.rerun()
                except Exception as e:
                    st.error(f"Load failed: {e}")

    st.markdown("---")

    # Query parameters
    st.markdown('<div class="section-header">Query Parameters</div>', unsafe_allow_html=True)
    top_k = st.slider("Final chunks (Re-ranked)", min_value=1, max_value=5, value=2)
    retrieve_k = st.slider("Initial chunks", min_value=5, max_value=25, value=10)
# ============================================================================
# MAIN TABS
# ============================================================================
tab_query, tab_eval, tab_sysinfo = st.tabs([
    "Query Interface",
    "Evaluation Results",
    "Architecture",
])


# ============================================================================
# TAB 1: QUERY INTERFACE
# ============================================================================
with tab_query:
    st.markdown('## <span class="gradient-text">F-16 Technical Manual Bot</span>', unsafe_allow_html=True)
    st.markdown("F-16 Technical Assistant")
    st.markdown("---")

    if not st.session_state.pipeline:
        st.warning("Pipeline not loaded. Click 'Load Pipeline' in the sidebar to begin.")
    else:
        col_query, col_history = st.columns([2, 1])

        with col_query:
            st.markdown('<div class="section-header">Query</div>', unsafe_allow_html=True)

            def set_query(text):
                st.session_state["query_input"] = text

            query = st.text_area(
                label="Question",
                label_visibility="collapsed",
                height=100,
                placeholder="Enter your question about the F-16 technical manual...",
                key="query_input"
            )

            # Example questions
            with st.expander("Example Questions"):
                examples = [
                    "What is the maximum airspeed of the F-16?",
                    "Describe the engine start procedure.",
                    "What are the emergency procedures for hydraulic failure?",
                    "What does the HUD low speed cue indicate?",
                    "What is the fuel capacity and fuel system configuration?",
                    "What are the G-limit restrictions for the F-16?",
                    "Explain the INS alignment procedure.",
                ]
                for ex in examples:
                    st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True, on_click=set_query, args=(ex,))

            run_query = st.button(
                "Run Query",
                type="primary",
                disabled=(not st.session_state.pipeline),
                use_container_width=True
            )

            if run_query:
                q = st.session_state.get("query_input", "").strip()
                if not q:
                    st.warning("Enter a question before running a query.")
                else:
                    with st.spinner("Embedding query, searching, reranking, generating..."):
                        try:
                            answer, docs, metrics = st.session_state.pipeline.query(
                                question=q,
                                top_k=top_k,
                                retrieve_k=retrieve_k,
                            )

                            # Store result in session state
                            result = {
                                "query": q,
                                "answer": answer,
                                "docs": docs,
                                "metrics": metrics,
                                "timestamp": time.strftime("%H:%M:%S"),
                            }
                            st.session_state.last_result = result
                            st.session_state.query_history.insert(0, result)

                        except Exception as e:
                            st.error(f"Query failed: {e}")

            # Display last result
            if st.session_state.last_result:
                res = st.session_state.last_result
                m = res["metrics"]

                st.markdown("---")
                st.markdown('<div class="section-header">Answer</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-panel">{res["answer"]}</div>', unsafe_allow_html=True)

                # Latency breakdown
                st.markdown('<div class="section-header">Latency Breakdown (Local Inference)</div>', unsafe_allow_html=True)
                st.caption("Generation time is hardware-dependent. Local models (via Ollama) typically take 10-20 seconds to process large RAG contexts and generate detailed responses without relying on cloud APIs.")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Total", f"{m.total_time:.2f}s")
                with col_m2:
                    st.metric("Retrieval", f"{m.retrieval_time:.3f}s")
                with col_m3:
                    st.metric("Rerank", f"{m.rerank_time:.3f}s")
                with col_m4:
                    st.metric("Generation (Ollama)", f"{m.generation_time:.2f}s")

                # Source chunks
                st.markdown("---")
                st.markdown('<div class="section-header">Source Chunks</div>', unsafe_allow_html=True)

                if res["docs"]:
                    for i, doc in enumerate(res["docs"], 1):
                        meta = doc.get("metadata", {})
                        rerank_score = doc.get("rerank_score", 0)
                        vector_score = doc.get("vector_score", 0)

                        with st.expander(
                            f"Chunk {i}  |  Rerank Score: {rerank_score:.4f}  |  "
                            f"Page: {meta.get('page', 'N/A')}  |  "
                            f"Section: {meta.get('section_heading', 'N/A')[:40]}"
                        ):
                            st.markdown(
                                f'<div class="chunk-card">{doc["text"]}</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f'<div class="chunk-meta">'
                                f'chunk_id: {meta.get("chunk_id", "N/A")}  |  '
                                f'vector_score: {vector_score:.4f}  |  '
                                f'rerank_score: {rerank_score:.4f}  |  '
                                f'tokens: {meta.get("tokens", "N/A")}  |  '
                                f'type: {meta.get("chunk_type", "N/A")}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No source chunks returned.")

        # Query History
        with col_history:
            st.markdown('<div class="section-header">Query History</div>', unsafe_allow_html=True)

            if not st.session_state.query_history:
                st.caption("No queries yet.")
            else:
                for i, item in enumerate(st.session_state.query_history[:8]):
                    with st.expander(f"{item['timestamp']}  {item['query'][:35]}..."):
                        st.caption(f"Time: {item['metrics'].total_time:.2f}s")
                        st.markdown(item["answer"][:300] + "..." if len(item["answer"]) > 300 else item["answer"])

            if st.session_state.query_history:
                if st.button("Clear History", use_container_width=True):
                    st.session_state.query_history = []
                    st.session_state.last_result = None
                    st.rerun()


# ============================================================================
# TAB 2: EVALUATION RESULTS
# ============================================================================
with tab_eval:
    st.markdown('## <span class="gradient-text">Evaluation Results</span>', unsafe_allow_html=True)
    st.markdown("Evaluation metrics displayed here (Evaluation benchmark dataset and LLM-as-judge) ")
    st.markdown("---")

    eval_results_path = Path(config.OUTPUT_DIR) / "eval_results.json"

    if not eval_results_path.exists():
        st.info(
            "No evaluation results found. Run:\n\n"
            "`python -m eval.eval_dataset --chunks 50`\n\n"
            "`python -m eval.eval_metrics`"
        )
    else:
        try:
            with open(eval_results_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            ragas = eval_data.get("ragas_metrics", {})
            retrieval = eval_data.get("retrieval_metrics", {})
            info = eval_data.get("dataset_info", {})

            # RAGAS metrics
            st.markdown('<div class="section-header">RAGAS Metrics (0 to 1, higher is better)</div>', unsafe_allow_html=True)

            ragas_descriptions = {
                "context_precision": "Fraction of retrieved chunks that are relevant",
                "context_recall": "Coverage of ground truth by retrieved context",
                "context_relevancy": "Overall semantic relevance of contexts",
                "faithfulness": "Answer grounded in context (no hallucination)",
            }

            ragas_cols = st.columns(3)
            numeric_ragas = {k: v for k, v in ragas.items() if isinstance(v, (int, float)) and k not in ("answer_correctness", "answer_relevancy")}

            for idx, (metric, score) in enumerate(numeric_ragas.items()):
                with ragas_cols[idx % 3]:
                    desc = ragas_descriptions.get(metric, "")
                    color = "#34d399" if score >= 0.7 else "#fbbf24" if score >= 0.4 else "#f87171"
                    st.markdown(
                        f'<div class="eval-card">'
                        f'<div class="metric-name">{metric.replace("_", " ")}</div>'
                        f'<div class="metric-value" style="color:{color}">{score:.3f}</div>'
                        f'<div class="metric-desc">{desc}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # Bar chart
            if numeric_ragas:
                st.markdown("---")
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(
                        x=list(numeric_ragas.keys()),
                        y=list(numeric_ragas.values()),
                        marker_color=[
                            "#34d399" if v >= 0.7 else "#fbbf24" if v >= 0.4 else "#f87171"
                            for v in numeric_ragas.values()
                        ],
                        text=[f"{v:.3f}" for v in numeric_ragas.values()],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        paper_bgcolor="#111827",
                        plot_bgcolor="#111827",
                        font=dict(color="#c8d3e8", family="Inter"),
                        yaxis=dict(range=[0, 1.1], gridcolor="#1e2a3a"),
                        xaxis=dict(gridcolor="#1e2a3a"),
                        margin=dict(t=20, b=20),
                        height=320,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.caption("Install plotly for charts: pip install plotly")

            # Retrieval metrics
            if retrieval:
                st.markdown("---")
                st.markdown('<div class="section-header">Retrieval Score Distribution</div>', unsafe_allow_html=True)
                ret_cols = st.columns(3)
                ret_items = [(k, v) for k, v in retrieval.items() if "mean" in k.lower()]
                for idx, (metric, score) in enumerate(ret_items):
                    with ret_cols[idx % 3]:
                        st.metric(metric.replace("_", " ").title(), f"{score:.4f}")

        except Exception as e:
            st.error(f"Failed to load eval results: {e}")


# ============================================================================
# TAB 3: ARCHITECTURE
# ============================================================================
with tab_sysinfo:
    st.markdown('## <span class="gradient-text">Architecture</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Under the hood - RAG Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(to right, #111827, #1a2333); padding: 24px; border-radius: 12px; border: 1px solid #1e3a5f; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);">
        <div style="text-align: center;"><div style="color: #3b82f6; font-size: 12px; font-weight: 800;">01</div><div style="font-size: 15px; font-weight: 600; color: #e2e8f0; margin-top: 4px;">PDF Ingestion</div></div>
        <div style="color: #4a5a78; font-weight: bold; font-size: 18px;">→</div>
        <div style="text-align: center;"><div style="color: #3b82f6; font-size: 12px; font-weight: 800;">02</div><div style="font-size: 15px; font-weight: 600; color: #e2e8f0; margin-top: 4px;">Chunking</div></div>
        <div style="color: #4a5a78; font-weight: bold; font-size: 18px;">→</div>
        <div style="text-align: center;"><div style="color: #3b82f6; font-size: 12px; font-weight: 800;">03</div><div style="font-size: 15px; font-weight: 600; color: #e2e8f0; margin-top: 4px;">Embedding</div></div>
        <div style="color: #4a5a78; font-weight: bold; font-size: 18px;">→</div>
        <div style="text-align: center;"><div style="color: #3b82f6; font-size: 12px; font-weight: 800;">04</div><div style="font-size: 15px; font-weight: 600; color: #e2e8f0; margin-top: 4px;">Vector Store</div></div>
        <div style="color: #4a5a78; font-weight: bold; font-size: 18px;">→</div>
        <div style="text-align: center;"><div style="color: #10b981; font-size: 12px; font-weight: 800;">05</div><div style="font-size: 15px; font-weight: 600; color: #e2e8f0; margin-top: 4px;">ANN Retrieval</div></div>
        <div style="color: #4a5a78; font-weight: bold; font-size: 18px;">→</div>
        <div style="text-align: center;"><div style="color: #10b981; font-size: 12px; font-weight: 800;">06</div><div style="font-size: 15px; font-weight: 600; color: #e2e8f0; margin-top: 4px;">Cross-Encoder Rerank</div></div>
        <div style="color: #4a5a78; font-weight: bold; font-size: 18px;">→</div>
        <div style="text-align: center;"><div style="color: #8b5cf6; font-size: 12px; font-weight: 800;">07</div><div style="font-size: 15px; font-weight: 600; color: #e2e8f0; margin-top: 4px;">LLM Generation</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_models, col_pipeline_cfg = st.columns(2)

    with col_models:
        st.markdown('<div class="section-header">Core AI Models</div>', unsafe_allow_html=True)

        model_rows = [
            ("Vector Embedder", config.EMBED_MODEL),
            ("LLM Generator", config.GEN_MODEL),
            ("Cross-Encoder Reranker", config.RERANKER_MODEL.split("/")[-1]),
        ]

        for label, value in model_rows:
            col_l, col_v = st.columns([1, 2])
            with col_l:
                st.caption(label)
            with col_v:
                st.markdown(f'<div class="cmd-block">{value}</div>', unsafe_allow_html=True)

    with col_pipeline_cfg:
        st.markdown('<div class="section-header">Infrastructure</div>', unsafe_allow_html=True)

        pipeline_rows = [
            ("Vector Database", "Qdrant"),
            ("Collection Name", config.QDRANT_COLLECTION),
            ("Vector Dimension", str(config.EMBEDDING_DIM)),
        ]

        for label, value in pipeline_rows:
            col_l, col_v = st.columns([1, 2])
            with col_l:
                st.caption(label)
            with col_v:
                st.markdown(f'<div class="cmd-block">{value}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"F-16 Technical Assistant ")