"""Streamlit front end for the RAG pipeline.

This file contains NO retrieval, caching, or generation logic of its own —
it only calls the real pipeline modules (optimizer.py, retriever.py,
memory_reservoir.py, llm.py) and narrates what each one actually returned,
step by step, so the mechanics of the pipeline are visible rather than
hidden behind a single "answer" box.
"""

import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import query_gen
from optimizer import optimize_prompt, parse_queries
from retriever import ChromaRetriever
from llm import GeminiRAG
from memory_reservoir import MemoryKnowledgeReservoir


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Pipeline Explorer", page_icon="◆", layout="wide")


# --------------------------------------------------------------------------
# Flat black theme — solid colors only, no gradients anywhere.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #000000;
        --panel: #0a0a0a;
        --panel-2: #111111;
        --border: #262626;
        --text: #e5e5e5;
        --muted: #888888;
        --hit: #4ade80;
        --miss: #f59e0b;
    }

    .stApp { background-color: var(--bg); }
    [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid var(--border); }
    [data-testid="stHeader"] { background-color: var(--bg); }

    h1, h2, h3, h4 { color: var(--text) !important; font-weight: 700; letter-spacing: 0.01em; }
    p, li, span, label { color: var(--text); }

    hr { border-color: var(--border); }

    .step-card {
        background-color: var(--panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
    }
    .step-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.2rem; }
    .step-desc { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.5rem; }

    .flow-map {
        text-align: center;
        font-size: 0.82rem;
        color: var(--muted);
        letter-spacing: 0.02em;
    }

    .pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        font-size: 0.78rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        color: var(--text);
        background-color: var(--panel-2);
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid;
    }
    .badge-hit { color: var(--hit); border-color: var(--hit); background-color: #0d1a10; }
    .badge-miss { color: var(--miss); border-color: var(--miss); background-color: #1a1408; }

    .stButton>button {
        background-color: var(--panel-2);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 6px;
        box-shadow: none;
    }
    .stButton>button:hover {
        background-color: #1c1c1c;
        border-color: #4a4a4a;
        color: var(--text);
    }

    .stProgress > div > div > div > div {
        background-color: var(--text) !important;
        background-image: none !important;
    }

    [data-testid="stMetricValue"] { color: var(--text); }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Component descriptions — grounded in the actual code, reused in the
# sidebar and inline next to each pipeline step.
# --------------------------------------------------------------------------
COMPONENTS = {
    "optimizer": {
        "label": "Query Optimizer",
        "file": "optimizer.py",
        "desc": (
            "An LLM call (OpenAI) turns the raw question into several diverse "
            "retrieval queries, covering different semantic angles instead of "
            "one literal rewording: **fact-decomposition** (split multi-fact "
            "questions), **broader-context** (step back to the general topic), "
            "**hypothetical-answer** (HyDE-style — a fake passage that sounds "
            "like the answer), **terminology-expansion** (spell out jargon / "
            "acronyms), and an **entity-only fallback** (just the core "
            "subject). This exists because a single query has an information "
            "'plateau' in vector search — different angles surface different "
            "chunks."
        ),
    },
    "retriever": {
        "label": "Retriever",
        "file": "retriever.py",
        "desc": (
            "Runs every generated query against the ChromaDB `documents` "
            "collection, then merges the results by document id, keeping "
            "track of which queries surfaced each document. A document "
            "found by more queries is a weak signal that it's more broadly "
            "relevant to the question."
        ),
    },
    "memory": {
        "label": "Memory Reservoir",
        "file": "memory_reservoir.py",
        "desc": (
            "A separate ChromaDB collection (`memory`, distinct from "
            "`documents`) that semantically caches past question/answer "
            "pairs. Before doing any retrieval or generation, the new "
            "question is compared against cached entries; if the closest "
            "one is within the distance threshold, its content is reused "
            "directly — skipping the OpenAI and Gemini calls entirely."
        ),
    },
    "llm": {
        "label": "Gemini RAG",
        "file": "llm.py",
        "desc": (
            "Concatenates every retrieved chunk into one context block and "
            "asks Gemini to answer strictly from that context, instructed "
            "not to use outside knowledge — the defining constraint of a RAG "
            "system."
        ),
    },
}

SAMPLE_QUESTIONS = {
    "Cardiac — electrical signals": (
        "What role do electrical signals play in the cardiac cycle, and "
        "how does a doctor visually measure this electrical activity?"
    ),
    "Cardiac — ACE inhibitor side effects": (
        "What are the common side effects of ACE inhibitor medications?"
    ),
    "Cardiac — blood pressure targets": (
        "What is the optimal adult blood pressure target according to "
        "Hypertension Canada?"
    ),
    "Ferrari — top speed": (
        "What is the top speed of the Ferrari Testarossa and its "
        "0-100 km/h acceleration time?"
    ),
    "Ferrari — design team": (
        "Who designed the Ferrari Testarossa and what studio were they from?"
    ),
}


# --------------------------------------------------------------------------
# Cached resources — one instance per server process, reused across runs
# so state (like memory-cache stats) persists as you interact with the app.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_retriever():
    return ChromaRetriever(db_path="./chroma_db", collection_name="documents")


@st.cache_resource(show_spinner=False)
def get_rag(api_key):
    return GeminiRAG(api_key)


@st.cache_resource(show_spinner=False)
def get_memory():
    return MemoryKnowledgeReservoir(db_path="./chroma_db")


def run_pipeline(question, top_k, threshold, force_regenerate):
    """Execute the real pipeline and collect what each stage produced."""
    result = {"question": question}

    memory = get_memory()
    memory.distance_threshold = threshold

    cached = memory.retrieve(question, top_k=3)
    best = cached[0] if cached else None
    cache_hit = bool(best and best["is_relevant"])
    result["cache_candidates"] = cached
    result["best_memory"] = best
    result["cache_hit"] = cache_hit
    result["force_regenerate"] = force_regenerate

    if cache_hit and not force_regenerate:
        result["source"] = "memory"
        result["answer"] = best["content"]
        result["stats_after"] = memory.statistics()
        return result

    result["source"] = "generated"

    f_t, max_l = query_gen.filter_tokens(question)
    queries = query_gen.generate_queries(question=question,max_l=max_l)
    # result["raw_queries"] = raw_queries
    result["queries"] = queries

    retriever = get_retriever()

    per_query_results = {q: retriever.retrieve(q, top_k=top_k) for q in queries}
    result["per_query_results"] = per_query_results

    documents = retriever.retrieve_multiple(queries, top_k=top_k)
    result["documents"] = documents

    rag = get_rag(GOOGLE_API_KEY)
    answer = rag.answer(question, documents)
    result["answer"] = answer
    result["prompt_template"] = rag.prompt.template

    memory.add(title=question, content=answer, source="generated", query=question)
    result["stats_after"] = memory.statistics()
    return result


# --------------------------------------------------------------------------
# Sidebar — pipeline docs, config, and live cache stats.
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Pipeline components")
    for key in ["memory", "optimizer", "retriever", "llm"]:
        c = COMPONENTS[key]
        with st.expander(f"{c['label']}  ·  {c['file']}"):
            st.write(c["desc"])

    st.markdown("---")
    st.markdown("### Configuration")
    top_k = st.slider("Documents per query (top_k)", 1, 10, 5)
    threshold = st.slider(
        "Memory relevance threshold (max distance)", 0.0, 2.0, 1.0, 0.05,
        help="Lower = stricter cache matching. Passed straight to "
             "MemoryKnowledgeReservoir.distance_threshold.",
    )
    force_regenerate = st.checkbox(
        "Force regenerate (bypass cache)",
        help="Runs the full optimizer → retrieval → Gemini path even if a "
             "cache hit is found, then overwrites that cache entry.",
    )

    st.markdown("---")
    st.markdown("### API keys")
    st.write(f"{'✓' if GOOGLE_API_KEY else '✗'} GOOGLE_API_KEY")
    st.write(f"{'✓' if OPENAI_API_KEY else '✗'} OPENAI_API_KEY")

    st.markdown("---")
    st.markdown("### Memory cache stats")
    stats = get_memory().statistics()
    s1, s2 = st.columns(2)
    s1.metric("Hits", stats["memory_hits"])
    s2.metric("Misses", stats["memory_misses"])
    st.metric("Hit rate", f"{stats['hit_rate']:.0%}")

    if st.button("🗑 Clear memory cache"):
        mem = get_memory()
        existing_ids = mem.collection.get()["ids"]
        if existing_ids:
            mem.collection.delete(ids=existing_ids)
        mem.total_memory_queries = 0
        mem.memory_hits = 0
        mem.memory_misses = 0
        st.success(f"Cleared {len(existing_ids)} cached entrie(s).")
        st.rerun()


# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
st.markdown("## RAG Pipeline Explorer")
st.markdown(
    '<p style="color:#888;">Multi-query retrieval-augmented generation, with a semantic '
    "answer cache in front of it. Every step below calls the actual project "
    "code — nothing is simulated.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="step-card flow-map">
    1 · QUESTION &nbsp;→&nbsp; 2 · MEMORY CHECK &nbsp;→&nbsp; 3 · QUERY OPTIMIZER
    &nbsp;→&nbsp; 4 · RETRIEVAL &nbsp;→&nbsp; 5 · GEMINI ANSWER &nbsp;→&nbsp; 6 · CACHE UPDATE
    </div>
    """,
    unsafe_allow_html=True,
)

missing_keys = [k for k, v in [("GOOGLE_API_KEY", GOOGLE_API_KEY), ("OPENAI_API_KEY", OPENAI_API_KEY)] if not v]
if missing_keys:
    st.markdown(
        f'<div class="step-card" style="border-color:#f59e0b;">'
        f'Missing from .env: <b>{", ".join(missing_keys)}</b>. '
        f"The pipeline cannot run without them, but you can still read the "
        f"component explanations in the sidebar."
        f"</div>",
        unsafe_allow_html=True,
    )

if "question_input" not in st.session_state:
    st.session_state["question_input"] = list(SAMPLE_QUESTIONS.values())[0]


def _apply_sample():
    choice = st.session_state["sample_choice"]
    if choice != "Custom question":
        st.session_state["question_input"] = SAMPLE_QUESTIONS[choice]


st.selectbox(
    "Sample question",
    list(SAMPLE_QUESTIONS) + ["Custom question"],
    key="sample_choice",
    on_change=_apply_sample,
)
st.text_area("Question sent to the pipeline", key="question_input", height=80)
question = st.session_state["question_input"].strip()

run_clicked = st.button("▶ Run pipeline", disabled=bool(missing_keys), type="primary")

if run_clicked:
    if not question:
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Running pipeline..."):
                st.session_state["result"] = run_pipeline(question, top_k, threshold, force_regenerate)
        except Exception as e:
            st.session_state["result"] = None
            st.error(f"Pipeline failed: {e}")
            st.exception(e)


# --------------------------------------------------------------------------
# Results — rendered from session_state so they survive sidebar reruns.
# --------------------------------------------------------------------------
result = st.session_state.get("result")

if result:
    st.markdown("---")

    # Step 1 — Memory cache lookup
    c = COMPONENTS["memory"]
    st.markdown(
        f'<div class="step-card"><div class="step-title">Step 1 · Memory Cache Lookup</div>'
        f'<div class="step-desc">{c["file"]} — checked before any retrieval or generation happens.</div>',
        unsafe_allow_html=True,
    )
    candidates = result["cache_candidates"]
    if candidates:
        df = pd.DataFrame(
            [
                {
                    "Title (cached question)": m["title"],
                    "Distance": round(m["distance"], 4) if m["distance"] is not None else None,
                    "Within threshold": m["is_relevant"],
                }
                for m in candidates
            ]
        )
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.caption("Memory collection is empty — nothing to compare against.")

    if result["cache_hit"] and result["force_regenerate"]:
        st.markdown('<span class="badge badge-miss">CACHE HIT — BYPASSED</span>', unsafe_allow_html=True)
        st.caption("A relevant cached answer was found, but 'Force regenerate' is enabled, so the full pipeline ran anyway.")
    elif result["cache_hit"]:
        st.markdown('<span class="badge badge-hit">CACHE HIT</span>', unsafe_allow_html=True)
        st.caption(f"Closest cached entry is within the distance threshold ({threshold}) — reused directly, no API calls made.")
    else:
        st.markdown('<span class="badge badge-miss">CACHE MISS</span>', unsafe_allow_html=True)
        st.caption("No cached entry was close enough — continuing to full retrieval + generation.")
    st.markdown("</div>", unsafe_allow_html=True)

    if result["source"] == "memory":
        c = COMPONENTS["llm"]
        st.markdown(
            f'<div class="step-card"><div class="step-title">Final Answer (served from memory)</div>'
            f'<div class="step-desc">No optimizer or Gemini call was made this run.</div>'
            f'<p>{result["answer"]}</p></div>',
            unsafe_allow_html=True,
        )
    else:
        # Step 2 — Query optimizer
        c = COMPONENTS["optimizer"]
        st.markdown(
            f'<div class="step-card"><div class="step-title">Step 2 · Query Optimizer</div>'
            f'<div class="step-desc">{c["file"]} — one OpenAI call, ordered by intent: fact-decomposition → '
            f'broader-context → hypothetical-answer (HyDE) → terminology-expansion → entity-only fallback '
            f'(some intents are skipped by the LLM if not applicable).</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Raw LLM output"):
            st.code(result["queries"], language="text")
        st.markdown("Parsed queries:")
        st.markdown(
            "".join(f'<span class="pill">Q{i}: {q}</span>' for i, q in enumerate(result["queries"], start=1)),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Step 3 — Retrieval
        c = COMPONENTS["retriever"]
        st.markdown(
            f'<div class="step-card"><div class="step-title">Step 3 · Multi-Query Retrieval</div>'
            f'<div class="step-desc">{c["file"]} — each query below hit ChromaDB independently, then results '
            f'were merged by document id.</div>',
            unsafe_allow_html=True,
        )
        for q, docs in result["per_query_results"].items():
            with st.expander(f'"{q}" → {len(docs)} doc(s)'):
                for d in docs:
                    dist = f"{d['distance']:.4f}" if d["distance"] is not None else "n/a"
                    st.markdown(f"**{d['id']}** · distance {dist}")
                    st.caption(d["content"][:220] + ("…" if len(d["content"]) > 220 else ""))

        st.markdown(f"**Merged result: {len(result['documents'])} unique document(s)**")
        merged_df = pd.DataFrame(
            [
                {
                    "Doc ID": d["id"],
                    "Retrieved by (# queries)": len(d["retrieved_by"]),
                    "Distance": round(d["distance"], 4) if d["distance"] is not None else None,
                    "Content preview": d["content"][:120] + ("…" if len(d["content"]) > 120 else ""),
                }
                for d in sorted(result["documents"], key=lambda d: -len(d["retrieved_by"]))
            ]
        )
        st.dataframe(merged_df, hide_index=True, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        # Step 4 — Gemini answer
        c = COMPONENTS["llm"]
        context_chars = sum(len(d["content"]) for d in result["documents"])
        st.markdown(
            f'<div class="step-card"><div class="step-title">Step 4 · Answer Generation (Gemini)</div>'
            f'<div class="step-desc">{c["file"]} — all {len(result["documents"])} retrieved chunks '
            f'({context_chars:,} characters of context) are concatenated and Gemini is instructed to '
            f"answer using ONLY that context.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Prompt template used"):
            st.code(result["prompt_template"], language="text")
        st.markdown(f"**Answer:**\n\n{result['answer']}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Step 5 — Memory update
        c = COMPONENTS["memory"]
        st.markdown(
            f'<div class="step-card"><div class="step-title">Step 5 · Memory Cache Update</div>'
            f'<div class="step-desc">{c["file"]} — the question/answer pair was upserted into the `memory` '
            f"collection under a hash of the question, so an equivalent future question can hit the cache "
            f"instead of repeating steps 2–4.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    stats = result["stats_after"]
    st.markdown(
        f'<div class="step-card"><div class="step-title">Cache stats after this run</div>'
        f'<span class="pill">Total queries: {stats["total_memory_queries"]}</span>'
        f'<span class="pill">Hits: {stats["memory_hits"]}</span>'
        f'<span class="pill">Misses: {stats["memory_misses"]}</span>'
        f'<span class="pill">Hit rate: {stats["hit_rate"]:.0%}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "This UI calls optimizer.py, retriever.py, memory_reservoir.py, and llm.py directly — "
    "no pipeline logic is reimplemented here."
)
