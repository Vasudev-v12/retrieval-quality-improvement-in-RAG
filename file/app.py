"""Streamlit front end for the RAG pipeline.

This file contains NO retrieval, caching, or generation logic of its own —
it only calls the real pipeline modules (optimizer.py, retriever.py,
memory_reservoir.py, llm.py) and narrates what each one actually returned,
step by step, so the mechanics of the pipeline are visible rather than
hidden behind a single "answer" box.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import rag_optimizer.query_gen as query_gen
# from optimizer import optimize_prompt, parse_queries
from rag_optimizer.hybridRag import ChromaHybridRag, compute_corpus_version
from rag_optimizer.context_expansion import EXPANSION_MODES, ContextExpander
from rag_optimizer.llm import GeminiRAG, OpenaiRAG
from rag_optimizer.memory_reservoir import build_reservoir


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
        "label": "Query Generator",
        "file": "query_gen.py",
        "desc": (
            "`build_queries()` takes the raw question, filters it to content "
            "tokens — keeping negations (`not`, `never`) and measurements "
            "(`1.8 mmol/L`), which invert or carry the answer — and then asks "
            "KeyBERT for key phrases plus an **entity-only fallback** (the "
            "longest entity or noun chunk, for when the question is too "
            "specific). A single query has an information 'plateau' in vector "
            "search, so several angles surface different chunks. Every query "
            "is labeled with the intent that produced it and returned in a "
            "deterministic order. Expansion is skipped for very short "
            "questions."
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
    "expansion": {
        "label": "Context Expansion",
        "file": "context_expansion.py",
        "desc": (
            "A second retrieval pass built from the first-pass results instead "
            "of a new query string. **Centroid search** averages the dense "
            "embeddings of the top chunks into one context vector and searches "
            "with it, pulling in chunks on the same broader topic even when they "
            "share no words with the question. **Neighbor window** reads each "
            "top chunk's `document_id` / `chunk_index` metadata and fetches the "
            "chunks right before and after it, restoring text cut at a chunk "
            "boundary."
        ),
    },
    "memory": {
        "label": "Memory Reservoir",
        "file": "memory_reservoir.py",
        "desc": (
            "A separate ChromaDB collection (`memory`, distinct from "
            "`documents`) that semantically caches past question/answer "
            "pairs. Before doing any retrieval or generation, the new "
            "question is compared against cached entries. A hit must pass "
            "four guards: within the distance threshold, clearly ahead of "
            "the runner-up (margin), enough word overlap with the cached "
            "question, and not stale (expired, or generated from a different "
            "version of the `documents` collection). Entries expire by "
            "policy (stable / time-sensitive / default; 'dynamic' questions "
            "such as 'today' are never cached) and the cache is capped, "
            "evicting expired entries first and then the least recently used."
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

# Each knowledge base is a separate ChromaDB collection. The benchmark
# corpora are indexed into bench_<domain> by ragas_eval.py, so benchmark
# questions only find their answers when that collection is searched.
KNOWLEDGE_BASES = {
    "Uploaded documents": "documents",
    "Benchmark — legal": "bench_legal",
    "Benchmark — medical": "bench_medical",
    "Benchmark — laboratory": "bench_laboratory",
}
BENCH_DIR = Path(__file__).with_name("benchmarks") / "data"


@st.cache_data(show_spinner=False)
def load_sample_questions(collection_name):
    """Sample questions for a knowledge base: the benchmark's own questions
    for a bench_<domain> collection, the built-in samples otherwise."""
    if not collection_name.startswith("bench_"):
        return SAMPLE_QUESTIONS
    path = BENCH_DIR / collection_name.removeprefix("bench_") / "questions.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {f"{row['id']} — {row['question'][:70]}": row["question"] for row in rows}


# --------------------------------------------------------------------------
# Cached resources — one instance per server process, reused across runs
# so state (like memory-cache stats) persists as you interact with the app.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_retriever(collection_name="documents"):
    return ChromaHybridRag(db_path="./chroma_db", collection_name=collection_name)


@st.cache_resource(show_spinner=False)
def get_rag(api_key):
    return OpenaiRAG(api_key)#GeminiRAG(api_key)


@st.cache_resource(show_spinner=False)
def get_memory():
    # Expiration and capacity policy come from the MEMORY_* environment
    # variables (see memory_reservoir.py).
    return build_reservoir(db_path="./chroma_db")


def run_pipeline(question, top_k, threshold, force_regenerate, expansion_mode="both",
                 centroid_k=3, window=1, collection_name="documents"):
    """Execute the real pipeline and collect what each stage produced."""
    result = {"question": question, "collection": collection_name}

    memory = get_memory()
    memory.distance_threshold = threshold

    retriever = get_retriever(collection_name)
    # Cached answers are only valid against the corpus they were generated from.
    corpus_version = compute_corpus_version(retriever.collection)

    cached = memory.retrieve(question, top_k=3, corpus_version=corpus_version)
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

    query_plan = query_gen.build_queries(question)
    queries = [entry["query"] for entry in query_plan["queries"]]
    result["query_plan"] = query_plan
    result["queries"] = queries

    per_query_results = {q: retriever.retrieve(q, top_k=top_k) for q in queries}
    result["per_query_results"] = per_query_results

    first_pass = retriever.retrieve_multiple(queries, top_k=top_k)
    expander = ContextExpander(retriever, centroid_k=centroid_k, window=window)
    documents = expander.expand(first_pass, mode=expansion_mode)
    result["first_pass_count"] = len(first_pass)
    result["expansion_mode"] = expansion_mode
    result["documents"] = documents

    rag = get_rag(OPENAI_API_KEY)
    answer = rag.answer(question, documents)
    result["answer"] = answer
    result["prompt_template"] = rag.prompt.template

    # `add` returns None when the question is classified as dynamic
    # information, which is deliberately not cached.
    memory_id = memory.add(
        title=question,
        content=answer,
        source="generated",
        query=question,
        corpus_version=corpus_version,
    )
    result["cached"] = memory_id is not None
    result["stats_after"] = memory.statistics()
    return result


# --------------------------------------------------------------------------
# Sidebar — pipeline docs, config, and live cache stats.
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Pipeline components")
    for key in ["memory", "optimizer", "retriever", "expansion", "llm"]:
        c = COMPONENTS[key]
        with st.expander(f"{c['label']}  ·  {c['file']}"):
            st.write(c["desc"])

    st.markdown("---")
    st.markdown("### Configuration")
    kb_label = st.selectbox(
        "Knowledge base (collection searched)", list(KNOWLEDGE_BASES),
        help="Benchmark questions must be asked against their own benchmark "
             "collection — the uploaded-documents collection does not contain them.",
    )
    collection_name = KNOWLEDGE_BASES[kb_label]
    chunk_count = get_retriever(collection_name).collection.count()
    st.caption(f"`{collection_name}` · {chunk_count:,} chunks")
    top_k = st.slider("Documents per query (top_k)", 1, 10, 5)
    threshold = st.slider(
        "Memory relevance threshold (max distance)", 0.0, 2.0, 1.0, 0.05,
        help="Lower = stricter cache matching. Passed straight to "
             "MemoryKnowledgeReservoir.distance_threshold.",
    )
    expansion_mode = st.selectbox(
        "Context expansion (second pass)", EXPANSION_MODES, index=EXPANSION_MODES.index("both"),
        help="centroid = search with the averaged embedding of the top chunks; "
             "neighbor = add the chunks adjacent to the top chunks in their document.",
    )
    centroid_k = st.slider("Centroid search: chunks added", 0, 10, 3,
                           disabled=expansion_mode not in ("centroid", "both"))
    window = st.slider("Neighbor window (chunks each side)", 0, 3, 1,
                       disabled=expansion_mode not in ("neighbor", "both"))
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
    st.markdown("### Expiration & capacity policy")
    st.caption(
        "Read from the environment (MEMORY_* variables) at startup, so these "
        "can be changed without touching code."
    )
    st.json(get_memory().policy_summary(), expanded=False)

    st.markdown("---")
    st.markdown("### Memory cache stats")
    stats = get_memory().statistics()
    s1, s2 = st.columns(2)
    s1.metric("Hits", stats["memory_hits"])
    s2.metric("Misses", stats["memory_misses"])
    st.metric("Hit rate", f"{stats['hit_rate']:.0%}")
    s3, s4 = st.columns(2)
    s3.metric("Entries", stats["entries"])
    s4.metric("Expired purged", stats["expired_purged"])
    s5, s6 = st.columns(2)
    s5.metric("LRU evicted", stats["evicted"])
    s6.metric("Not cached (dynamic)", stats["not_cached_dynamic"])

    if st.button("🗑 Clear memory cache"):
        mem = get_memory()
        existing_ids = mem.collection.get()["ids"]
        if existing_ids:
            mem.collection.delete(ids=existing_ids)
        mem.total_memory_queries = 0
        mem.memory_hits = 0
        mem.memory_misses = 0
        mem.expired_purged = 0
        mem.evicted = 0
        mem.not_cached_dynamic = 0
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
    1 · QUESTION &nbsp;→&nbsp; 2 · MEMORY CHECK &nbsp;→&nbsp; 3 · QUERY GENERATION
    &nbsp;→&nbsp; 4 · RETRIEVAL &nbsp;→&nbsp; 5 · CONTEXT EXPANSION &nbsp;→&nbsp; 6 · ANSWER
    &nbsp;→&nbsp; 7 · CACHE UPDATE
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

if chunk_count == 0:
    st.markdown(
        f'<div class="step-card" style="border-color:#f59e0b;">'
        f"The <b>{collection_name}</b> collection is empty, so retrieval will find nothing. "
        f"Benchmark collections are built by <code>python ragas_eval.py</code>; uploaded "
        f"documents by <code>insert_doc.py</code>.</div>",
        unsafe_allow_html=True,
    )

samples = load_sample_questions(collection_name)
sample_key = f"sample_choice_{collection_name}"

if "question_input" not in st.session_state:
    st.session_state["question_input"] = next(iter(samples.values()), "")

def _apply_sample():
    choice = st.session_state[sample_key]
    if choice != "Custom question":
        st.session_state["question_input"] = samples[choice]

st.selectbox(
    "Sample question",
    list(samples) + ["Custom question"],
    key=sample_key,
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
                st.session_state["result"] = run_pipeline(
                    question, top_k, threshold, force_regenerate, expansion_mode, centroid_k, window,
                    collection_name,
                )
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
                    "Title overlap": round(m["title_overlap"], 2),
                    "Policy": m.get("policy", ""),
                    "Expires at (UTC)": m.get("expires_at") or "never",
                    "Expired": m.get("is_expired", False),
                    "Accepted": m["is_relevant"],
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
        st.caption(f"Closest cached entry passed every guard (distance ≤ {threshold}, margin, title overlap, "
                   f"not stale) — reused directly, no API calls made.")
    else:
        st.markdown('<span class="badge badge-miss">CACHE MISS</span>', unsafe_allow_html=True)
        st.caption("No cached entry passed every guard (distance, margin, title overlap, not expired, same "
                   "corpus version) — continuing to full retrieval + generation.")
    st.markdown("</div>", unsafe_allow_html=True)

    if result["source"] == "memory":
        c = COMPONENTS["llm"]
        st.markdown(
            f'<div class="step-card"><div class="step-title">Final Answer (served from memory)</div>'
            f'<div class="step-desc">No query generation or LLM call was made this run.</div>'
            f'<p>{result["answer"]}</p></div>',
            unsafe_allow_html=True,
        )
    else:
        # Step 2 — Query generation
        c = COMPONENTS["optimizer"]
        plan = result["query_plan"]
        st.markdown(
            f'<div class="step-card"><div class="step-title">Step 2 · Query Generation</div>'
            f'<div class="step-desc">{c["file"]} — the raw question is filtered to content tokens '
            f'(negations and measurements preserved), then expanded into labeled queries in a '
            f'deterministic order: original → key phrases (by score) → entity fallback.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Filtered question:** `{plan['filtered']}`")
        st.caption(
            f"{plan['token_count']} content token(s); "
            f"{'expanded' if plan['expanded'] else 'too short to expand — original question only'}"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": index,
                        "Intent": entry["intent"],
                        "Score": round(entry["score"], 4) if entry["score"] is not None else None,
                        "Query": entry["query"],
                    }
                    for index, entry in enumerate(plan["queries"], start=1)
                ]
            ),
            hide_index=True,
            width="stretch",
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

        st.markdown(f"**Merged result: {result['first_pass_count']} unique document(s)**")
        merged_df = pd.DataFrame(
            [
                {
                    "Doc ID": d["id"],
                    "Retrieved by (# queries)": len(d["retrieved_by"]),
                    "Distance": round(d["distance"], 4) if d["distance"] is not None else None,
                    "Content preview": d["content"][:120] + ("…" if len(d["content"]) > 120 else ""),
                }
                for d in result["documents"] if d["expansion"] == "seed"
            ]
        )
        st.dataframe(merged_df, hide_index=True, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        # Step 4 — Context expansion
        c = COMPONENTS["expansion"]
        added = [d for d in result["documents"] if d["expansion"] != "seed"]
        st.markdown(
            f'<div class="step-card"><div class="step-title">Step 4 · Context Expansion '
            f'({result["expansion_mode"]})</div>'
            f'<div class="step-desc">{c["file"]} — second pass over the first-pass chunks: centroid '
            f'search on their averaged embedding, and/or the neighboring chunks from the same document '
            f'(found via document_id / chunk_index metadata).</div>',
            unsafe_allow_html=True,
        )
        n_centroid = sum(d["expansion"] == "centroid" for d in added)
        n_neighbor = sum(d["expansion"] == "neighbor" for d in added)
        st.markdown(
            f'<span class="pill">First pass: {result["first_pass_count"]}</span>'
            f'<span class="pill">+ Centroid: {n_centroid}</span>'
            f'<span class="pill">+ Neighbors: {n_neighbor}</span>'
            f'<span class="pill">Final context: {len(result["documents"])}</span>',
            unsafe_allow_html=True,
        )
        if added:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Doc ID": d["id"],
                            "Added by": d["expansion"],
                            "Distance": round(d["distance"], 4) if d["distance"] is not None else None,
                            "Content preview": d["content"][:120] + ("…" if len(d["content"]) > 120 else ""),
                        }
                        for d in added
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No chunks added — expansion disabled or nothing new found.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Step 5 — Gemini answer
        c = COMPONENTS["llm"]
        context_chars = sum(len(d["content"]) for d in result["documents"])
        st.markdown(
            f'<div class="step-card"><div class="step-title">Step 5 · Answer Generation (Gemini)</div>'
            f'<div class="step-desc">{c["file"]} — all {len(result["documents"])} retrieved chunks '
            f'({context_chars:,} characters of context) are concatenated and Gemini is instructed to '
            f"answer using ONLY that context.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Prompt template used"):
            st.code(result["prompt_template"], language="text")
        st.markdown(f"**Answer:**\n\n{result['answer']}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Step 6 — Memory update
        c = COMPONENTS["memory"]
        if result.get("cached"):
            st.markdown(
                f'<div class="step-card"><div class="step-title">Step 6 · Memory Cache Update</div>'
                f'<div class="step-desc">{c["file"]} — the question/answer pair was upserted into the `memory` '
                f"collection under a hash of the question, tagged with the current corpus version and an "
                f"expiration time from the active policy, so an equivalent future question can hit the cache "
                f"instead of repeating steps 2–5 until that entry expires.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="step-card" style="border-color:#f59e0b;">'
                f'<div class="step-title">Step 6 · Not Cached</div>'
                f'<div class="step-desc">{c["file"]} — this question was classified as dynamic '
                f"information, so the answer was deliberately not written to the cache: replaying it "
                f"later would mislead more than a fresh retrieval costs.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    stats = result["stats_after"]
    st.markdown(
        f'<div class="step-card"><div class="step-title">Cache stats after this run</div>'
        f'<span class="pill">Total queries: {stats["total_memory_queries"]}</span>'
        f'<span class="pill">Hits: {stats["memory_hits"]}</span>'
        f'<span class="pill">Misses: {stats["memory_misses"]}</span>'
        f'<span class="pill">Hit rate: {stats["hit_rate"]:.0%}</span>'
        f'<span class="pill">Entries: {stats["entries"]}</span>'
        f'<span class="pill">Expired purged: {stats["expired_purged"]}</span>'
        f'<span class="pill">LRU evicted: {stats["evicted"]}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# --------------------------------------------------------------------------
# Memory cache browser — every entry currently stored in the `memory`
# collection, with the policy and expiration metadata that governs it.
# --------------------------------------------------------------------------
st.markdown("## Memory cache contents")
st.markdown(
    '<p style="color:#888;">Everything currently stored in the `memory` collection — '
    "the answers a future question can be served from, with the expiration policy "
    "attached to each one. Expired entries are shown until the next retrieval purges "
    "them.</p>",
    unsafe_allow_html=True,
)

browser_cols = st.columns([1, 1, 3])
with browser_cols[0]:
    if st.button("↻ Refresh"):
        st.rerun()
with browser_cols[1]:
    hide_expired = st.checkbox("Hide expired", value=False)

cache_entries = get_memory().list_entries(include_expired=not hide_expired)

if not cache_entries:
    st.markdown(
        '<div class="step-card">Cache is empty — run a question above, or the entries '
        "were purged on expiry. Entries appear here once an answer is generated and "
        "the policy allows caching it.</div>",
        unsafe_allow_html=True,
    )
else:
    def _countdown(entry):
        if entry["expires_at"] is None:
            return "never expires"
        remaining = entry["seconds_until_expiry"]
        if remaining is None:
            return "unknown"
        if remaining <= 0:
            return "expired"
        if remaining < 60:
            return f"{remaining:.0f}s"
        if remaining < 3600:
            return f"{remaining / 60:.0f}m"
        if remaining < 86400:
            return f"{remaining / 3600:.1f}h"
        return f"{remaining / 86400:.1f}d"

    cache_df = pd.DataFrame(
        [
            {
                "Cached question (title)": entry["title"],
                "Policy": entry["policy"],
                "Expires in": _countdown(entry),
                "Expired": entry["is_expired"],
                "Created (UTC)": (entry["created_at"] or "")[:19],
                "Last read (UTC)": (entry["last_accessed"] or "")[:19],
                "Answer preview": entry["content"][:90]
                + ("…" if len(entry["content"]) > 90 else ""),
            }
            for entry in cache_entries
        ]
    )
    st.dataframe(cache_df, hide_index=True, width="stretch")

    st.markdown("**Full entries**")
    for entry in cache_entries:
        label = f"{'⚠ EXPIRED · ' if entry['is_expired'] else ''}{entry['title'][:90]}"
        with st.expander(label):
            st.markdown("**Cached answer**")
            st.write(entry["content"])
            st.markdown("**Entry metadata**")
            st.json(
                {
                    "id": entry["id"],
                    "policy": entry["policy"],
                    "ttl_seconds": entry["ttl_seconds"],
                    "created_at": entry["created_at"],
                    "expires_at": entry["expires_at"],
                    "last_accessed": entry["last_accessed"],
                    "is_expired": entry["is_expired"],
                    "source": entry["source"],
                    "corpus_version": entry["corpus_version"],
                },
                expanded=False,
            )

st.markdown("---")
st.caption(
    "This UI calls query_gen.py, hybridRag.py, context_expansion.py, memory_reservoir.py, and llm.py "
    "directly — no pipeline logic is reimplemented here."
)
