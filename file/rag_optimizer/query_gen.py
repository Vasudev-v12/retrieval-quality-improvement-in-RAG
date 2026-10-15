# all-MiniLM-L6-V2
'''all-MiniLM-L6-v2 is a lightweight, open-source sentence-transformer model that converts
text (sentences, phrases, or short paragraphs) into a 384-dimensional dense vector space.'''


# en_core_web_sm
'''en_core_web_sm is a small, default English language pipeline for the spaCy Natural Language Processing (NLP) library in Python'''

"""Turn a question into the set of queries sent to the retriever.

``build_queries`` is the entry point every caller should use: it takes the
raw question, does its own token filtering, and returns the queries together
with the filtered form it derived them from.  Callers used to each decide
whether to pass the raw or the filtered question, and they disagreed, so the
same question produced different retrieval queries depending on which entry
point ran it.

Queries are returned in a deterministic order (the original question first,
then key phrases by descending score) and each one is labeled with the intent
that produced it, so logs and the UI can say *why* a query exists.
"""

import re
from typing import Any, Optional

# Expansion is skipped below this many content tokens: there is nothing for
# KeyBERT to find in a two-word question, and the n-gram range cannot be
# satisfied.
DEFAULT_MIN_TOKENS_FOR_EXPANSION = 4
# Explicit, tunable KeyBERT parameters. The previous implementation derived
# all three from logarithms of the token count, which forbade 1- and 2-word
# key phrases on a typical question and asked for impossible ranges on short
# ones. Tune these against nabl_112a_rag_evaluation_30.json via eval.py.
DEFAULT_TOP_N = 3
DEFAULT_NGRAM_RANGE = (1, 3)

SPACY_MODEL = "en_core_web_sm"
KEYWORD_MODEL = "all-MiniLM-L6-v2"

INTENT_ORIGINAL = "original"
INTENT_KEYPHRASE = "keyphrase"
INTENT_ENTITY = "entity"

# Tokens spaCy calls stop words that change what a question *means* in this
# corpus. Dropping "not" makes "what should I not take with nitroglycerin"
# identical to its opposite, which matters for a document full of explicit
# contraindications.
PRESERVED_STOPWORDS = frozenset(
    {
        "not", "no", "never", "none", "nor", "neither", "cannot", "without",
        "except", "unless", "less", "more", "under", "over", "above", "below",
        "between", "before", "after", "during", "than",
    }
)
# Punctuation that carries meaning inside a measurement: 1.8 mmol/L,
# < 120 / 80 mmHg, A1c < 7.0%.
MEASUREMENT_PUNCT = frozenset({".", "/", "%", "<", ">", "=", "-", "+", "~"})

_NUMERIC = re.compile(r"\d")

_nlp = None
_keyword_model = None


def get_nlp():
    """Load spaCy on first use.

    Loading at import time cost every consumer of this module a model load
    plus two progress prints, including ones that never generate queries.
    """
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load(SPACY_MODEL)
    return _nlp


def get_keyword_model():
    """Load KeyBERT on first use (see :func:`get_nlp`)."""
    global _keyword_model
    if _keyword_model is None:
        from keybert import KeyBERT

        _keyword_model = KeyBERT(model=KEYWORD_MODEL)
    return _keyword_model


def _is_meaningful(token) -> bool:
    """Keep a token unless it is genuinely noise.

    Stop words and punctuation are dropped, except negations and comparison
    words (which invert meaning) and anything numeric or part of a
    measurement (which is usually the answer being asked for).
    """
    text = token.text
    lowered = text.casefold()

    if lowered in PRESERVED_STOPWORDS:
        return True
    if _NUMERIC.search(text):
        return True
    if token.is_punct:
        return text in MEASUREMENT_PUNCT
    return not token.is_stop


def filter_tokens(question: str):
    """Return (filtered_question, content_token_count).

    Kept for the ``__main__`` demo and for callers that only want the filtered
    form; the pipeline should call :func:`build_queries` instead, which does
    this itself.
    """
    doc = get_nlp()(question)
    filtered_tokens = [token.text for token in doc if _is_meaningful(token)]
    return " ".join(filtered_tokens).strip(), len(filtered_tokens)


def _entity_query(question: str) -> Optional[str]:
    """Bare subject of the question, as a fallback for over-specific queries.

    Uses the longest named entity, else the longest noun chunk, so an
    over-constrained question can still reach the chunk that merely mentions
    its subject.
    """
    doc = get_nlp()(question)
    candidates = [entity.text.strip() for entity in doc.ents]
    candidates += [chunk.text.strip() for chunk in doc.noun_chunks]
    candidates = [text for text in candidates if text]
    if not candidates:
        return None
    return max(candidates, key=len)


def build_queries(
    question: str,
    top_n: int = DEFAULT_TOP_N,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    min_tokens_for_expansion: int = DEFAULT_MIN_TOKENS_FOR_EXPANSION,
) -> dict[str, Any]:
    """Build the retrieval queries for `question`.

    Returns ``{"question", "filtered", "token_count", "expanded", "queries"}``
    where each query is ``{"query", "intent", "score"}``, ordered
    deterministically: the original question first, then key phrases by
    descending score, then the entity-only fallback.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    low, high = ngram_range
    if low < 1 or high < low:
        raise ValueError("ngram_range must be (low, high) with 1 <= low <= high")

    question = question.strip()
    filtered, token_count = filter_tokens(question)

    queries: list[dict[str, Any]] = [
        {"query": question, "intent": INTENT_ORIGINAL, "score": None}
    ]
    seen = {_normalize(question)}

    expanded = token_count >= min_tokens_for_expansion and top_n > 0
    if expanded:
        keywords = get_keyword_model().extract_keywords(
            filtered,
            keyphrase_ngram_range=ngram_range,
            stop_words="english",
            top_n=top_n,
        )
        # KeyBERT returns (phrase, score) sorted by descending score; keeping
        # that order makes the query list reproducible across runs, which the
        # previous `set` did not.
        for phrase, score in keywords:
            phrase = phrase.strip()
            key = _normalize(phrase)
            if not phrase or key in seen:
                continue
            seen.add(key)
            queries.append(
                {"query": phrase, "intent": INTENT_KEYPHRASE, "score": float(score)}
            )

        entity = _entity_query(question)
        if entity and _normalize(entity) not in seen:
            seen.add(_normalize(entity))
            queries.append({"query": entity, "intent": INTENT_ENTITY, "score": None})

    return {
        "question": question,
        "filtered": filtered,
        "token_count": token_count,
        "expanded": expanded,
        "queries": queries,
    }


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def generate_queries(question: str, max_l: Optional[int] = None, **kwargs) -> list:
    """Return just the query strings for `question`.

    Thin wrapper over :func:`build_queries`. `max_l` is accepted and ignored:
    the token count is derived internally now, so a caller cannot pass a count
    that disagrees with the question it passed.
    """
    plan = build_queries(question, **kwargs)
    return [entry["query"] for entry in plan["queries"]]


if __name__ == "__main__":
    import chromadb

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="documents")

    question = "what are the reasons for persistant dry cough and low blood pressure"
    print("=========================================================================")
    while True:
        print(":::Question:::\n", question)
        plan = build_queries(question)
        print(f"filtered: {plan['filtered']}  (tokens={plan['token_count']}, "
              f"expanded={plan['expanded']})")

        print("\n:::Queries:::")
        for number, entry in enumerate(plan["queries"], start=1):
            score = "" if entry["score"] is None else f" score={entry['score']:.4f}"
            print(f"{number}) [{entry['intent']}]{score} {entry['query']}")

        print("\n:::responses:::")
        for number, entry in enumerate(plan["queries"], start=1):
            results = collection.query(query_texts=entry["query"], n_results=2)
            print(number, ") Query: ", entry["query"])
            print(results["documents"])
        print("=========================================================================")
        question = input(":::Enter question => ")


"""
What is the difference between type 1 and type 2 diabetes definitions and what is the hemoglobin A1c target for diabetic patients
How does coronary artery disease develop from plaque buildup and what are the specific arterial branches that supply blood to the heart muscle

difference between valvular stenosis and regurgitation heart valve disease
normal adult blood pressure targets according to Hypertension Canada

SCAD vs plaque heart attack
smoking cessation heart timeline
"""
