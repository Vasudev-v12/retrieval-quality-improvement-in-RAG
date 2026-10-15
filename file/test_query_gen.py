"""Checks for the query generator (rag_optimizer/query_gen.py).

Run with (from the `file` directory): python test_query_gen.py
Exits non-zero if any check fails, so it also works as a CI smoke test.

Needs spaCy and KeyBERT (see requirements.txt); if either is missing the
checks are reported as SKIPPED rather than failed.
"""

import sys

PASS = []
FAIL = []
SKIP = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


def skip(name, reason):
    SKIP.append((name, reason))
    print(f"  SKIP  {name}  ({reason})")


def models_available():
    try:
        import spacy  # noqa: F401
        import keybert  # noqa: F401
    except ImportError as error:
        return str(error)
    try:
        import rag_optimizer.query_gen as query_gen

        query_gen.get_nlp()
        query_gen.get_keyword_model()
    except Exception as error:  # model not downloaded, etc.
        return str(error)
    return None


# ----------------------------------------------------------------------
print("\n[1] Query generator: filtering preserves meaning")
import rag_optimizer.query_gen as query_gen

unavailable = models_available()
if unavailable:
    skip("query generator checks", f"spaCy/KeyBERT unavailable: {unavailable[:60]}")
else:
    negated, _ = query_gen.filter_tokens(
        "What should I not take with nitroglycerin?"
    )
    plain, _ = query_gen.filter_tokens("What should I take with nitroglycerin?")
    check("negation survives filtering", "not" in negated.casefold().split(), negated)
    check("negated and plain questions differ", negated != plain, f"{negated!r} == {plain!r}")

    numeric, _ = query_gen.filter_tokens("Is an LDL of 1.8 mmol/L the target?")
    check("decimal number survives intact", "1.8" in numeric, numeric)
    check("unit slash survives", "mmol/L" in numeric or "mmol / L" in numeric, numeric)

    never, _ = query_gen.filter_tokens("Should I never stop my medication?")
    check("'never' survives filtering", "never" in never.casefold(), never)

    print("\n[2] Query generator: deterministic, labeled, guarded")
    question = (
        "What role do electrical signals play in the cardiac cycle, and how "
        "does a doctor visually measure this electrical activity?"
    )
    first = query_gen.build_queries(question)
    second = query_gen.build_queries(question)
    check("query order is deterministic",
          [q["query"] for q in first["queries"]] == [q["query"] for q in second["queries"]],
          f"{[q['query'] for q in first['queries']]} vs {[q['query'] for q in second['queries']]}")
    check("original question is first",
          first["queries"][0]["query"] == question and first["queries"][0]["intent"] == "original",
          first["queries"][0])
    check("every query carries an intent",
          all(q["intent"] for q in first["queries"]), first["queries"])
    check("key phrases are scored descending",
          [q["score"] for q in first["queries"] if q["intent"] == "keyphrase"]
          == sorted((q["score"] for q in first["queries"] if q["intent"] == "keyphrase"), reverse=True),
          first["queries"])
    check("no duplicate queries",
          len({q["query"].casefold() for q in first["queries"]}) == len(first["queries"]),
          first["queries"])
    check("plan reports the filtered form it used",
          first["filtered"] and first["filtered"] != question, first["filtered"])

    short = query_gen.build_queries("ECG purpose")
    check("short question skips expansion", short["expanded"] is False, short)
    check("short question still yields the original query",
          [q["query"] for q in short["queries"]] == ["ECG purpose"], short["queries"])

    check("generate_queries ignores a stale max_l",
          query_gen.generate_queries(question, max_l=999)
          == [q["query"] for q in first["queries"]])

    try:
        query_gen.build_queries("")
        check("empty question rejected", False, "no error raised")
    except ValueError:
        check("empty question rejected", True)
    try:
        query_gen.build_queries(question, ngram_range=(3, 1))
        check("invalid ngram_range rejected", False, "no error raised")
    except ValueError:
        check("invalid ngram_range rejected", True)


print("\n" + "=" * 70)
print(f"PASSED: {len(PASS)}    FAILED: {len(FAIL)}    SKIPPED: {len(SKIP)}")
if SKIP:
    print("\nSkipped:")
    for name, reason in SKIP:
        print(f"  - {name}: {reason}")
if FAIL:
    print("\nFailures:")
    for name, detail in FAIL:
        print(f"  - {name}: {detail}")
print("=" * 70)
sys.exit(1 if FAIL else 0)
