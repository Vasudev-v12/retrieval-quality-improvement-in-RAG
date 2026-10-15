"""Checks for the memory reservoir's expiration and capacity policies.

Run with (from the `file` directory): python test_memory_policies.py
Exits non-zero if any check fails, so it also works as a CI smoke test.

Every check drives the real MemoryKnowledgeReservoir / ReservoirConfig /
build_reservoir against a temporary ChromaDB, using a deterministic offline
embedding function, so nothing here touches the application's own
``chroma_db`` collections and no API key is needed.
"""

import hashlib
import os
import sys
import time
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory  # cleanup errors ignored: ChromaDB holds the sqlite handle on Windows

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

import rag_optimizer.memory_reservoir as mr
from rag_optimizer.memory_reservoir import (
    NEVER_EXPIRES,
    POLICY_DYNAMIC,
    POLICY_STABLE,
    POLICY_TIME_SENSITIVE,
    MemoryKnowledgeReservoir,
    ReservoirConfig,
    build_reservoir,
)

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


class DemoEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic bag-of-words embedding; offline, no model download.

    Same approach as test.py: keeps distances reproducible across runs so the
    threshold-dependent checks below do not depend on a downloaded model.
    """

    def __call__(self, input: Documents) -> Embeddings:
        vectors: Embeddings = []
        for text in input:
            vector = [0.0] * 64
            for word in text.lower().split():
                token = word.strip(".,?!").encode("utf-8")
                index = int.from_bytes(hashlib.sha256(token).digest()[:4]) % len(vector)
                vector[index] += 1.0
            magnitude = sqrt(sum(value * value for value in vector))
            if magnitude:
                vector = [value / magnitude for value in vector]
            vectors.append(vector)
        return vectors


def new_reservoir(tmp, name="memory", **kwargs):
    return MemoryKnowledgeReservoir(
        db_path=str(Path(tmp) / "chroma_db"),
        collection_name=name,
        embedding_function=DemoEmbeddingFunction(),
        **kwargs,
    )


def meta_of(reservoir, entry_id):
    got = reservoir.collection.get(ids=[entry_id], include=["metadatas"])
    return got["metadatas"][0] if got["ids"] else None


# ----------------------------------------------------------------------
print("\n[1] Config defaults and env overrides")
cfg = ReservoirConfig.from_env({})
check("default TTL is 1 day", cfg.default_ttl_seconds == 86400.0, cfg.default_ttl_seconds)
check("stable TTL is 30 days", cfg.stable_ttl_seconds == 2592000.0, cfg.stable_ttl_seconds)
check("time-sensitive TTL is 1 hour", cfg.time_sensitive_ttl_seconds == 3600.0, cfg.time_sensitive_ttl_seconds)
check("max_entries defaults to 500", cfg.max_entries == 500, cfg.max_entries)
check("features are ON by default (not None)", cfg.default_ttl_seconds and cfg.max_entries)

cfg_env = ReservoirConfig.from_env(
    {
        "MEMORY_DEFAULT_TTL_SECONDS": "120",
        "MEMORY_STABLE_TTL_SECONDS": "0",
        "MEMORY_MAX_ENTRIES": "7",
        "MEMORY_SKIP_DYNAMIC": "false",
        "MEMORY_DISTANCE_THRESHOLD": "none",
    }
)
check("env sets default TTL", cfg_env.default_ttl_seconds == 120.0, cfg_env.default_ttl_seconds)
check("env 0 means never-expires", cfg_env.stable_ttl_seconds == NEVER_EXPIRES, cfg_env.stable_ttl_seconds)
check("env sets max_entries", cfg_env.max_entries == 7, cfg_env.max_entries)
check("env sets skip_dynamic off", cfg_env.skip_dynamic is False, cfg_env.skip_dynamic)
check("env 'none' disables threshold", cfg_env.distance_threshold is None, cfg_env.distance_threshold)

print("\n[2] Invalid config values fall back safely (warnings above are expected)")
cfg_bad = ReservoirConfig.from_env(
    {
        "MEMORY_DEFAULT_TTL_SECONDS": "not-a-number",
        "MEMORY_MAX_ENTRIES": "-5",
        "MEMORY_MARGIN": "abc",
        "MEMORY_TITLE_OVERLAP_FLOOR": "9",
        "MEMORY_DYNAMIC_PATTERNS": "[unclosed",
    }
)
check("bad TTL falls back to default", cfg_bad.default_ttl_seconds == 86400.0, cfg_bad.default_ttl_seconds)
check("negative max_entries falls back", cfg_bad.max_entries == 500, cfg_bad.max_entries)
check("bad margin falls back", cfg_bad.margin == 0.15, cfg_bad.margin)
check("out-of-range overlap floor falls back", cfg_bad.title_overlap_floor == 0.2, cfg_bad.title_overlap_floor)
check("bad regex falls back to default patterns", cfg_bad.dynamic_patterns == mr.DEFAULT_DYNAMIC_PATTERNS)

print("\n[3] Differentiated policies and per-entry TTL")
with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r = new_reservoir(tmp, distance_threshold=5.0)
    stable_id = r.add("Heart chambers anatomy", "Four chambers.", policy=POLICY_STABLE, query="How many chambers does the heart have")
    ts_id = r.add("Latest cholesterol targets", "LDL < 1.8 mmol/L.", query="What are the latest cholesterol targets")
    dyn_id = r.add("Clinic queue today", "Twelve patients.", query="How many patients are waiting today")
    explicit_id = r.add("Explicit ttl entry", "Body.", ttl_seconds=45, query="anything")

    check("stable policy stored", meta_of(r, stable_id)["policy"] == POLICY_STABLE, meta_of(r, stable_id)["policy"])
    check("time-sensitive classified from query", meta_of(r, ts_id)["policy"] == POLICY_TIME_SENSITIVE, meta_of(r, ts_id)["policy"])
    check("dynamic question is NOT cached", dyn_id is None, dyn_id)
    check("dynamic counter incremented", r.not_cached_dynamic == 1, r.not_cached_dynamic)
    check("explicit ttl honored", meta_of(r, explicit_id)["ttl_seconds"] == 45.0, meta_of(r, explicit_id)["ttl_seconds"])

    stable_ttl = meta_of(r, stable_id)["ttl_seconds"]
    ts_ttl = meta_of(r, ts_id)["ttl_seconds"]
    check("stable TTL longer than time-sensitive", stable_ttl > ts_ttl, f"{stable_ttl} vs {ts_ttl}")
    check("UTC expiry stored as ISO + epoch",
          meta_of(r, ts_id)["expires_at"].endswith("+00:00") and isinstance(meta_of(r, ts_id)["expires_at_epoch"], float),
          meta_of(r, ts_id)["expires_at"])
    check("creation timestamps stored", "created_at" in meta_of(r, ts_id) and "created_at_epoch" in meta_of(r, ts_id))
    check("explicit policy arg beats keyword classification",
          meta_of(r, r.add("Current guidelines doc", "x", policy=POLICY_STABLE, query="what are the current guidelines"))["policy"] == POLICY_STABLE)
    r.client._system.stop()

print("\n[4] Expiration: expired entries are a miss, and get purged")
with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r = new_reservoir(tmp, distance_threshold=5.0)
    short_id = r.add("Ferrari Testarossa top speed", "290 km/h.", ttl_seconds=1, query="Ferrari Testarossa top speed")
    hits = r.retrieve("Ferrari Testarossa top speed")
    check("fresh entry is served", hits and hits[0]["is_relevant"], hits[0]["is_relevant"] if hits else "no hits")
    check("count is 1 before expiry", r.collection.count() == 1, r.collection.count())

    time.sleep(1.2)
    hits_after = r.retrieve("Ferrari Testarossa top speed")
    check("expired entry is a cache miss", not any(h["is_relevant"] for h in hits_after), hits_after)
    check("expired entry purged from ChromaDB (lazy cleanup)", r.collection.count() == 0, r.collection.count())
    check("purge counter incremented", r.expired_purged >= 1, r.expired_purged)
    check("miss counted in statistics", r.statistics()["memory_misses"] >= 1, r.statistics())
    r.client._system.stop()

print("\n[5] TTL survives frequent access (reads do not extend retention)")
with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r = new_reservoir(tmp, distance_threshold=5.0)
    entry_id = r.add("Angina warning signs", "Chest discomfort.", ttl_seconds=2, query="Angina warning signs")
    original_expiry = meta_of(r, entry_id)["expires_at_epoch"]
    for _ in range(5):
        r.retrieve("Angina warning signs")
        time.sleep(0.25)
    still = meta_of(r, entry_id)
    check("repeated reads do not move expires_at", still and still["expires_at_epoch"] == original_expiry,
          f"{still['expires_at_epoch'] if still else None} vs {original_expiry}")
    check("reads did refresh last_accessed", still["last_accessed_epoch"] > still["created_at_epoch"],
          f"{still['last_accessed_epoch']} vs {still['created_at_epoch']}")
    time.sleep(1.0)
    hits = r.retrieve("Angina warning signs")
    check("frequently-read entry still expires on schedule", not any(h["is_relevant"] for h in hits), hits)
    r.client._system.stop()

print("\n[6] last_accessed updates only on accepted hits")
with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r = new_reservoir(tmp, distance_threshold=5.0)
    entry_id = r.add("Beta blocker side effects", "Fatigue, cold extremities.", query="Beta blocker side effects")
    before = meta_of(r, entry_id)["last_accessed_epoch"]
    time.sleep(0.05)
    r.distance_threshold = 0.0000001  # nothing can be accepted now
    rejected = r.retrieve("Beta blocker side effects")
    after_reject = meta_of(r, entry_id)["last_accessed_epoch"]
    check("candidate was returned but rejected", rejected and not rejected[0]["is_relevant"])
    check("rejected candidate NOT touched", after_reject == before, f"{after_reject} vs {before}")
    r.distance_threshold = 5.0
    time.sleep(0.05)
    accepted = r.retrieve("Beta blocker side effects")
    after_accept = meta_of(r, entry_id)["last_accessed_epoch"]
    check("accepted hit WAS touched", accepted[0]["is_relevant"] and after_accept > before, f"{after_accept} vs {before}")
    r.client._system.stop()

print("\n[7] Capacity: LRU eviction, multi-entry overflow, empty collection")
with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r = new_reservoir(tmp, distance_threshold=5.0, max_entries=3)
    check("empty collection: purge is safe", r.purge_expired() == 0)
    check("empty collection: enforce is safe", r._enforce_size_limit() == 0)
    check("empty collection: retrieve is safe", r.retrieve("nothing here") == [])

    ids = []
    for i in range(3):
        ids.append(r.add(f"Topic number {i}", f"Body {i}", query=f"Topic number {i}"))
        time.sleep(0.02)
    check("at capacity, nothing evicted", r.collection.count() == 3, r.collection.count())

    # Touch the oldest entry so it is no longer the LRU victim.
    time.sleep(0.02)
    r.retrieve("Topic number 0")
    ids.append(r.add("Topic number 3", "Body 3", query="Topic number 3"))
    surviving = set(r.collection.get(include=[])["ids"])
    check("cap enforced after overflow", r.collection.count() == 3, r.collection.count())
    check("recently-read entry survived", ids[0] in surviving, "oldest-but-recently-read was evicted")
    check("least-recently-used entry evicted", ids[1] not in surviving, "LRU victim still present")
    check("eviction counter incremented", r.evicted == 1, r.evicted)
    r.client._system.stop()

with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r = new_reservoir(tmp, distance_threshold=5.0, max_entries=2)
    for i in range(6):
        r.add(f"Bulk topic {i}", f"Body {i}", query=f"Bulk topic {i}")
    check("overflow by multiple entries respects cap", r.collection.count() == 2, r.collection.count())
    r.client._system.stop()

print("\n[8] Identical timestamps evict deterministically")
with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r = new_reservoir(tmp, distance_threshold=5.0, max_entries=2)
    frozen = r._now()
    original_now = MemoryKnowledgeReservoir._now
    MemoryKnowledgeReservoir._now = staticmethod(lambda: frozen)  # identical timestamps
    try:
        for i in range(4):
            r.add(f"Tie topic {i}", f"Body {i}", query=f"Tie topic {i}")
        count_a = r.collection.count()
        survivors_a = sorted(r.collection.get(include=[])["ids"])
    finally:
        MemoryKnowledgeReservoir._now = staticmethod(original_now)
    check("cap enforced with identical timestamps", count_a == 2, count_a)
    r.client._system.stop()

with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r2 = new_reservoir(tmp, distance_threshold=5.0, max_entries=2)
    MemoryKnowledgeReservoir._now = staticmethod(lambda: frozen)
    try:
        for i in range(4):
            r2.add(f"Tie topic {i}", f"Body {i}", query=f"Tie topic {i}")
        survivors_b = sorted(r2.collection.get(include=[])["ids"])
    finally:
        MemoryKnowledgeReservoir._now = staticmethod(original_now)
    check("tie-break is repeatable", survivors_a == survivors_b, f"{survivors_a} vs {survivors_b}")
    r2.client._system.stop()

print("\n[9] Expired entries are removed before valid ones are evicted")
with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r = new_reservoir(tmp, distance_threshold=5.0, max_entries=2)
    doomed = r.add("Short lived topic", "Body", ttl_seconds=1, query="Short lived topic")
    keep_1 = r.add("Long lived topic one", "Body", ttl_seconds=600, query="Long lived topic one")
    time.sleep(1.2)
    keep_2 = r.add("Long lived topic two", "Body", ttl_seconds=600, query="Long lived topic two")
    survivors = set(r.collection.get(include=[])["ids"])
    check("expired entry removed", doomed not in survivors, "expired entry survived")
    check("both valid entries kept", keep_1 in survivors and keep_2 in survivors, survivors)
    check("no valid entry was evicted", r.evicted == 0, r.evicted)
    r.client._system.stop()

print("\n[10] Backward compatibility of existing call sites")
with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    # Old-style construction, as test.py does it.
    r = new_reservoir(tmp, distance_threshold=0.75)
    check("old distance_threshold kwarg respected", r.distance_threshold == 0.75, r.distance_threshold)
    old_id = r.add("Ferrari Testarossa", "Mid-engine sports car from 1984.", source="retrieved document", query="Tell me about the Ferrari Testarossa")
    check("add() still returns an id", isinstance(old_id, str), old_id)
    same_id = r.add("Ferrari Testarossa", "Updated body.", source="new source")
    check("stable id still reused on upsert", same_id == old_id, f"{same_id} vs {old_id}")
    check("single entry after upsert", r.collection.count() == 1, r.collection.count())
    hits = r.retrieve("1984 Ferrari Testarossa sports car")
    check("retrieve() returns old keys", {"id", "title", "content", "distance", "metadata", "is_relevant"} <= set(hits[0]), set(hits[0]))
    check("statistics() keeps original keys",
          {"total_memory_queries", "memory_hits", "memory_misses", "hit_rate"} <= set(r.statistics()),
          set(r.statistics()))
    r.client._system.stop()

with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r = new_reservoir(tmp, distance_threshold=5.0, ttl_seconds=None, max_entries=None)
    never_id = r.add("Never expiring topic", "Body", query="Never expiring topic")
    meta = meta_of(r, never_id)
    check("ttl_seconds=None means never expires", "expires_at_epoch" not in meta, meta)
    check("ttl_seconds property reflects None", r.ttl_seconds is None, r.ttl_seconds)
    check("never-expiring entry is served", r.retrieve("Never expiring topic")[0]["is_relevant"])
    r.client._system.stop()

print("\n[11] Legacy entries (written before expiration existed) are not served")
with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r = new_reservoir(tmp, distance_threshold=5.0)
    # Simulate a pre-change row: title + timestamp only, no expiry fields.
    r.collection.upsert(
        ids=["memory_legacy"],
        documents=["Legacy cached answer."],
        metadatas=[{"title": "Legacy cardiac question", "timestamp": "2024-01-01T00:00:00+00:00"}],
    )
    hits = r.retrieve("Legacy cardiac question")
    legacy = next((h for h in hits if h["id"] == "memory_legacy"), None)
    check("legacy row is flagged expired", legacy is not None and legacy["is_expired"], legacy)
    check("legacy row is not served", legacy is not None and not legacy["is_relevant"], legacy)
    r.client._system.stop()

print("\n[12] build_reservoir applies env config (shared by CLI and Streamlit)")
with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    os.environ["MEMORY_DEFAULT_TTL_SECONDS"] = "1"
    os.environ["MEMORY_MAX_ENTRIES"] = "2"
    try:
        r = build_reservoir(
            db_path=str(Path(tmp) / "chroma_db"),
            embedding_function=DemoEmbeddingFunction(),
        )
        check("build_reservoir picked up env TTL", r.config.default_ttl_seconds == 1.0, r.config.default_ttl_seconds)
        check("build_reservoir picked up env cap", r.config.max_entries == 2, r.config.max_entries)
        r.distance_threshold = 5.0
        r.add("Env configured topic", "Body", query="Env configured topic")
        time.sleep(1.2)
        hits = r.retrieve("Env configured topic")
        check("env-configured TTL actually expires entries", not any(h["is_relevant"] for h in hits), hits)
        r.client._system.stop()
    finally:
        os.environ.pop("MEMORY_DEFAULT_TTL_SECONDS", None)
        os.environ.pop("MEMORY_MAX_ENTRIES", None)

print("\n[13] list_entries() exposes cache contents for display")
with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    r = new_reservoir(tmp, distance_threshold=5.0)
    r.add("Aortic valve function", "It stops backflow into the ventricle.",
          source="generated", query="Aortic valve function", ttl_seconds=600)
    time.sleep(0.02)
    r.add("Doomed short entry", "Body.", source="generated",
          query="Doomed short entry", ttl_seconds=1)

    entries = r.list_entries()
    check("list_entries returns every entry", len(entries) == 2, len(entries))
    fields = {
        "id", "title", "content", "policy", "source", "query", "created_at",
        "last_accessed", "expires_at", "ttl_seconds", "corpus_version",
        "is_expired", "seconds_until_expiry",
    }
    check("entries carry display fields", fields <= set(entries[0]), set(entries[0]))
    check("newest entry is first", entries[0]["title"] == "Doomed short entry", entries[0]["title"])
    check("content is the cached answer",
          any(e["content"] == "It stops backflow into the ventricle." for e in entries))
    check("countdown is positive while valid", entries[0]["seconds_until_expiry"] > 0,
          entries[0]["seconds_until_expiry"])
    check("limit is respected", len(r.list_entries(limit=1)) == 1)

    time.sleep(1.2)
    all_entries = r.list_entries()
    expired = [e for e in all_entries if e["is_expired"]]
    check("expired entry still listed and flagged", len(expired) == 1, expired)
    check("include_expired=False hides it", len(r.list_entries(include_expired=False)) == 1,
          r.list_entries(include_expired=False))
    check("listing does not purge or mutate", r.collection.count() == 2, r.collection.count())

    r.add("Never expiring entry", "Body.", query="Never expiring entry", ttl_seconds=0)
    never = next(e for e in r.list_entries() if e["title"] == "Never expiring entry")
    check("never-expiring entry has no expiry", never["expires_at"] is None and never["seconds_until_expiry"] is None,
          never)
    r.client._system.stop()

print("\n" + "=" * 70)
print(f"PASSED: {len(PASS)}    FAILED: {len(FAIL)}")
if FAIL:
    print("\nFailures:")
    for name, detail in FAIL:
        print(f"  - {name}: {detail}")
print("=" * 70)
sys.exit(1 if FAIL else 0)
