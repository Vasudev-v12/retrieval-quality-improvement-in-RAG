"""A small semantic cache for knowledge retrieved by the RAG system.

This module intentionally does not store chat history.  Each entry represents
reusable knowledge as a title-content pair and lives in its own ChromaDB
collection, separate from the static ``documents`` knowledge base.

Expiration and capacity are policy-driven and configured from the
environment, so they can be changed without editing code.  All of the
settings below are read by :meth:`ReservoirConfig.from_env`, which
:func:`build_reservoir` uses; an invalid value logs a warning and falls back
to the documented default rather than taking the application down.

    MEMORY_DEFAULT_TTL_SECONDS          default 86400 (1 day); 0 = never expires
    MEMORY_STABLE_TTL_SECONDS           default 2592000 (30 days); 0 = never expires
    MEMORY_TIME_SENSITIVE_TTL_SECONDS   default 3600 (1 hour)
    MEMORY_MAX_ENTRIES                  default 500; "none"/"0" disables the cap
    MEMORY_DISTANCE_THRESHOLD           default 1.0; "none" disables the check
    MEMORY_MARGIN                       default 0.15
    MEMORY_TITLE_OVERLAP_FLOOR          default 0.2
    MEMORY_SKIP_DYNAMIC                 default true; dynamic answers are not cached
    MEMORY_TIME_SENSITIVE_PATTERNS      ";"-separated regexes, replaces the defaults
    MEMORY_DYNAMIC_PATTERNS             ";"-separated regexes, replaces the defaults

Timestamps are always stored as UTC, both as an ISO string (for humans) and
as an epoch float (so ChromaDB can filter on them server-side).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import os
import re
import sys
from typing import Any, Optional

import chromadb

# Expiration policies. `POLICY_STABLE` is for knowledge that only goes out of
# date when the underlying corpus changes (the `corpus_version` check already
# covers that, so its TTL is only a backstop). `POLICY_DYNAMIC` is for answers
# that would mislead if replayed later, and those are not cached at all.
POLICY_DEFAULT = "default"
POLICY_STABLE = "stable"
POLICY_TIME_SENSITIVE = "time_sensitive"
POLICY_DYNAMIC = "dynamic"
POLICY_EXPLICIT = "explicit"

# A TTL of 0 seconds means "no clock expiry", not "expire immediately".
NEVER_EXPIRES = 0.0

DEFAULT_TTL_SECONDS = 86400.0
DEFAULT_STABLE_TTL_SECONDS = 2592000.0
DEFAULT_TIME_SENSITIVE_TTL_SECONDS = 3600.0
DEFAULT_MAX_ENTRIES = 500
DEFAULT_DISTANCE_THRESHOLD = 1.0
DEFAULT_MARGIN = 0.15
DEFAULT_TITLE_OVERLAP_FLOOR = 0.2

# Last-resort heuristics, used only when the caller gives no explicit policy
# and the entry metadata carries no hint. They are replaceable from the
# environment precisely so the system does not depend on this word list.
DEFAULT_TIME_SENSITIVE_PATTERNS = (
    r"\bcurrent(?:ly)?\b",
    r"\blatest\b",
    r"\bnewest\b",
    r"\bmost recent\b",
    r"\bup[\s-]?to[\s-]?date\b",
    r"\bas of\b",
    r"\bthis (?:year|month|week)\b",
    r"\b20\d{2}\b",
    r"\b(?:price|cost|fee)s?\b",
    r"\bavailab(?:le|ility)\b",
)
DEFAULT_DYNAMIC_PATTERNS = (
    r"\btoday\b",
    r"\bright now\b",
    r"\breal[\s-]?time\b",
    r"\blive\b",
    r"\byesterday\b",
    r"\btomorrow\b",
    r"\bthis (?:morning|afternoon|evening)\b",
    r"\bat the moment\b",
)


class _Unset:
    """Sentinel so an explicit ``None`` stays distinguishable from "not given"."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


def _warn(message: str) -> None:
    print(f"[memory_reservoir] {message}", file=sys.stderr)


def _env_float(
    env: dict[str, str],
    name: str,
    default: float,
    *,
    allow_none: bool = False,
) -> Optional[float]:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    text = raw.strip()
    if allow_none and text.lower() in {"none", "off", "disabled"}:
        return None
    try:
        value = float(text)
    except ValueError:
        _warn(f"{name}={raw!r} is not a number; using default {default}.")
        return default
    if value < 0:
        _warn(f"{name}={raw!r} is negative; using default {default}.")
        return default
    return value


def _env_int(
    env: dict[str, str],
    name: str,
    default: Optional[int],
    *,
    zero_disables: bool = False,
) -> Optional[int]:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    text = raw.strip()
    if text.lower() in {"none", "off", "disabled", "unlimited"}:
        return None
    try:
        value = int(text)
    except ValueError:
        _warn(f"{name}={raw!r} is not an integer; using default {default}.")
        return default
    if value < 0:
        _warn(f"{name}={raw!r} is negative; using default {default}.")
        return default
    if value == 0:
        return None if zero_disables else default
    return value


def _env_bool(env: dict[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    _warn(f"{name}={raw!r} is not a boolean; using default {default}.")
    return default


def _env_patterns(
    env: dict[str, str], name: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    patterns = []
    for candidate in raw.split(";"):
        pattern = candidate.strip()
        if not pattern:
            continue
        try:
            re.compile(pattern)
        except re.error as error:
            _warn(f"{name} pattern {pattern!r} is not valid regex ({error}); skipped.")
            continue
        patterns.append(pattern)
    if not patterns:
        _warn(f"{name} contained no usable patterns; using defaults.")
        return default
    return tuple(patterns)


@dataclass
class ReservoirConfig:
    """Expiration and capacity policy for the memory reservoir.

    One instance is shared by every entry point (see :func:`build_reservoir`),
    so the CLI and the Streamlit app never diverge on cache behavior.
    """

    default_ttl_seconds: float = DEFAULT_TTL_SECONDS
    stable_ttl_seconds: float = DEFAULT_STABLE_TTL_SECONDS
    time_sensitive_ttl_seconds: float = DEFAULT_TIME_SENSITIVE_TTL_SECONDS
    max_entries: Optional[int] = DEFAULT_MAX_ENTRIES
    distance_threshold: Optional[float] = DEFAULT_DISTANCE_THRESHOLD
    margin: float = DEFAULT_MARGIN
    title_overlap_floor: float = DEFAULT_TITLE_OVERLAP_FLOOR
    skip_dynamic: bool = True
    time_sensitive_patterns: tuple[str, ...] = DEFAULT_TIME_SENSITIVE_PATTERNS
    dynamic_patterns: tuple[str, ...] = DEFAULT_DYNAMIC_PATTERNS
    _compiled: dict[str, tuple[Any, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.margin < 0:
            _warn(f"margin={self.margin} is negative; using {DEFAULT_MARGIN}.")
            self.margin = DEFAULT_MARGIN
        if not 0.0 <= self.title_overlap_floor <= 1.0:
            _warn(
                f"title_overlap_floor={self.title_overlap_floor} is out of range; "
                f"using {DEFAULT_TITLE_OVERLAP_FLOOR}."
            )
            self.title_overlap_floor = DEFAULT_TITLE_OVERLAP_FLOOR
        if self.distance_threshold is not None and self.distance_threshold < 0:
            _warn(
                f"distance_threshold={self.distance_threshold} is negative; "
                f"using {DEFAULT_DISTANCE_THRESHOLD}."
            )
            self.distance_threshold = DEFAULT_DISTANCE_THRESHOLD
        for name in (
            "default_ttl_seconds",
            "stable_ttl_seconds",
            "time_sensitive_ttl_seconds",
        ):
            if getattr(self, name) < 0:
                _warn(
                    f"{name}={getattr(self, name)} is negative; "
                    f"treating as 0 (never expires)."
                )
                setattr(self, name, NEVER_EXPIRES)
        if self.max_entries is not None and self.max_entries <= 0:
            _warn(f"max_entries={self.max_entries} is not positive; disabling the cap.")
            self.max_entries = None
        self._compiled = {
            POLICY_TIME_SENSITIVE: tuple(
                re.compile(pattern, re.IGNORECASE)
                for pattern in self.time_sensitive_patterns
            ),
            POLICY_DYNAMIC: tuple(
                re.compile(pattern, re.IGNORECASE)
                for pattern in self.dynamic_patterns
            ),
        }

    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> "ReservoirConfig":
        """Build a config from environment variables, falling back to defaults."""
        if env is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:  # python-dotenv is optional at this layer
                pass
            env = dict(os.environ)
        return cls(
            default_ttl_seconds=_env_float(
                env, "MEMORY_DEFAULT_TTL_SECONDS", DEFAULT_TTL_SECONDS
            ),
            stable_ttl_seconds=_env_float(
                env, "MEMORY_STABLE_TTL_SECONDS", DEFAULT_STABLE_TTL_SECONDS
            ),
            time_sensitive_ttl_seconds=_env_float(
                env,
                "MEMORY_TIME_SENSITIVE_TTL_SECONDS",
                DEFAULT_TIME_SENSITIVE_TTL_SECONDS,
            ),
            max_entries=_env_int(
                env, "MEMORY_MAX_ENTRIES", DEFAULT_MAX_ENTRIES, zero_disables=True
            ),
            distance_threshold=_env_float(
                env,
                "MEMORY_DISTANCE_THRESHOLD",
                DEFAULT_DISTANCE_THRESHOLD,
                allow_none=True,
            ),
            margin=_env_float(env, "MEMORY_MARGIN", DEFAULT_MARGIN),
            title_overlap_floor=_env_float(
                env, "MEMORY_TITLE_OVERLAP_FLOOR", DEFAULT_TITLE_OVERLAP_FLOOR
            ),
            skip_dynamic=_env_bool(env, "MEMORY_SKIP_DYNAMIC", True),
            time_sensitive_patterns=_env_patterns(
                env, "MEMORY_TIME_SENSITIVE_PATTERNS", DEFAULT_TIME_SENSITIVE_PATTERNS
            ),
            dynamic_patterns=_env_patterns(
                env, "MEMORY_DYNAMIC_PATTERNS", DEFAULT_DYNAMIC_PATTERNS
            ),
        )

    def ttl_for_policy(self, policy: str) -> float:
        if policy == POLICY_STABLE:
            return self.stable_ttl_seconds
        if policy == POLICY_TIME_SENSITIVE:
            return self.time_sensitive_ttl_seconds
        if policy == POLICY_DYNAMIC:
            # Only reached when skip_dynamic is off: cache it, but briefly.
            return min(self.time_sensitive_ttl_seconds, 300.0) or 300.0
        return self.default_ttl_seconds

    def classify(self, text: Optional[str], hint: Optional[str] = None) -> str:
        """Pick a policy for `text`, preferring an explicit `hint`.

        Precedence is deliberate: a caller-supplied or metadata-supplied
        policy always wins, and the configured regexes are only consulted
        when nothing else identifies the information type. When neither says
        anything, the configurable default policy applies.
        """
        if hint:
            normalized = hint.strip().lower()
            if normalized in {
                POLICY_DEFAULT,
                POLICY_STABLE,
                POLICY_TIME_SENSITIVE,
                POLICY_DYNAMIC,
                POLICY_EXPLICIT,
            }:
                return normalized
            _warn(f"unknown policy hint {hint!r}; using {POLICY_DEFAULT}.")
            return POLICY_DEFAULT
        if not text:
            return POLICY_DEFAULT
        for pattern in self._compiled.get(POLICY_DYNAMIC, ()):
            if pattern.search(text):
                return POLICY_DYNAMIC
        for pattern in self._compiled.get(POLICY_TIME_SENSITIVE, ()):
            if pattern.search(text):
                return POLICY_TIME_SENSITIVE
        return POLICY_DEFAULT


class MemoryKnowledgeReservoir:
    """Persist and semantically retrieve cached RAG knowledge.

    ChromaDB returns a *distance*, not a universal similarity score.  The
    project's existing collection does not set ``hnsw:space``, so ChromaDB's
    default L2 distance is used (smaller values are more relevant).  The
    threshold is consequently an upper bound on distance and is kept here so
    it can be tuned without changing the RAG pipeline.

    Expiration is anchored to each entry's creation time, so an entry that is
    read often still expires on schedule; reads only refresh ``last_accessed``,
    which exists for capacity eviction, not for retention.
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "memory",
        distance_threshold: Any = _UNSET,
        embedding_function: Optional[Any] = None,
        margin: Any = _UNSET,
        title_overlap_floor: Any = _UNSET,
        ttl_seconds: Any = _UNSET,
        max_entries: Any = _UNSET,
        config: Optional[ReservoirConfig] = None,
    ) -> None:
        self.config = config if config is not None else ReservoirConfig.from_env()

        # Explicit keyword arguments still win over the configured policy, so
        # existing call sites and tests keep their old behavior.
        if not isinstance(distance_threshold, _Unset):
            if distance_threshold is not None and distance_threshold < 0:
                raise ValueError("distance_threshold must be non-negative or None")
            self.config.distance_threshold = distance_threshold
        if not isinstance(margin, _Unset):
            if margin < 0:
                raise ValueError("margin must be non-negative")
            self.config.margin = margin
        if not isinstance(title_overlap_floor, _Unset):
            if not 0.0 <= title_overlap_floor <= 1.0:
                raise ValueError("title_overlap_floor must be between 0.0 and 1.0")
            self.config.title_overlap_floor = title_overlap_floor
        if not isinstance(ttl_seconds, _Unset):
            # Backward-compatible alias: the old `ttl_seconds` argument meant
            # "expire everything after this long", i.e. the default TTL.
            self.config.default_ttl_seconds = (
                NEVER_EXPIRES if ttl_seconds is None else float(ttl_seconds)
            )
        if not isinstance(max_entries, _Unset):
            self.config.max_entries = max_entries

        self.client = chromadb.PersistentClient(path=db_path)
        # A separate dynamic collection prevents cached knowledge from
        # changing the existing static `documents` collection.
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function
        )
        self.total_memory_queries = 0
        self.memory_hits = 0
        self.memory_misses = 0
        self.expired_purged = 0
        self.evicted = 0
        self.not_cached_dynamic = 0

    # ------------------------------------------------------------------
    # Policy accessors kept as attributes for backward compatibility: the
    # existing app sets `reservoir.distance_threshold` directly from a slider.
    # ------------------------------------------------------------------
    @property
    def distance_threshold(self) -> Optional[float]:
        return self.config.distance_threshold

    @distance_threshold.setter
    def distance_threshold(self, value: Optional[float]) -> None:
        if value is not None and value < 0:
            raise ValueError("distance_threshold must be non-negative or None")
        self.config.distance_threshold = value

    @property
    def margin(self) -> float:
        return self.config.margin

    @margin.setter
    def margin(self, value: float) -> None:
        if value < 0:
            raise ValueError("margin must be non-negative")
        self.config.margin = value

    @property
    def title_overlap_floor(self) -> float:
        return self.config.title_overlap_floor

    @title_overlap_floor.setter
    def title_overlap_floor(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("title_overlap_floor must be between 0.0 and 1.0")
        self.config.title_overlap_floor = value

    @property
    def ttl_seconds(self) -> Optional[float]:
        """Default TTL in seconds; None when nothing expires on a clock."""
        ttl = self.config.default_ttl_seconds
        return None if ttl == NEVER_EXPIRES else ttl

    @ttl_seconds.setter
    def ttl_seconds(self, value: Optional[float]) -> None:
        self.config.default_ttl_seconds = (
            NEVER_EXPIRES if value is None else float(value)
        )

    @property
    def max_entries(self) -> Optional[int]:
        return self.config.max_entries

    @max_entries.setter
    def max_entries(self, value: Optional[int]) -> None:
        self.config.max_entries = value

    @staticmethod
    def _stable_id(title: str) -> str:
        """Build a deterministic ID so the same logical title is upserted."""
        normalized_title = " ".join(title.casefold().split())
        digest = hashlib.sha256(normalized_title.encode("utf-8")).hexdigest()
        return f"memory_{digest}"

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @property
    def distance_metric(self) -> str:
        """Return the configured Chroma HNSW metric (L2 when unspecified)."""
        metadata = self.collection.metadata or {}
        return str(metadata.get("hnsw:space", "l2")).lower()

    @property
    def hit_rate(self) -> float:
        if self.total_memory_queries == 0:
            return 0.0
        return self.memory_hits / self.total_memory_queries

    def statistics(self) -> dict[str, Any]:
        """Return lightweight per-instance query statistics."""
        return {
            "total_memory_queries": self.total_memory_queries,
            "memory_hits": self.memory_hits,
            "memory_misses": self.memory_misses,
            "hit_rate": self.hit_rate,
            "expired_purged": self.expired_purged,
            "evicted": self.evicted,
            "not_cached_dynamic": self.not_cached_dynamic,
            "entries": self.collection.count(),
        }

    def policy_summary(self) -> dict[str, Any]:
        """Return the active expiration/capacity policy, for display or logs."""
        return {
            "default_ttl_seconds": self.config.default_ttl_seconds,
            "stable_ttl_seconds": self.config.stable_ttl_seconds,
            "time_sensitive_ttl_seconds": self.config.time_sensitive_ttl_seconds,
            "max_entries": self.config.max_entries,
            "distance_threshold": self.config.distance_threshold,
            "margin": self.config.margin,
            "title_overlap_floor": self.config.title_overlap_floor,
            "skip_dynamic": self.config.skip_dynamic,
        }

    @staticmethod
    def _normalized_tokens(text: str) -> set[str]:
        return set(" ".join(text.casefold().split()).split())

    def _title_overlap(self, query: str, title: str) -> float:
        """Jaccard overlap between normalized query and cached-title tokens.

        A cheap lexical check alongside the embedding distance: two
        differently-worded-but-same-intent questions should still overlap
        on their key terms, while two unrelated questions that happen to
        land at a similar distance usually won't.
        """
        query_tokens = self._normalized_tokens(query)
        title_tokens = self._normalized_tokens(title)
        union = query_tokens | title_tokens
        if not union:
            return 0.0
        return len(query_tokens & title_tokens) / len(union)

    # ------------------------------------------------------------------
    # Expiration
    # ------------------------------------------------------------------
    def _resolve_policy_and_ttl(
        self,
        policy: Optional[str],
        ttl_seconds: Optional[float],
        query: Optional[str],
        title: str,
    ) -> tuple[str, float]:
        """Decide the policy and TTL for a new entry.

        An explicit `ttl_seconds` is honored as-is; otherwise the policy (from
        the caller, else classified from the query text) selects the TTL.
        """
        if ttl_seconds is not None:
            if ttl_seconds < 0:
                _warn(
                    f"ttl_seconds={ttl_seconds} is negative; "
                    f"treating as 0 (never expires)."
                )
                ttl_seconds = NEVER_EXPIRES
            return (policy or POLICY_EXPLICIT), float(ttl_seconds)
        resolved = self.config.classify(query or title, hint=policy)
        return resolved, float(self.config.ttl_for_policy(resolved))

    def _expiry_epoch(self, metadata: dict[str, Any]) -> Optional[float]:
        """Epoch seconds at which this entry expires, or None if it never does.

        New entries carry `expires_at_epoch` directly. Entries written before
        expiration existed are aged against the configured default TTL using
        whatever creation timestamp they have, so they cannot be served
        indefinitely.
        """
        explicit = metadata.get("expires_at_epoch")
        if isinstance(explicit, (int, float)):
            return float(explicit)
        if metadata.get("ttl_seconds") == NEVER_EXPIRES:
            return None

        default_ttl = self.config.default_ttl_seconds
        if default_ttl == NEVER_EXPIRES:
            return None
        created_epoch = metadata.get("created_at_epoch")
        if not isinstance(created_epoch, (int, float)):
            created_raw = metadata.get("created_at") or metadata.get("timestamp")
            if not created_raw:
                return None
            try:
                created_epoch = datetime.fromisoformat(str(created_raw)).timestamp()
            except ValueError:
                return None
        return float(created_epoch) + default_ttl

    def _is_expired(
        self, metadata: dict[str, Any], now_epoch: Optional[float] = None
    ) -> bool:
        """True when the entry's expiration timestamp has passed."""
        expiry = self._expiry_epoch(metadata)
        if expiry is None:
            return False
        if now_epoch is None:
            now_epoch = self._now().timestamp()
        return now_epoch >= expiry

    def _is_stale(self, metadata: dict[str, Any]) -> bool:
        """Backward-compatible alias for :meth:`_is_expired`."""
        return self._is_expired(metadata)

    def purge_expired(self, now_epoch: Optional[float] = None) -> int:
        """Delete entries whose stored expiration has passed; return the count.

        ChromaDB does the filtering server-side on `expires_at_epoch`, so this
        never pulls the collection into memory.
        """
        if now_epoch is None:
            now_epoch = self._now().timestamp()
        try:
            expired = self.collection.get(
                where={"expires_at_epoch": {"$lte": now_epoch}}, include=[]
            )
        except Exception as error:  # backend/version guard
            _warn(f"expired-entry lookup failed ({error}); skipping lazy purge.")
            return 0
        expired_ids = expired.get("ids", []) or []
        if not expired_ids:
            return 0
        self.collection.delete(ids=expired_ids)
        self.expired_purged += len(expired_ids)
        return len(expired_ids)

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------
    def _touch_last_accessed(
        self, ids: list[str], metadatas: list[Optional[dict[str, Any]]]
    ) -> None:
        """Refresh `last_accessed` on accepted entries, for LRU eviction.

        Only entries actually served as a cache hit are touched: merely being
        considered as a candidate must not protect an entry from eviction.
        Creation and expiration timestamps are left untouched, so a frequently
        read entry still expires on time.
        """
        if not ids:
            return
        now = self._now()
        now_iso = now.isoformat()
        now_epoch = now.timestamp()
        updated_metadatas = []
        for metadata in metadatas:
            merged = dict(metadata or {})
            merged["last_accessed"] = now_iso
            merged["last_accessed_epoch"] = now_epoch
            updated_metadatas.append(merged)
        self.collection.update(ids=ids, metadatas=updated_metadatas)

    def _enforce_size_limit(self) -> int:
        """Keep the collection within `max_entries`; return entries removed.

        Expired entries go first, so valid knowledge is only evicted when the
        cache is genuinely over capacity. The metadata scan below is the one
        unavoidable full read: ChromaDB cannot sort by metadata, so there is no
        server-side "oldest N" query. It runs only when over capacity, and asks
        for metadata only (no documents, no embeddings).
        """
        max_entries = self.config.max_entries
        if max_entries is None:
            return 0
        if self.collection.count() <= max_entries:
            return 0

        removed = self.purge_expired()
        if self.collection.count() <= max_entries:
            return removed

        existing = self.collection.get(include=["metadatas"])
        ids = existing.get("ids", []) or []
        metadatas = existing.get("metadatas", []) or []
        if not ids:
            return removed

        now_epoch = self._now().timestamp()
        # Catch entries that are expired but have no server-filterable
        # `expires_at_epoch` (written before expiration was introduced).
        stale_ids = [
            entry_id
            for entry_id, metadata in zip(ids, metadatas)
            if self._is_expired(metadata or {}, now_epoch)
        ]
        if stale_ids:
            self.collection.delete(ids=stale_ids)
            self.expired_purged += len(stale_ids)
            removed += len(stale_ids)
            stale_set = set(stale_ids)
            keep = [
                (entry_id, metadata)
                for entry_id, metadata in zip(ids, metadatas)
                if entry_id not in stale_set
            ]
        else:
            keep = list(zip(ids, metadatas))

        overflow = len(keep) - max_entries
        if overflow <= 0:
            return removed

        def recency_key(
            item: tuple[str, Optional[dict[str, Any]]]
        ) -> tuple[float, float, str]:
            entry_id, metadata = item
            metadata = metadata or {}
            last_accessed = metadata.get("last_accessed_epoch")
            created = metadata.get("created_at_epoch")
            if not isinstance(last_accessed, (int, float)):
                last_accessed = created if isinstance(created, (int, float)) else 0.0
            if not isinstance(created, (int, float)):
                created = 0.0
            # The id is the final tie-break, so entries sharing a timestamp
            # are still evicted in a stable, repeatable order.
            return (float(last_accessed), float(created), entry_id)

        evict_ids = [
            entry_id for entry_id, _ in sorted(keep, key=recency_key)[:overflow]
        ]
        if evict_ids:
            self.collection.delete(ids=evict_ids)
            self.evicted += len(evict_ids)
            removed += len(evict_ids)
        return removed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(
        self,
        title: str,
        content: str,
        source: Optional[str] = None,
        query: Optional[str] = None,
        corpus_version: Optional[str] = None,
        policy: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
    ) -> Optional[str]:
        """Add or replace a title-content memory and return its stable ID.

        Returns ``None`` when the resolved policy is `dynamic` and dynamic
        answers are not cached, because replaying such an answer later would
        mislead more than a fresh retrieval would cost.
        """
        if not title or not title.strip():
            raise ValueError("title must be a non-empty string")
        if not content or not content.strip():
            raise ValueError("content must be a non-empty string")

        resolved_policy, resolved_ttl = self._resolve_policy_and_ttl(
            policy, ttl_seconds, query, title
        )
        if resolved_policy == POLICY_DYNAMIC and self.config.skip_dynamic:
            self.not_cached_dynamic += 1
            return None

        memory_id = self._stable_id(title)
        now = self._now()
        now_iso = now.isoformat()
        now_epoch = now.timestamp()
        metadata: dict[str, Any] = {
            "title": title.strip(),
            "created_at": now_iso,
            "created_at_epoch": now_epoch,
            # `timestamp` is kept as a mirror of `created_at` so entries
            # written by earlier versions stay readable both ways.
            "timestamp": now_iso,
            "last_accessed": now_iso,
            "last_accessed_epoch": now_epoch,
            "policy": resolved_policy,
            "ttl_seconds": resolved_ttl,
        }
        if resolved_ttl != NEVER_EXPIRES:
            expires_epoch = now_epoch + resolved_ttl
            metadata["expires_at"] = datetime.fromtimestamp(
                expires_epoch, tz=timezone.utc
            ).isoformat()
            metadata["expires_at_epoch"] = expires_epoch
        if source is not None:
            metadata["source"] = source
        if query is not None:
            metadata["query"] = query
        if corpus_version is not None:
            # Ties this entry to the `documents` collection state it was
            # generated from, so a later corpus change can invalidate it.
            metadata["corpus_version"] = corpus_version

        # Upsert replaces an earlier entry with the same stable title ID,
        # avoiding duplicate versions of one logical memory.
        self.collection.upsert(
            ids=[memory_id], documents=[content.strip()], metadatas=[metadata]
        )
        self._enforce_size_limit()
        return memory_id

    def list_entries(
        self,
        include_expired: bool = True,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return cached entries, newest first, for inspection or display.

        This is a deliberate full read of the collection for operator-facing
        views; the retrieval path never calls it. Expired entries are included
        by default and flagged, so a viewer can see what is about to be purged
        rather than being shown a silently filtered cache.
        """
        got = self.collection.get(include=["documents", "metadatas"])
        ids = got.get("ids", []) or []
        documents = got.get("documents", []) or []
        metadatas = got.get("metadatas", []) or []

        now_epoch = self._now().timestamp()
        entries: list[dict[str, Any]] = []
        for index, entry_id in enumerate(ids):
            metadata = metadatas[index] or {}
            is_expired = self._is_expired(metadata, now_epoch)
            if is_expired and not include_expired:
                continue
            expiry = self._expiry_epoch(metadata)
            entries.append(
                {
                    "id": entry_id,
                    "title": metadata.get("title", ""),
                    "content": documents[index] if index < len(documents) else "",
                    "policy": metadata.get("policy", POLICY_DEFAULT),
                    "source": metadata.get("source"),
                    "query": metadata.get("query"),
                    "created_at": metadata.get("created_at") or metadata.get("timestamp"),
                    "last_accessed": metadata.get("last_accessed"),
                    "expires_at": metadata.get("expires_at"),
                    "ttl_seconds": metadata.get("ttl_seconds"),
                    "corpus_version": metadata.get("corpus_version"),
                    "is_expired": is_expired,
                    "seconds_until_expiry": None if expiry is None else expiry - now_epoch,
                }
            )

        entries.sort(key=lambda entry: entry.get("created_at") or "", reverse=True)
        if limit is not None:
            entries = entries[:limit]
        return entries

    def is_relevant(
        self,
        distance: Optional[float],
        *,
        margin_ok: bool = True,
        title_overlap: Optional[float] = None,
        is_stale: bool = False,
        corpus_matches: bool = True,
    ) -> bool:
        """Evaluate relevance using distance plus the false-positive guards.

        All guards must pass: the distance must clear `distance_threshold`,
        the hit must not be an ambiguous tie with the runner-up
        (`margin_ok`), the title must share enough terms with the query
        (`title_overlap` vs `title_overlap_floor`), and the entry must not
        be stale (expired or from a different `corpus_version`).
        """
        if distance is None:
            return False
        if self.distance_threshold is not None and distance > self.distance_threshold:
            return False
        if not margin_ok:
            return False
        if is_stale or not corpus_matches:
            return False
        if title_overlap is not None and title_overlap < self.title_overlap_floor:
            return False
        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        corpus_version: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Semantically retrieve memory entries and update hit/miss counters.

        Expired entries are purged before the query runs (lazy cleanup) and
        any that slip through are reported with `is_expired` and never marked
        relevant, so an expired answer is always a cache miss.

        `corpus_version`, when passed, invalidates any hit whose stored
        version doesn't match — including legacy entries with none at all,
        since those predate version tracking and can't be trusted against
        the current `documents` collection.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.total_memory_queries += 1
        self.purge_expired()

        count = self.collection.count()
        if count == 0:
            self.memory_misses += 1
            return []

        results = self.collection.query(
            query_texts=[query.strip()], n_results=min(top_k, count)
        )
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        now_epoch = self._now().timestamp()
        memories: list[dict[str, Any]] = []
        for index, memory_id in enumerate(ids):
            metadata = metadatas[index] or {}
            distance = distances[index] if index < len(distances) else None
            next_distance = distances[index + 1] if index + 1 < len(distances) else None

            # Only the top-1 hit is ever served (see main.py/app.py), so the
            # margin check against the runner-up only matters there.
            margin_ok = True
            if index == 0 and next_distance is not None and distance is not None:
                cleared_by_margin = (next_distance - distance) >= self.margin
                runner_up_out_of_range = (
                    self.distance_threshold is not None
                    and next_distance > self.distance_threshold
                )
                margin_ok = cleared_by_margin or runner_up_out_of_range

            title_overlap = self._title_overlap(query, metadata.get("title", ""))
            is_expired = self._is_expired(metadata, now_epoch)
            corpus_matches = (
                corpus_version is None
                or metadata.get("corpus_version") == corpus_version
            )

            memories.append(
                {
                    "id": memory_id,
                    "title": metadata.get("title", ""),
                    "content": documents[index],
                    "distance": distance,
                    "metadata": metadata,
                    "title_overlap": title_overlap,
                    "policy": metadata.get("policy", POLICY_DEFAULT),
                    "expires_at": metadata.get("expires_at"),
                    "is_expired": is_expired,
                    "is_relevant": self.is_relevant(
                        distance,
                        margin_ok=margin_ok,
                        title_overlap=title_overlap,
                        is_stale=is_expired,
                        corpus_matches=corpus_matches,
                    ),
                }
            )

        accepted = next((memory for memory in memories if memory["is_relevant"]), None)
        if accepted is not None:
            self.memory_hits += 1
            self._touch_last_accessed([accepted["id"]], [accepted["metadata"]])
        else:
            self.memory_misses += 1
        return memories


def build_reservoir(
    db_path: str = "./chroma_db",
    collection_name: str = "memory",
    embedding_function: Optional[Any] = None,
    config: Optional[ReservoirConfig] = None,
) -> MemoryKnowledgeReservoir:
    """Construct the reservoir with the environment-configured policy.

    Every entry point (CLI and Streamlit) goes through here, so expiration and
    capacity behave identically no matter how the pipeline is started.
    """
    return MemoryKnowledgeReservoir(
        db_path=db_path,
        collection_name=collection_name,
        embedding_function=embedding_function,
        config=config if config is not None else ReservoirConfig.from_env(),
    )
