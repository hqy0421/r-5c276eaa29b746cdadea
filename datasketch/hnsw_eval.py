"""Immutable archiving and relevance evaluation of HNSW retrieval runs.

This module supports comparing the quality of multiple similar-case
retrieval runs against human relevance judgments that arrive late and may
be corrected or withdrawn. The workflow is:

1. **Record runs.** Each call to :meth:`HNSWRunArchive.record_run` queries
   the live :class:`~datasketch.hnsw.HNSW` index with its *native
   approximate* search (``index.query`` is used as-is; results are not
   re-ranked, de-duplicated, or replaced by exact search) and preserves a
   self-contained snapshot of the run: the query points, and for every
   query the actual hit IDs, their order, their distances, and the sample
   values (points) as they were at query time.
2. **Judge later.** Human judgments bind to a preserved (run, query, hit)
   triple — never to the live index. Unjudged hits stay *unknown* and are
   never treated as irrelevant.
3. **Evaluate anytime.** :meth:`HNSWRunArchive.evaluate` recomputes
   per-hit, per-query, and per-run metrics purely from the preserved
   records and the current judgment state. It never re-queries the index.

Guarantees:

* Later index insertions, updates, and deletions cannot change a preserved
  run — every hit value is snapshotted at record time.
* New runs never overwrite old runs — run IDs are unique and run records
  are written once (append-only on disk when a directory is used).
* Corrections and withdrawals are recorded as judgment events; evaluation
  always reflects the latest state of each judgment, with withdrawn
  judgments returning to *unknown*.

Metrics (empty denominators are left blank, i.e. ``None``):

* ``coverage`` = labeled hits / actual hits.
* ``labeled_relevance`` = relevant hits / labeled hits.
* Run-level metrics aggregate hit counts over all queries of the run
  (micro-average), not per-query averages.

Examples:
    .. code-block:: python

        import numpy as np
        from datasketch.hnsw import HNSW
        from datasketch.hnsw_eval import HNSWRunArchive

        index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y), seed=42)
        data = np.random.random_sample((100, 8))
        for i, point in enumerate(data):
            index.insert(i, point)

        archive = HNSWRunArchive("my_archive")  # or HNSWRunArchive() in-memory
        run_id = archive.record_run(index, {"q0": data[0], "q1": data[1]}, k=10)

        # The index may change afterwards; the preserved run does not.
        index.remove(0)
        index.insert(1000, np.random.random_sample(8))

        # Judgments arrive late, may be corrected and withdrawn.
        hit = archive.get_run(run_id)["queries"][0]["hits"][0]["key"]
        archive.judge(run_id, "q0", hit, relevant=True)
        archive.judge(run_id, "q0", hit, relevant=False)  # correction
        archive.withdraw(run_id, "q0", hit)               # back to unknown

        report = archive.evaluate(run_id)      # recomputed from the archive
        all_reports = archive.evaluate_all()   # compare runs side by side
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Hashable, Optional, Union

import numpy as np

from datasketch.hnsw import HNSW

__all__ = ["HNSWRunArchive"]

#: Per-hit judgment states used in evaluation reports.
JUDGMENT_RELEVANT = "relevant"
JUDGMENT_IRRELEVANT = "irrelevant"
JUDGMENT_UNKNOWN = "unknown"

_ACTION_JUDGE = "judge"
_ACTION_WITHDRAW = "withdraw"

_RUNS_SUBDIR = "runs"
_JUDGMENTS_FILE = "judgments.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_key(key: Hashable) -> Union[str, int, float, bool, None]:
    """Normalize an HNSW key to a JSON scalar so preserved records and
    judgments can be serialized and compared reliably."""
    if isinstance(key, np.generic):
        key = key.item()
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    raise TypeError(
        f"HNSW key {key!r} of type {type(key).__name__} cannot be archived; "
        "keys must be JSON scalars (str, int, float, bool, or None)."
    )


class HNSWRunArchive:
    """Archive of HNSW retrieval runs with correctable relevance judgments
    and hit-count-based evaluation.

    Args:
        directory (Optional[Union[str, Path]]): If given, run records and
            judgment events are persisted under this directory in append-only
            files (one JSON file per run, one JSON-lines file for judgment
            events) and reloaded on construction. If None, the archive is
            kept in memory only.

    Note:
        Hit keys must be JSON scalars (``str``, ``int``, ``float``, ``bool``
        or ``None``; NumPy scalars are converted automatically). Query IDs
        are normalized to strings.
    """

    def __init__(self, directory: Optional[Union[str, Path]] = None) -> None:
        self._dir = Path(directory) if directory is not None else None
        # run_id -> preserved run record.
        self._runs: dict[str, dict] = {}
        # run_id -> query_id -> set of preserved hit keys (for binding checks).
        self._hit_keys: dict[str, dict[str, set]] = {}
        # Judgment events in arrival order (audit trail of corrections and
        # withdrawals).
        self._events: list[dict] = []
        # (run_id, query_id, key) -> current relevance (True/False); absent
        # means unknown (never judged or withdrawn).
        self._current: dict[tuple, bool] = {}
        if self._dir is not None and self._dir.exists():
            if not self._dir.is_dir():
                raise NotADirectoryError(f"{self._dir} exists and is not a directory")
            self._load()

    @property
    def directory(self) -> Optional[Path]:
        """The storage directory of this archive, or None if in-memory."""
        return self._dir

    def list_runs(self) -> list[str]:
        """Return the IDs of all preserved runs, in the order they were
        recorded."""
        return list(self._runs)

    def record_run(
        self,
        index: HNSW,
        queries: Union[Mapping[Hashable, Any], Any],
        k: Optional[int] = None,
        ef: Optional[int] = None,
        *,
        run_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Query the index and preserve the run as an immutable record.

        The index is queried with its native approximate search
        (``index.query(query_point, k=k, ef=ef)``); the returned hit IDs,
        their order, and their distances are preserved exactly as returned.
        For every hit, the sample value (``index[key]``) is snapshotted at
        this moment, so later index insertions, updates, and deletions
        cannot change the preserved record.

        Args:
            index (HNSW): The index to query.
            queries: Either a mapping of query ID to query point, or a
                sequence of query points (IDs become "0", "1", ...), or a
                1-D array (a single query) / 2-D array (one query per row).
            k (Optional[int]): Number of neighbors per query, passed to
                ``index.query``.
            ef (Optional[int]): Search breadth, passed to ``index.query``.
            run_id (Optional[str]): ID for the new run. Must not already
                exist — preserved runs cannot be overwritten. If None, an
                ID of the form ``run-000001`` is assigned.
            metadata (Optional[dict]): Free-form metadata stored with the
                run (e.g. index parameters).

        Returns:
            str: The ID of the newly preserved run.

        Raises:
            ValueError: If ``run_id`` already exists, is invalid, or query
                IDs are duplicated.
            TypeError: If a hit key is not a JSON scalar or the record is
                not JSON-serializable.
        """
        run_id = self._allocate_run_id(run_id)
        items = self._normalize_queries(queries)
        record = {
            "run_id": run_id,
            "created_at": _utc_now_iso(),
            "k": k,
            "ef": ef,
            "metadata": copy.deepcopy(metadata) if metadata is not None else {},
            "queries": [],
        }
        for query_id, point in items:
            query_point = np.asarray(point)
            # Native approximate retrieval: use the index's own search as-is.
            neighbors = index.query(query_point, k=k, ef=ef)
            hits = []
            seen = set()
            for rank, (key, distance) in enumerate(neighbors):
                nkey = _normalize_key(key)
                if nkey in seen:
                    raise ValueError(
                        f"query {query_id!r} returned hit key {nkey!r} more than "
                        "once; judgments could not be bound unambiguously"
                    )
                seen.add(nkey)
                hits.append(
                    {
                        "rank": rank,
                        "key": nkey,
                        # Snapshot the sample value as it is right now.
                        "distance": float(distance),
                        "value": np.asarray(index[key]).tolist(),
                    }
                )
            record["queries"].append(
                {
                    "query_id": query_id,
                    "query": query_point.tolist(),
                    "hits": hits,
                }
            )
        try:
            json.dumps(record)
        except TypeError as e:
            raise TypeError(f"run record is not JSON-serializable: {e}") from e
        if self._dir is not None:
            self._ensure_storage()
            # "x": a new run must never overwrite an existing run file.
            with open(self._dir / _RUNS_SUBDIR / f"{run_id}.json", "x", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        self._store_record(record)
        return run_id

    def get_run(self, run_id: str) -> dict:
        """Return a deep copy of the preserved record of a run, including
        query points and, for every hit, its rank, key, distance, and the
        sample value snapshotted at record time.

        Raises:
            KeyError: If the run ID is unknown.
        """
        if run_id not in self._runs:
            raise KeyError(f"unknown run_id {run_id!r}")
        return copy.deepcopy(self._runs[run_id])

    def judge(self, run_id: str, query_id: Hashable, key: Hashable, relevant: bool) -> None:
        """Record (or correct) a human relevance judgment for a preserved hit.

        Judgments bind to the (run, query, hit) triple of a preserved run,
        not to the live index. Judging the same hit again corrects the
        previous judgment.

        Args:
            run_id (str): The preserved run.
            query_id: The query within the run.
            key: The hit key within that query's preserved hits.
            relevant (bool): True for relevant, False for irrelevant.

        Raises:
            KeyError: If the (run, query, hit) triple is not part of the
                preserved record.
            TypeError: If ``relevant`` is not a bool.
        """
        if isinstance(relevant, np.generic):
            relevant = relevant.item()
        if not isinstance(relevant, bool):
            raise TypeError(f"relevant must be a bool, got {type(relevant).__name__}")
        query_id = str(query_id)
        key = _normalize_key(key)
        self._check_binding(run_id, query_id, key)
        self._record_event(
            {
                "run_id": run_id,
                "query_id": query_id,
                "key": key,
                "action": _ACTION_JUDGE,
                "relevant": relevant,
            }
        )

    def withdraw(self, run_id: str, query_id: Hashable, key: Hashable) -> None:
        """Withdraw the judgment for a preserved hit, returning it to
        *unknown*. Withdrawing a hit that has no judgment is a no-op.

        Raises:
            KeyError: If the (run, query, hit) triple is not part of the
                preserved record.
        """
        query_id = str(query_id)
        key = _normalize_key(key)
        self._check_binding(run_id, query_id, key)
        self._record_event(
            {
                "run_id": run_id,
                "query_id": query_id,
                "key": key,
                "action": _ACTION_WITHDRAW,
            }
        )

    def judgment_of(self, run_id: str, query_id: Hashable, key: Hashable) -> Optional[bool]:
        """Return the current judgment of a preserved hit: True (relevant),
        False (irrelevant), or None (unknown — never judged or withdrawn).

        Raises:
            KeyError: If the (run, query, hit) triple is not part of the
                preserved record.
        """
        query_id = str(query_id)
        key = _normalize_key(key)
        self._check_binding(run_id, query_id, key)
        return self._current.get((run_id, query_id, key))

    def history(self, run_id: Optional[str] = None) -> list[dict]:
        """Return the judgment events (judgments, corrections, withdrawals)
        in arrival order, optionally restricted to one run."""
        return copy.deepcopy([e for e in self._events if run_id is None or e["run_id"] == run_id])

    def evaluate(self, run_id: str) -> dict:
        """Evaluate a preserved run against the current judgment state.

        The evaluation is recomputed from the preserved record and the
        current judgments only; the index is never re-queried, so
        corrections, withdrawals, and later index changes are reflected
        without re-running retrieval.

        Args:
            run_id (str): The preserved run.

        Returns:
            dict: A report with three levels:

            * ``queries``: per query, ``num_hits`` (actual hits),
              ``num_labeled``, ``num_relevant``, ``coverage``
              (labeled / hits), ``labeled_relevance`` (relevant / labeled),
              and ``hits`` — per hit its ``rank``, ``key``, ``distance``,
              and ``judgment`` ("relevant" / "irrelevant" / "unknown").
            * ``run``: run-level totals aggregated by hit counts over all
              queries, with the same two ratios.
            * Ratios with an empty denominator are None (left blank).
        """
        if run_id not in self._runs:
            raise KeyError(f"unknown run_id {run_id!r}")
        record = self._runs[run_id]
        queries_out = []
        total_hits = total_labeled = total_relevant = 0
        for query in record["queries"]:
            query_id = query["query_id"]
            hits_out = []
            num_labeled = num_relevant = 0
            for hit in query["hits"]:
                current = self._current.get((run_id, query_id, hit["key"]))
                if current is None:
                    judgment = JUDGMENT_UNKNOWN
                else:
                    num_labeled += 1
                    num_relevant += int(current)
                    judgment = JUDGMENT_RELEVANT if current else JUDGMENT_IRRELEVANT
                hits_out.append(
                    {
                        "rank": hit["rank"],
                        "key": hit["key"],
                        "distance": hit["distance"],
                        "judgment": judgment,
                    }
                )
            num_hits = len(query["hits"])
            queries_out.append(
                {
                    "query_id": query_id,
                    "num_hits": num_hits,
                    "num_labeled": num_labeled,
                    "num_relevant": num_relevant,
                    "coverage": num_labeled / num_hits if num_hits else None,
                    "labeled_relevance": num_relevant / num_labeled if num_labeled else None,
                    "hits": hits_out,
                }
            )
            total_hits += num_hits
            total_labeled += num_labeled
            total_relevant += num_relevant
        return {
            "run_id": run_id,
            "created_at": record["created_at"],
            "k": record["k"],
            "ef": record["ef"],
            "metadata": copy.deepcopy(record["metadata"]),
            "queries": queries_out,
            "run": {
                "num_queries": len(queries_out),
                "num_hits": total_hits,
                "num_labeled": total_labeled,
                "num_relevant": total_relevant,
                "coverage": total_labeled / total_hits if total_hits else None,
                "labeled_relevance": total_relevant / total_labeled if total_labeled else None,
            },
        }

    def evaluate_all(self) -> dict[str, dict]:
        """Evaluate all preserved runs (see :meth:`evaluate`), keyed by run
        ID, in record order — convenient for comparing runs."""
        return {run_id: self.evaluate(run_id) for run_id in self._runs}

    def _allocate_run_id(self, run_id: Optional[str]) -> str:
        if run_id is None:
            n = 1
            while f"run-{n:06d}" in self._runs:
                n += 1
            return f"run-{n:06d}"
        if not isinstance(run_id, str):
            raise TypeError(f"run_id must be a str, got {type(run_id).__name__}")
        if not run_id or run_id in (".", "..") or "/" in run_id or "\\" in run_id:
            raise ValueError(f"invalid run_id {run_id!r}")
        if run_id in self._runs:
            raise ValueError(f"run_id {run_id!r} already exists; preserved runs cannot be overwritten")
        return run_id

    @staticmethod
    def _normalize_queries(queries: Union[Mapping[Hashable, Any], Any]) -> list[tuple[str, Any]]:
        if isinstance(queries, Mapping):
            items = list(queries.items())
        elif isinstance(queries, np.ndarray):
            if queries.ndim == 1:
                items = [("0", queries)]
            elif queries.ndim == 2:
                items = [(str(i), row) for i, row in enumerate(queries)]
            else:
                raise ValueError("queries array must be 1-D (single query) or 2-D (one query per row)")
        elif isinstance(queries, (str, bytes)):
            raise TypeError("queries must be a mapping or a sequence of query points, not a string")
        else:
            items = [(str(i), point) for i, point in enumerate(queries)]
        items = [(str(query_id), point) for query_id, point in items]
        query_ids = [query_id for query_id, _ in items]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError(f"duplicate query IDs: {query_ids!r}")
        return items

    def _check_binding(self, run_id: str, query_id: str, key: Hashable) -> None:
        if run_id not in self._runs:
            raise KeyError(f"unknown run_id {run_id!r}")
        queries = self._hit_keys[run_id]
        if query_id not in queries:
            raise KeyError(f"run {run_id!r} has no query {query_id!r}")
        if key not in queries[query_id]:
            raise KeyError(
                f"key {key!r} is not a preserved hit of query {query_id!r} in run "
                f"{run_id!r}; judgments must bind to a recorded (run, query, hit) triple"
            )

    def _store_record(self, record: dict) -> None:
        run_id = record["run_id"]
        self._runs[run_id] = record
        self._hit_keys[run_id] = {
            query["query_id"]: {hit["key"] for hit in query["hits"]} for query in record["queries"]
        }

    def _record_event(self, event: dict) -> None:
        event = dict(event, ts=_utc_now_iso())
        self._apply_event(event)
        if self._dir is not None:
            self._ensure_storage()
            with open(self._dir / _JUDGMENTS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

    def _apply_event(self, event: dict) -> None:
        action = event["action"]
        binding = (event["run_id"], event["query_id"], event["key"])
        if action == _ACTION_JUDGE:
            self._current[binding] = bool(event["relevant"])
        elif action == _ACTION_WITHDRAW:
            self._current.pop(binding, None)
        else:
            raise ValueError(f"unknown judgment action {action!r}")
        self._events.append(event)

    def _ensure_storage(self) -> None:
        (self._dir / _RUNS_SUBDIR).mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        runs_dir = self._dir / _RUNS_SUBDIR
        if runs_dir.is_dir():
            for path in sorted(runs_dir.glob("*.json")):
                with open(path, encoding="utf-8") as f:
                    record = json.load(f)
                run_id = record["run_id"]
                if run_id in self._runs:
                    raise ValueError(f"duplicate run record {run_id!r} in {self._dir}")
                self._store_record(record)
        judgments_file = self._dir / _JUDGMENTS_FILE
        if judgments_file.is_file():
            with open(judgments_file, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    try:
                        self._check_binding(event["run_id"], event["query_id"], event["key"])
                        self._apply_event(event)
                    except (KeyError, ValueError) as e:
                        raise ValueError(f"invalid judgment event at {judgments_file}:{lineno}: {e}") from e
