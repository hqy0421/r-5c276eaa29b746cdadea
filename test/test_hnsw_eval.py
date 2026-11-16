import json
import tempfile
import unittest

import numpy as np

from datasketch.hnsw import HNSW
from datasketch.hnsw_eval import HNSWRunArchive


def l2_distance(x, y):
    return float(np.linalg.norm(x - y))


class TestHNSWRunArchive(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.data = self.rng.random_sample((100, 8))
        self.index = HNSW(distance_func=l2_distance, m=8, ef_construction=40, seed=7)
        for i, point in enumerate(self.data):
            self.index.insert(i, point)
        self.queries = {"q0": self.data[0], "q1": self.rng.random_sample(8)}

    def _record(self, archive, k=5, **kwargs):
        return archive.record_run(self.index, self.queries, k=k, **kwargs)

    def test_record_preserves_native_results(self):
        archive = HNSWRunArchive()
        run_id = self._record(archive, k=5)
        record = archive.get_run(run_id)

        self.assertEqual(record["run_id"], run_id)
        self.assertEqual(record["k"], 5)
        self.assertEqual([q["query_id"] for q in record["queries"]], ["q0", "q1"])
        for query_record in record["queries"]:
            query_id = query_record["query_id"]
            # The query point itself is preserved.
            self.assertEqual(query_record["query"], np.asarray(self.queries[query_id]).tolist())
            # Hits are exactly what the native approximate search returned:
            # same IDs, same order, same distances.
            expected = self.index.query(np.asarray(self.queries[query_id]), k=5)
            self.assertEqual([h["key"] for h in query_record["hits"]], [key for key, _ in expected])
            self.assertEqual([h["rank"] for h in query_record["hits"]], list(range(len(expected))))
            self.assertEqual([h["distance"] for h in query_record["hits"]], [float(d) for _, d in expected])
            # Sample values are snapshotted at record time.
            for hit in query_record["hits"]:
                self.assertEqual(hit["value"], np.asarray(self.data[hit["key"]]).tolist())

    def test_index_mutation_does_not_change_preserved_run(self):
        archive = HNSWRunArchive()
        run_id = self._record(archive, k=5)
        before = archive.get_run(run_id)
        eval_before = archive.evaluate(run_id)

        hit_key = before["queries"][0]["hits"][0]["key"]
        hit_value = list(before["queries"][0]["hits"][0]["value"])
        # Insert, update, and delete after recording.
        self.index.insert(1000, np.zeros(8))
        self.index[hit_key] = np.zeros(8)  # update a preserved hit's point
        self.index.remove(hit_key)  # delete a preserved hit

        after = archive.get_run(run_id)
        self.assertEqual(before, after)
        # The preserved sample value is the one from record time.
        self.assertEqual(after["queries"][0]["hits"][0]["value"], hit_value)
        self.assertNotEqual(after["queries"][0]["hits"][0]["value"], np.zeros(8).tolist())
        # Evaluation is recomputed from the preserved record, unaffected.
        self.assertEqual(eval_before, archive.evaluate(run_id))

    def test_new_run_does_not_overwrite_old_run(self):
        archive = HNSWRunArchive()
        run_id_1 = self._record(archive, k=5)
        # Insert an exact copy of query q1: it becomes the top hit of q1.
        self.index.insert(1000, np.array(self.queries["q1"]))
        run_id_2 = self._record(archive, k=5)
        self.assertNotEqual(run_id_1, run_id_2)
        self.assertEqual(archive.list_runs(), [run_id_1, run_id_2])
        # The old run still holds the old hits; the new run reflects the
        # changed index without touching the old record.
        hits_1 = [h["key"] for h in archive.get_run(run_id_1)["queries"][1]["hits"]]
        hits_2 = [h["key"] for h in archive.get_run(run_id_2)["queries"][1]["hits"]]
        self.assertNotIn(1000, hits_1)
        self.assertEqual(hits_2[0], 1000)
        # An explicit run_id cannot be reused.
        archive.record_run(self.index, self.queries, k=5, run_id="myrun")
        with self.assertRaises(ValueError):
            archive.record_run(self.index, self.queries, k=5, run_id="myrun")
        self.assertEqual(archive.list_runs(), [run_id_1, run_id_2, "myrun"])

    def test_judgment_must_bind_to_preserved_hit(self):
        archive = HNSWRunArchive()
        run_id = self._record(archive, k=3)
        hit_key = archive.get_run(run_id)["queries"][0]["hits"][0]["key"]
        with self.assertRaises(KeyError):
            archive.judge("no-such-run", "q0", hit_key, True)
        with self.assertRaises(KeyError):
            archive.judge(run_id, "no-such-query", hit_key, True)
        with self.assertRaises(KeyError):
            archive.judge(run_id, "q0", 999999, True)
        with self.assertRaises(TypeError):
            archive.judge(run_id, "q0", hit_key, "yes")
        with self.assertRaises(KeyError):
            archive.withdraw(run_id, "q0", 999999)
        with self.assertRaises(KeyError):
            archive.judgment_of(run_id, "q0", 999999)

    def test_unknown_is_not_irrelevant_and_metrics(self):
        archive = HNSWRunArchive()
        run_id = self._record(archive, k=4)
        record = archive.get_run(run_id)
        q0_keys = [h["key"] for h in record["queries"][0]["hits"]]
        q1_keys = [h["key"] for h in record["queries"][1]["hits"]]

        archive.judge(run_id, "q0", q0_keys[0], True)
        archive.judge(run_id, "q0", q0_keys[1], False)
        archive.judge(run_id, "q1", q1_keys[0], True)
        # All other hits stay unknown.

        report = archive.evaluate(run_id)
        q0, q1 = report["queries"]
        # Per-hit judgments.
        self.assertEqual(
            [h["judgment"] for h in q0["hits"]],
            ["relevant", "irrelevant", "unknown", "unknown"],
        )
        self.assertEqual(
            [h["judgment"] for h in q1["hits"]],
            ["relevant", "unknown", "unknown", "unknown"],
        )
        # Per-query metrics: coverage = labeled / hits,
        # labeled_relevance = relevant / labeled.
        self.assertEqual((q0["num_hits"], q0["num_labeled"], q0["num_relevant"]), (4, 2, 1))
        self.assertAlmostEqual(q0["coverage"], 2 / 4)
        self.assertAlmostEqual(q0["labeled_relevance"], 1 / 2)
        self.assertEqual((q1["num_hits"], q1["num_labeled"], q1["num_relevant"]), (4, 1, 1))
        self.assertAlmostEqual(q1["coverage"], 1 / 4)
        self.assertAlmostEqual(q1["labeled_relevance"], 1.0)
        # Run-level metrics aggregate hit counts over all queries.
        run = report["run"]
        self.assertEqual((run["num_queries"], run["num_hits"]), (2, 8))
        self.assertEqual((run["num_labeled"], run["num_relevant"]), (3, 2))
        self.assertAlmostEqual(run["coverage"], 3 / 8)
        self.assertAlmostEqual(run["labeled_relevance"], 2 / 3)
        # Unknown hits count in neither numerator of labeled_relevance.
        self.assertIsNone(archive.judgment_of(run_id, "q0", q0_keys[2]))
        self.assertTrue(archive.judgment_of(run_id, "q0", q0_keys[0]))
        self.assertFalse(archive.judgment_of(run_id, "q0", q0_keys[1]))
        # The report is JSON-serializable output.
        json.dumps(report)

    def test_correction_and_withdrawal_recompute_from_preserved_results(self):
        archive = HNSWRunArchive()
        run_id = self._record(archive, k=4)
        key = archive.get_run(run_id)["queries"][0]["hits"][0]["key"]

        archive.judge(run_id, "q0", key, True)
        q0 = lambda: archive.evaluate(run_id)["queries"][0]
        self.assertEqual((q0()["num_labeled"], q0()["num_relevant"]), (1, 1))
        # Correction: relevant -> irrelevant.
        archive.judge(run_id, "q0", key, False)
        self.assertEqual((q0()["num_labeled"], q0()["num_relevant"]), (1, 0))
        self.assertAlmostEqual(q0()["labeled_relevance"], 0.0)
        # Withdrawal: back to unknown, not irrelevant.
        archive.withdraw(run_id, "q0", key)
        self.assertEqual((q0()["num_labeled"], q0()["num_relevant"]), (0, 0))
        self.assertIsNone(q0()["labeled_relevance"])
        self.assertAlmostEqual(q0()["coverage"], 0.0)
        self.assertEqual(q0()["hits"][0]["judgment"], "unknown")
        # The full judgment trail is kept.
        self.assertEqual([e["action"] for e in archive.history(run_id)], ["judge", "judge", "withdraw"])
        # Withdrawing an unjudged hit is a no-op.
        archive.withdraw(run_id, "q0", key)
        self.assertEqual(q0()["hits"][0]["judgment"], "unknown")

        # Recomputation never re-queries the index: even with the index
        # emptied, evaluation of the preserved run is unchanged.
        archive.judge(run_id, "q0", key, True)
        eval_before_clear = archive.evaluate(run_id)
        self.index.clear()
        self.assertEqual(archive.evaluate(run_id), eval_before_clear)

    def test_empty_denominators_are_blank(self):
        archive = HNSWRunArchive()
        # A run whose queries returned no hits at all.
        run_id = self._record(archive, k=0)
        report = archive.evaluate(run_id)
        for query in report["queries"]:
            self.assertEqual(query["num_hits"], 0)
            self.assertIsNone(query["coverage"])
            self.assertIsNone(query["labeled_relevance"])
        self.assertEqual(report["run"]["num_hits"], 0)
        self.assertIsNone(report["run"]["coverage"])
        self.assertIsNone(report["run"]["labeled_relevance"])
        # Hits but no judgments: coverage is 0.0, labeled_relevance is blank.
        run_id = self._record(archive, k=3)
        report = archive.evaluate(run_id)
        for query in report["queries"]:
            self.assertEqual(query["num_labeled"], 0)
            self.assertEqual(query["coverage"], 0.0)
            self.assertIsNone(query["labeled_relevance"])
        self.assertEqual(report["run"]["coverage"], 0.0)
        self.assertIsNone(report["run"]["labeled_relevance"])

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = HNSWRunArchive(directory)
            run_id_1 = self._record(archive, k=4)
            self.index.insert(1000, np.zeros(8))
            run_id_2 = self._record(archive, k=4, metadata={"note": "after insert"})
            key = archive.get_run(run_id_1)["queries"][0]["hits"][0]["key"]
            archive.judge(run_id_1, "q0", key, True)
            archive.judge(run_id_1, "q0", key, False)  # correction
            archive.withdraw(run_id_1, "q0", key)  # withdrawal
            archive.judge(run_id_2, "q1", archive.get_run(run_id_2)["queries"][1]["hits"][0]["key"], True)

            # A fresh archive over the same directory sees everything.
            reloaded = HNSWRunArchive(directory)
            self.assertEqual(reloaded.list_runs(), archive.list_runs())
            self.assertEqual(reloaded.get_run(run_id_1), archive.get_run(run_id_1))
            self.assertEqual(reloaded.evaluate(run_id_1), archive.evaluate(run_id_1))
            self.assertEqual(reloaded.evaluate(run_id_2), archive.evaluate(run_id_2))
            self.assertEqual(reloaded.history(), archive.history())
            self.assertEqual(reloaded.evaluate(run_id_2)["metadata"], {"note": "after insert"})

            # New judgments in the reloaded archive persist as well.
            reloaded.judge(run_id_1, "q0", key, True)
            again = HNSWRunArchive(directory)
            self.assertTrue(again.judgment_of(run_id_1, "q0", key))
            self.assertEqual(again.evaluate(run_id_1)["queries"][0]["num_relevant"], 1)

    def test_get_run_returns_a_copy(self):
        archive = HNSWRunArchive()
        run_id = self._record(archive, k=3)
        # Mutating the returned record must not touch the archive.
        record = archive.get_run(run_id)
        original_key = record["queries"][0]["hits"][0]["key"]
        record["queries"][0]["hits"][0]["key"] = "tampered"
        record["queries"][0]["hits"][0]["value"][0] = -1.0
        self.assertEqual(archive.get_run(run_id)["queries"][0]["hits"][0]["key"], original_key)

    def test_queries_input_forms(self):
        archive = HNSWRunArchive()
        # A sequence of points gets positional query IDs.
        run_id = archive.record_run(self.index, [self.data[0], self.data[1]], k=2)
        self.assertEqual([q["query_id"] for q in archive.get_run(run_id)["queries"]], ["0", "1"])
        # A 2-D array is one query per row.
        run_id = archive.record_run(self.index, self.data[:3], k=2)
        self.assertEqual([q["query_id"] for q in archive.get_run(run_id)["queries"]], ["0", "1", "2"])
        # A 1-D array is a single query.
        run_id = archive.record_run(self.index, self.data[0], k=2)
        self.assertEqual(len(archive.get_run(run_id)["queries"]), 1)
        # Duplicate query IDs after normalization are rejected.
        with self.assertRaises(ValueError):
            archive.record_run(self.index, {1: self.data[0], "1": self.data[1]}, k=2)

    def test_non_scalar_hit_key_rejected(self):
        index = HNSW(distance_func=l2_distance, seed=1)
        for i, point in enumerate(self.data[:10]):
            index.insert((i, "tuple-key"), point)
        archive = HNSWRunArchive()
        with self.assertRaises(TypeError):
            archive.record_run(index, {"q0": self.data[0]}, k=2)
        self.assertEqual(archive.list_runs(), [])

    def test_evaluate_all_compares_runs(self):
        archive = HNSWRunArchive()
        run_id_1 = self._record(archive, k=3)
        run_id_2 = self._record(archive, k=5)
        key = archive.get_run(run_id_2)["queries"][0]["hits"][0]["key"]
        archive.judge(run_id_2, "q0", key, True)
        reports = archive.evaluate_all()
        self.assertEqual(list(reports), [run_id_1, run_id_2])
        self.assertEqual(reports[run_id_1]["run"]["num_hits"], 6)
        self.assertEqual(reports[run_id_2]["run"]["num_hits"], 10)
        self.assertEqual(reports[run_id_2]["run"]["num_relevant"], 1)

    def test_k_none_records_all_native_hits(self):
        archive = HNSWRunArchive()
        run_id = archive.record_run(self.index, {"q0": self.data[0]}, k=None)
        record = archive.get_run(run_id)
        expected = self.index.query(self.data[0], k=None)
        self.assertEqual(len(record["queries"][0]["hits"]), len(expected))
        self.assertEqual(
            [h["key"] for h in record["queries"][0]["hits"]],
            [key for key, _ in expected],
        )


if __name__ == "__main__":
    unittest.main()
