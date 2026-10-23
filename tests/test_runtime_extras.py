"""后台写盘与离线意图替身的测试。"""

from __future__ import annotations

import threading
import time
import unittest

from mooki.core.async_writer import BackgroundWriter
from mooki.core.schema import Intent
from mooki.intent import nb


class TestBackgroundWriter(unittest.TestCase):
    def test_submit_executes_after_flush(self):
        writer = BackgroundWriter("t")
        seen: list[int] = []
        for i in range(20):
            writer.submit(seen.append, i)
        writer.flush()
        self.assertEqual(len(seen), 20)
        self.assertEqual(writer.stats()["executed"], 20)
        writer.close()

    def test_failure_does_not_kill_thread(self):
        writer = BackgroundWriter("t2")
        seen: list[str] = []

        def boom():
            raise RuntimeError("disk full")

        writer.submit(boom)
        writer.submit(seen.append, "after-failure")
        writer.flush()
        self.assertEqual(seen, ["after-failure"])
        self.assertGreaterEqual(writer.stats()["failed"], 1)
        writer.close()

    def test_flush_waits_for_pending(self):
        writer = BackgroundWriter("t3")
        done = threading.Event()

        def slow():
            time.sleep(0.05)
            done.set()

        writer.submit(slow)
        writer.flush(timeout=2.0)
        self.assertTrue(done.is_set())
        writer.close()


class TestNaiveBayes(unittest.TestCase):
    def test_predict_returns_valid_label(self):
        model = nb.build_model()
        label, prob = model.predict("我的退款三天了还没到账")
        self.assertIn(label, {i.value for i in Intent})
        self.assertGreaterEqual(prob, 0.3)
        self.assertLessEqual(prob, 0.9)

    def test_holdout_model_excludes_samples(self):
        exclude = {"我要退款，东西不想要了"}
        full = nb.build_model()
        held = nb.build_model(exclude)
        self.assertLess(held.n_docs, full.n_docs)

    def test_predict_intent_shape_matches_llm_output(self):
        out = nb.predict_intent("我要投诉你们")
        self.assertEqual(set(out), {"label", "confidence", "urgency"})
        self.assertIn(out["urgency"], {"P0", "P1", "P2", "P3"})


if __name__ == "__main__":
    unittest.main()
