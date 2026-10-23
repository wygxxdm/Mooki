"""路由分支测试 —— 受控 Agent 的关键就在这几个 if 上，必须锁死。

这里不跑整张图，只测路由函数：它们决定了"什么时候追问、什么时候转人工、
什么时候继续调工具"，是安全阀所在。
"""

from __future__ import annotations

import types
import unittest

from mooki.core.config import Settings
from mooki.core.schema import Intent, IntentResult, Urgency
from mooki.graph.runtime import AgentRuntime


def make_intent(label: Intent, urgency: Urgency = Urgency.P2, conf=0.8, margin=0.5, clarify=False):
    return IntentResult(
        label=label, urgency=urgency, confidence=conf, margin=margin, need_clarify=clarify
    )


class TestRouteAfterIntent(unittest.TestCase):
    def setUp(self):
        self.route = AgentRuntime._route_after_intent

    def test_p0_goes_to_handoff(self):
        state = {"intent": make_intent(Intent.REFUND, Urgency.P0), "message": "我要退款"}
        self.assertEqual(self.route(None, state), "handoff")

    def test_explicit_human_request_is_honored(self):
        state = {"intent": make_intent(Intent.PROMO, urgency=Urgency.P3), "message": "转人工客服"}
        self.assertEqual(self.route(None, state), "handoff")

    def test_other_intent_asks_back(self):
        state = {"intent": make_intent(Intent.OTHER), "message": "在吗"}
        self.assertEqual(self.route(None, state), "clarify")

    def test_high_urgency_does_not_waste_turn_on_clarify(self):
        state = {
            "intent": make_intent(Intent.REFUND, Urgency.P1, conf=0.4, margin=0.02, clarify=True),
            "message": "钱一直没退，很急",
        }
        self.assertEqual(self.route(None, state), "plan")

    def test_normal_case_plans(self):
        state = {"intent": make_intent(Intent.LOGISTICS), "message": "物流到哪了"}
        self.assertEqual(self.route(None, state), "plan")


class TestRouteAfterObserve(unittest.TestCase):
    def setUp(self):
        self.rt = types.SimpleNamespace(settings=Settings())
        self.route = AgentRuntime._route_after_observe

    def _state(self, results, iterations, task_type="transaction", label=Intent.REFUND):
        return {
            "intent": make_intent(label),
            "task_type": task_type,
            "tool_results": results,
            "iterations": iterations,
        }

    def test_not_found_stops_looping(self):
        state = self._state([{"name": "order.query", "ok": True, "data": {"found": False}}], 1)
        self.assertEqual(self.route(self.rt, state), "respond")

    def test_transaction_forces_policy_search(self):
        results = [{"name": "order.query", "ok": True, "data": {"found": True}}]
        state = self._state(results, 1)
        self.assertEqual(self.route(self.rt, state), "plan")

    def test_iteration_cap_terminates(self):
        results = [
            {"name": "order.query", "ok": True, "data": {"found": True}},
            {"name": "refund.apply", "ok": True, "data": {"found": True}},
            {"name": "kb.search", "ok": True, "data": {"results": []}},
        ]
        state = self._state(results, 3)
        self.assertEqual(self.route(self.rt, state), "respond")

    def test_tool_failure_falls_back_to_search(self):
        results = [{"name": "order.query", "ok": False, "data": None}]
        state = self._state(results, 1)
        self.assertEqual(self.route(self.rt, state), "plan")

    def test_query_task_answers_after_one_tool(self):
        results = [{"name": "logistics.track", "ok": True, "data": {"found": True}}]
        state = self._state(results, 1, task_type="query", label=Intent.LOGISTICS)
        self.assertEqual(self.route(self.rt, state), "respond")


if __name__ == "__main__":
    unittest.main()
