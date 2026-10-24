"""人工接管：转人工不该是"建完工单就结束"。

没有这一层时，P0 投诉的处理是：建单 → 回一句"2 小时内联系你" → 结束。
用户说的那句话、系统查到的事实，全部留在这条 trace 里，接手的专员看不到，
用户还得从头再说一遍。

这一层做的事很简单：转人工节点用 LangGraph 的 interrupt 把 run 挂起，
专员处理完调 /api/handoff/resume 把答复送回同一个 thread_id，run 继续走完。

为什么非 LangGraph 不可：挂起意味着"状态得存在进程之外、下次能找回来"。
mini_graph 没有检查点，进程一重启上下文就没了，这件事它做不到。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mooki.core.config import Settings
from mooki.core.schema import ChatRequest
from mooki.graph.runtime import HAS_INTERRUPT, HAS_LANGGRAPH, AgentRuntime

P0_MESSAGE = "我要投诉，再不解决就去12315"


def build(interrupt: bool) -> AgentRuntime:
    tmp = Path(tempfile.mkdtemp(prefix="mooki-handoff-"))
    return AgentRuntime(
        settings=Settings(
            llm_offline=True,
            enable_human_interrupt=interrupt,
            ledger_file=tmp / "ledger.db",
        )
    )


@unittest.skipUnless(HAS_LANGGRAPH and HAS_INTERRUPT, "需要 langgraph + interrupt 支持")
class TestHumanInterrupt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = build(True)

    def test_p0_pauses_instead_of_finishing(self):
        resp = self.rt.chat(ChatRequest(message=P0_MESSAGE, session_id="s1", user_id="u_1001"))
        self.assertTrue(resp.waiting_human)
        self.assertEqual(resp.handoff.get("reason"), "awaiting_human")
        self.assertTrue(resp.handoff.get("ticket_id", "").startswith("TK"))
        # 挂起时不能给出"已处理"的假答复
        self.assertIn("转接人工客服", resp.reply)

    def test_resume_returns_human_reply(self):
        resp = self.rt.chat(ChatRequest(message=P0_MESSAGE, session_id="s2", user_id="u_1001"))
        resumed = self.rt.resume(resp.trace_id, "已核实，补发 30 元无门槛券，订单已加急。")
        self.assertEqual(resumed.reply, "已核实，补发 30 元无门槛券，订单已加急。")
        self.assertFalse(resumed.waiting_human)
        self.assertTrue(resumed.handoff.get("resumed"))

    def test_resume_unknown_trace_does_not_crash(self):
        with self.assertRaises(Exception):
            self.rt.resume("trace-does-not-exist", "随便回一句")

    def test_non_handoff_turn_is_not_paused(self):
        resp = self.rt.chat(ChatRequest(message="退款一般多久能到账", session_id="s3", user_id="u_1001"))
        self.assertFalse(resp.waiting_human)
        self.assertTrue(resp.reply)


class TestInterruptDisabled(unittest.TestCase):
    def test_default_off_keeps_immediate_handoff(self):
        rt = build(False)
        resp = rt.chat(ChatRequest(message=P0_MESSAGE, session_id="d1", user_id="u_1001"))
        self.assertFalse(resp.waiting_human)
        self.assertIn("已为你转接人工客服", resp.reply)

    def test_resume_refuses_when_disabled(self):
        rt = build(False)
        with self.assertRaises(RuntimeError) as ctx:
            rt.resume("any", "x")
        self.assertIn("human_interrupt_disabled", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
