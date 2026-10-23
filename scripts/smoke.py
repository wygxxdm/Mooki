"""冒烟测试：跑通 5 条典型链路，打印回复与关键指标。

用法：python -m scripts.smoke
"""

from __future__ import annotations

import sys

from mooki.core.schema import ChatRequest
from mooki.graph.runtime import GRAPH_ENGINE, AgentRuntime

CASES = [
    ("查订单", "SO20260512001 这个单子到哪了"),
    ("退款判断", "SO20261018006 我不想要了，能退款吗"),
    ("超期无理由", "SO20260518002 我想无理由退货"),
    ("高紧急投诉", "我要投诉，再不解决就去12315"),
    ("模糊追问", "在吗"),
    ("政策问答", "退款一般多久能到账"),
]


def main() -> int:
    runtime = AgentRuntime()
    print(f"engine={GRAPH_ENGINE} chunks={runtime.index.size} domains={runtime.index.domains()}")
    print("=" * 72)
    for idx, (title, message) in enumerate(CASES):
        response = runtime.chat(ChatRequest(message=message, session_id=f"smoke-{idx}"))
        intent = response.intent
        print(f"[{title}] {message}")
        print(f"  intent : {intent.label.value if intent else '-'} / {intent.urgency.value if intent else '-'}"
              f" conf={intent.confidence if intent else 0:.2f} margin={intent.margin if intent else 0:.2f}"
              f" clarify={intent.need_clarify if intent else '-'}")
        print(f"  tools  : {response.tools_used}")
        print(f"  cite   : {[c['chunk_id'] for c in response.citations][:3]}")
        print(f"  degrad : {response.degraded} | latency={response.usage.get('latency_ms')}ms"
              f" | assembly={response.usage.get('assembly', {}).get('utilization')}")
        print(f"  reply  : {response.reply}")
        print("-" * 72)
    stats = runtime.stats()
    print("intent_cache", stats["intent_cache"])
    print("retrieval_cache", stats["retrieval_cache"])
    print("breaker", stats["breaker"]["dense"]["state"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
