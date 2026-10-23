"""评测入口。

用法：
    python -m scripts.run_eval retrieval    # 检索消融
    python -m scripts.run_eval intent       # 意图识别
    python -m scripts.run_eval dialogue     # 端到端对话
    python -m scripts.run_eval all
    python -m scripts.run_eval all --online # 用真实模型（结果不可复现，仅看观感）

评测默认强制离线：换成真实 LLM 后每次跑出来的数字都不一样，消融和对比就失去意义。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from mooki.core.config import Settings
from mooki.eval.retrieval_eval import run_ablation
from mooki.graph.runtime import AgentRuntime


def isolated_settings() -> Settings:
    """评测用配置：强制离线 + 独立账本。

    账本必须隔离：`refund.apply` 会改状态，不隔离的话第二次跑评测就命中
    "已在退款中"，数字每次都变 —— 可复现性直接没了。
    """
    tmp = Path(tempfile.mkdtemp(prefix="mooki-eval-"))
    return Settings(llm_offline=True, ledger_file=tmp / "ledger.jsonl")


def build_eval_runtime(online: bool) -> AgentRuntime:
    if online:
        print("⚠️  在线模式：LLM 与温度都会引入随机性，本报告不可作为对比依据\n")
        return AgentRuntime(settings=Settings(ledger_file=Path(tempfile.mkdtemp()) / "ledger.jsonl"))
    return AgentRuntime(settings=isolated_settings())


def main(argv: list[str]) -> int:
    target = argv[1] if len(argv) > 1 else "all"
    runtime = build_eval_runtime("--online" in argv)

    if target in ("retrieval", "all"):
        from mooki.eval.retrieval_eval import render_markdown, run_ablation
        from mooki.intent.labels import domain_for

        def resolver(query: str) -> str | None:
            # 域先验必须来自线上同一套意图识别，而不是评测集的标注字段
            return domain_for(runtime.classifier.classify(query).label)

        report = run_ablation(runtime.retriever, runtime.settings, domain_resolver=resolver)
        print(render_markdown(report))

    if target in ("intent", "all"):
        from mooki.eval.intent_eval import render_markdown as render_intent
        from mooki.eval.intent_eval import run_intent_eval

        report = run_intent_eval(runtime.classifier, runtime=runtime)
        print(render_intent(report))

    if target in ("dialogue", "all"):
        from mooki.eval.dialogue_eval import render_markdown as render_dialogue
        from mooki.eval.dialogue_eval import run_dialogue_eval

        report = run_dialogue_eval(runtime)
        print(render_dialogue(report))

    if target in ("scale", "all"):
        from mooki.eval.scale_eval import render_markdown as render_scale
        from mooki.eval.scale_eval import run_scale_eval

        report = run_scale_eval(runtime)
        print(render_scale(report))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
