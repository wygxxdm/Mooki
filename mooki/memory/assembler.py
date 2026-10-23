"""上下文拼装：把预算切成几份，按优先级抢占，并输出可审计的分配报告。

为什么不能"把能塞的都塞进去"：
1. 超预算直接报错或被截断，报错位置不可控；
2. 塞满之后模型注意力被稀释，答案质量反而下降（lost in the middle）；
3. 没有分配报告，就无法解释"为什么这次答得差"—— 排查时只能靠猜。

这里的策略：
- 先按固定比例给每个 section 配额；
- 用不完的配额回流，优先补给检索（证据最重要），其次历史；
- 检索超配额按"整块丢弃"而不是截断文本，避免把一条政策切成半句；
- 每段都记录 tokens / 是否截断 / 丢了多少，落进 trace。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mooki.core.text import estimate_tokens, truncate_tokens

# (section, 比例, 丢弃优先级：数字越小越先被砍)
ALLOCATION: tuple[tuple[str, float, int], ...] = (
    ("system", 0.12, 9),     # 系统提示，最后才动
    ("profile", 0.05, 9),    # 画像很短，基本不会被砍
    ("summary", 0.10, 3),
    ("history", 0.33, 2),
    ("retrieval", 0.35, 1),  # 证据最重要，但不是不能压缩
    ("query", 0.05, 0),
)


@dataclass
class SectionReport:
    tokens: int
    quota: int
    truncated: bool = False
    dropped_items: int = 0


@dataclass
class AssemblyReport:
    budget: int
    used: int
    sections: dict[str, SectionReport] = field(default_factory=dict)
    surplus_reused: int = 0

    @property
    def utilization(self) -> float:
        return round(self.used / self.budget, 3) if self.budget else 0.0

    def as_dict(self) -> dict:
        return {
            "budget": self.budget,
            "used": self.used,
            "utilization": self.utilization,
            "surplus_reused": self.surplus_reused,
            "sections": {
                k: {
                    "tokens": v.tokens,
                    "quota": v.quota,
                    "truncated": v.truncated,
                    "dropped_items": v.dropped_items,
                }
                for k, v in self.sections.items()
            },
        }


class ContextAssembler:
    def __init__(self, budget: int = 2400):
        self.budget = budget

    def _cut_history(self, history: list[str], quota: int) -> tuple[list[str], int]:
        kept: list[str] = []
        used = 0
        for line in reversed(history):  # 从最新往回留
            cost = estimate_tokens(line)
            if used + cost > quota:
                break
            kept.append(line)
            used += cost
        kept.reverse()
        return kept, len(history) - len(kept)

    def _cut_retrieval(self, passages: list[str], quota: int) -> tuple[list[str], int]:
        kept: list[str] = []
        used = 0
        for passage in passages:  # 从高相关往下留
            cost = estimate_tokens(passage)
            if used + cost > quota:
                if not kept:  # 第一条就超，硬截断保底
                    kept.append(truncate_tokens(passage, quota))
                    used = quota
                break
            kept.append(passage)
            used += cost
        return kept, len(passages) - len(kept)

    def assemble(
        self,
        *,
        system: str,
        profile: dict,
        summary: str,
        history: list[str],
        retrieval: list[str],
        query: str,
        budget: int | None = None,
    ) -> tuple[list[dict], AssemblyReport, dict]:
        budget = budget or self.budget
        quota = {name: int(budget * ratio) for name, ratio, _ in ALLOCATION}

        profile_text = "用户画像：" + "，".join(f"{k}={v}" for k, v in profile.items() if v)
        raw = {
            "system": [system],
            "profile": [profile_text if profile else ""],
            "summary": [f"历史摘要：{summary}" if summary else ""],
            "history": history,
            "retrieval": retrieval,
            "query": [f"用户当前问题：{query}"],
        }

        report = AssemblyReport(budget=budget, used=0)
        result: dict[str, list[str]] = {}
        surplus = 0

        for name, _, _ in ALLOCATION:
            items = [x for x in raw[name] if x and x.strip()]
            natural = sum(estimate_tokens(x) for x in items)
            cap = quota[name]

            if natural <= cap:
                result[name] = items
                report.sections[name] = SectionReport(tokens=natural, quota=cap)
                surplus += cap - natural
                continue

            # 超配额：按 section 特性裁剪
            if name == "history":
                kept, dropped = self._cut_history(items, cap)
                used = sum(estimate_tokens(x) for x in kept)
                result[name] = kept
                report.sections[name] = SectionReport(used, cap, truncated=True, dropped_items=dropped)
            elif name == "retrieval":
                kept, dropped = self._cut_retrieval(items, cap)
                used = sum(estimate_tokens(x) for x in kept)
                result[name] = kept
                report.sections[name] = SectionReport(used, cap, truncated=True, dropped_items=dropped)
            else:
                joined = "\n".join(items)
                cut = truncate_tokens(joined, cap)
                result[name] = [cut]
                report.sections[name] = SectionReport(estimate_tokens(cut), cap, truncated=True)

        # 剩余预算回补：给被截断的 section 加额度（检索优先）
        if surplus > 0:
            for name in ("retrieval", "history", "summary"):
                sec = report.sections.get(name)
                if sec is None or not sec.truncated or surplus <= 0:
                    continue
                extra = min(surplus, quota[name])
                if name == "retrieval":
                    kept, dropped = self._cut_retrieval([x for x in raw[name] if x.strip()], quota[name] + extra)
                    result[name] = kept
                    sec.dropped_items = dropped
                elif name == "history":
                    kept, dropped = self._cut_history(raw[name], quota[name] + extra)
                    result[name] = kept
                    sec.dropped_items = dropped
                new_tokens = sum(estimate_tokens(x) for x in result[name])
                surplus -= new_tokens - sec.tokens
                sec.tokens = new_tokens
                sec.quota = quota[name] + extra
                if surplus <= 0:
                    break
            report.surplus_reused = max(0, int(report.surplus_reused) + 0)
            report.surplus_reused = max(0, surplus)

        # 检索片段之间用空行分隔：下游按块取证据时不会把两条政策粘成一条
        blocks = {
            k: ("\n\n" if k == "retrieval" else "\n").join(v) for k, v in result.items() if v
        }
        report.used = sum(estimate_tokens(v) for v in blocks.values())

        messages = [{"role": "system", "content": system}]
        if blocks.get("profile"):
            messages.append({"role": "system", "content": blocks["profile"]})
        if blocks.get("summary"):
            messages.append({"role": "system", "content": blocks["summary"]})
        if blocks.get("retrieval"):
            messages.append({"role": "system", "content": "参考资料：\n" + blocks["retrieval"]})
        for line in result.get("history", []):
            role = "assistant" if line.startswith("assistant") else "user"
            messages.append({"role": role, "content": line.split(":", 1)[-1].strip()})
        messages.append({"role": "user", "content": query})

        return messages, report, blocks
