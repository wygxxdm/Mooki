"""LLM 接入层。

设计要点：
1. 传输层（ChatClient）与业务层（LLMService）分离 —— 换模型不动业务代码。
2. offline 模式下用 HeuristicChatClient，保证整套链路可在无 API Key 环境下跑通、
   且评测结果可复现（这是离线评测能进 CI 的前提）。
3. 所有结构化输出统一走 _extract_json，解析失败降级为 None，由调用方决定兜底。
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

import httpx

from mooki.core.config import Settings
from mooki.core.text import estimate_tokens

TASK_MARK = "[[task:{name}]]"
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)

ANSWER_SYSTEM = (
    "你是电商平台售后客服。只依据参考资料回答，不允许编造政策、时效、金额。"
    "资料不足时明确说明并建议转人工。回答控制在 4 句以内，先给结论。"
    "如果用户给了订单号且系统已查到单据数据，必须结合单据数据回答，不要只讲政策。"
)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|```$", "", text, flags=re.S).strip()
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class ChatClient(Protocol):
    name: str

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str: ...


class OpenAIChatClient:
    """兼容 OpenAI /v1/chat/completions 协议的任意模型服务（通义、DeepSeek、vLLM…）。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.name = settings.llm_model
        self._client = httpx.Client(timeout=settings.llm_timeout_s)

    def _endpoint(self) -> str:
        base = self.settings.llm_base_url.rstrip("/")
        return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        resp = self._client.post(self._endpoint(), json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class HeuristicChatClient:
    """离线确定性实现：按 task 标记走不同启发式。

    存在的意义不是"假装有 LLM"，而是让意图投票、检索消融、端到端评测这些
    实验在没有 Key、没有网络抖动的情况下可重复 —— 否则每次跑出来的数字都不一样，
    消融实验就没有意义了。
    """

    name = "heuristic-offline"

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        system = messages[0]["content"] if messages else ""
        prompt = messages[-1]["content"] if messages else ""
        task = "generic"
        m = re.search(r"\[\[task:(\w+)\]\]", system)
        if m:
            task = m.group(1)
        handler = getattr(self, f"_t_{task}", None)
        return handler(prompt) if handler else self._generic(prompt)

    # ---------- 各任务 ----------

    @staticmethod
    def _generic(prompt: str) -> str:
        return "好的，我来帮你处理这个问题。"

    @staticmethod
    def _t_intent(prompt: str) -> str:
        # 交给朴素贝叶斯，而不是再做一遍关键词匹配 —— 与规则路同源会让投票失去意义
        from mooki.intent.nb import predict_intent  # 延迟导入避免循环依赖

        return json.dumps(predict_intent(prompt))

    _REWRITE_SYNONYMS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (("退钱", "退回来", "钱没退", "钱还没"), ("退款", "到账")),
        (("到哪", "还没到", "到货"), ("物流", "派送")),
        (("重开", "抬头", "税号", "开错"), ("红冲", "换开")),
        (("降价", "便宜了", "差价"), ("价格保护",)),
        (("进水", "摔了", "碎屏", "拆机"), ("人为损坏",)),
        (("券", "满减", "红包"), ("优惠券", "使用门槛")),
        (("买小了", "尺码", "大一码", "小一号"), ("换货",)),
        (("多久", "几天", "时效"), ("时效",)),
    )

    @classmethod
    def _t_rewrite(cls, prompt: str) -> str:
        text = prompt.split("原始问题：", 1)[-1].strip()
        for filler in ("你好", "麻烦问一下", "请问一下", "我想知道", "帮我看看", "嗯"):
            text = text.replace(filler, "")
        text = text.strip("，。！？ ")
        extra: list[str] = []
        for sources, targets in cls._REWRITE_SYNONYMS:
            if any(s in text for s in sources):
                extra.extend(t for t in targets if t not in text and t not in extra)
        return " ".join([text, *extra]).strip()

    _PREFERRED_TOOL: dict[str, tuple[str, ...]] = {
        "order_status": ("order.query", "kb.search"),
        "refund": ("order.query", "refund.apply", "kb.search"),
        "return_exchange": ("order.query", "return.create", "kb.search"),
        "logistics": ("logistics.track", "order.query", "kb.search"),
        "complaint": ("ticket.create", "order.query", "kb.search"),
        "product_consult": ("kb.search",),
        "invoice": ("order.query", "invoice.issue", "kb.search"),
        "promo": ("promo.check", "kb.search"),
        "account": ("kb.search", "ticket.create"),
        "other": ("kb.search", "ticket.create"),
    }

    @classmethod
    def _t_plan(cls, prompt: str) -> str:
        allowed = re.findall(r'"([a-z_]+\.[a-z_]+)"', prompt)
        order_ids = re.findall(r"SO\d{9,}", prompt)
        question = ""
        m = re.search(r"用户问题：(.+)", prompt)
        if m:
            question = m.group(1).strip()[:120]
        intent = ""
        m = re.search(r"用户意图：(\w+)", prompt)
        if m:
            intent = m.group(1)
        if not allowed:
            return json.dumps({"tool": "", "args": {}})

        preferred = [t for t in cls._PREFERRED_TOOL.get(intent, ()) if t in allowed]
        name = preferred[0] if preferred else allowed[0]
        if not order_ids and name in ("order.query", "refund.apply", "return.create", "logistics.track"):
            name = "kb.search" if "kb.search" in allowed else name

        args: dict[str, Any] = {}
        if name == "kb.search":
            args = {"query": question}
        elif name == "ticket.create":
            args = {"reason": "customer_request", "summary": question}
        elif order_ids:
            args = {"order_id": order_ids[0]}
            if name in ("refund.apply", "return.create"):
                args["reason"] = "no_reason"
        return json.dumps({"tool": name, "args": args})

    @staticmethod
    def _t_rerank(prompt: str) -> str:
        query = prompt.split("查询：", 1)[-1].split("\n", 1)[0].strip()
        cands = re.findall(r"^\[(\d+)\]\s*(.*)$", prompt, re.M)
        q_bigrams = {query[i : i + 2] for i in range(max(0, len(query) - 1))}
        scored = []
        for idx, text in cands:
            d_bigrams = {text[i : i + 2] for i in range(max(0, len(text) - 1))}
            scored.append((len(q_bigrams & d_bigrams), int(idx)))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return json.dumps({"order": [i for _, i in scored]})

    @staticmethod
    def _t_answer(prompt: str) -> str:
        head, _, tail = prompt.partition("用户问题：")
        question = tail.strip().split("\n")[0] if tail else ""

        def blocks_after(marker: str) -> list[str]:
            if marker not in head:
                return []
            body = head.split(marker, 1)[1]
            return [b.strip() for b in body.split("\n\n") if b.strip()]

        facts = ""
        passages: list[str] = []
        for block in blocks_after("参考资料："):
            if block.startswith("单据数据："):
                facts = block[len("单据数据：") :]
            elif not block.startswith("最近对话") and not block.startswith("历史摘要"):
                passages.append(block)

        def first_sentences(text: str, n: int) -> str:
            parts = [p for p in re.split(r"(?<=[。！？；])", text) if p.strip()]
            return "".join(parts[:n]).strip()

        out: list[str] = []
        if facts:
            out.append("已为你核实：" + first_sentences(facts, 2).rstrip("；。") + "。")
        if passages:
            body = re.sub(r"^[^：]{0,40}：", "", passages[0], count=1)
            out.append("按平台规则，" + first_sentences(body, 2))
        if not out:
            out.append(f"关于「{question[:24]}」，我已记录，会尽快核实后回复你。")
        return "".join(out)[:400]

    @staticmethod
    def _t_judge(prompt: str) -> str:
        """离线判分：用参考答案的字符 bigram 覆盖率近似语义覆盖度。

        不用"整串命中"，是因为参考答案是自然语言长句，字面包含几乎永远不成立，
        会把所有正确答案都打成 1 分 —— 那是判分器坏了，不是答案坏了。
        """
        ref = prompt.split("参考答案：", 1)[-1] if "参考答案：" in prompt else prompt
        ans = prompt.split("待评答案：", 1)[-1].split("参考答案：", 1)[0] if "待评答案：" in prompt else prompt

        def bigrams(text: str) -> set[str]:
            clean = re.sub(r"[\s，。；：、！？,.;:!?]", "", text)
            return {clean[i : i + 2] for i in range(max(0, len(clean) - 1))}

        ref_grams = bigrams(ref)
        if not ref_grams:
            return json.dumps({"score": 3, "reason": "参考答案为空"})
        coverage = len(ref_grams & bigrams(ans)) / len(ref_grams)
        if coverage >= 0.55:
            score = 5
        elif coverage >= 0.40:
            score = 4
        elif coverage >= 0.25:
            score = 3
        elif coverage >= 0.12:
            score = 2
        else:
            score = 1
        return json.dumps({"score": score, "reason": f"参考要点覆盖率 {coverage:.2f}"})

    @staticmethod
    def _t_summary(prompt: str) -> str:
        turns = prompt.split("对话：", 1)[-1].strip().splitlines()
        keep = [t for t in turns if re.search(r"SO\d+|退款|退货|物流|发票", t)][:4]
        return "；".join(t[-60:] for t in keep) or "用户咨询售后问题，尚未给出明确诉求。"


class LLMService:
    def __init__(self, client: ChatClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _run(self, task: str, system: str, prompt: str, temperature: float = 0.0) -> str:
        messages = [
            {"role": "system", "content": TASK_MARK.format(name=task) + "\n" + system},
            {"role": "user", "content": prompt},
        ]
        self.calls += 1
        self.prompt_tokens += sum(estimate_tokens(m["content"]) for m in messages)
        out = self.client.chat(messages, temperature=temperature)
        self.completion_tokens += estimate_tokens(out)
        return out

    # ---------- 业务方法 ----------

    def classify_intent(self, text: str) -> dict[str, Any] | None:
        system = (
            "你是电商售后意图分类器。从 order_status/refund/return_exchange/logistics/"
            "complaint/product_consult/invoice/promo/account/other 中选一个，"
            "并给出 urgency（P0-P3：P0 资损或强投诉、P1 明确催促、P2 一般、P3 仅咨询）。"
            '只输出 JSON：{"label":"...","confidence":0.0-1.0,"urgency":"P2"}'
        )
        return _extract_json(self._run("intent", system, text))

    def rewrite_query(self, text: str, history: list[str] | None = None) -> str:
        system = (
            "把用户的口语化售后问题改写成一个便于检索的查询：补全指代、去掉寒暄、"
            "保留订单号等实体，不要回答。只输出改写后的一句话。"
        )
        prompt = f"历史：{history[-2:] if history else []}\n原始问题：{text}"
        out = self._run("rewrite", system, prompt)
        return out.strip().splitlines()[0][:200] if out.strip() else text

    def plan_tool(self, text: str, allowed: list[str], intent_hint: str = "") -> dict[str, Any] | None:
        system = (
            "你是售后 Agent 的工具选择器。只能从给定工具中选一个，不要编造工具名。"
            '只输出 JSON：{"tool":"<name>","args":{...}}，无需调用工具时 tool 为空字符串。'
        )
        prompt = (
            f"用户意图：{intent_hint or 'unknown'}\n"
            f"可选工具：{json.dumps(allowed, ensure_ascii=False)}\n"
            f"用户问题：{text}\n"
            "工具参数中如需订单号，从问题里原样提取（形如 SO 开头加数字）。"
        )
        return _extract_json(self._run("plan", system, prompt))

    def rerank(self, query: str, docs: list[str]) -> list[int] | None:
        if len(docs) <= 1:
            return list(range(len(docs)))
        system = (
            "按与查询的相关性对候选段落排序，最相关的在前。"
            '只输出 JSON：{"order":[0,2,1]}，元素是候选的 0 基下标，必须包含全部下标。'
        )
        prompt = f"查询：{query}\n候选：\n" + "\n".join(
            f"[{i}] {d[:180]}" for i, d in enumerate(docs)
        )
        out = _extract_json(self._run("rerank", system, prompt))
        if not out or not isinstance(out.get("order"), list):
            return None
        order: list[int] = []
        for item in out["order"]:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(docs) and idx not in order:
                order.append(idx)
        # 补齐模型漏掉的下标，保证输出是一个完整排列
        order.extend(i for i in range(len(docs)) if i not in order)
        return order

    def answer(self, question: str, contexts: list[str], profile: dict[str, Any]) -> str:
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)) or "（无）"
        prompt = f"用户画像：{json.dumps(profile, ensure_ascii=False)}\n参考资料：\n{ctx}\n\n用户问题：{question}"
        return self._run("answer", ANSWER_SYSTEM, prompt, temperature=self.settings.llm_temperature).strip()

    def answer_conversation(self, prompt: str) -> str:
        """由上层（记忆拼装器）给定完整提示：含参考资料、历史、画像与当前问题。"""
        return self._run("answer", ANSWER_SYSTEM, prompt, temperature=self.settings.llm_temperature).strip()

    def judge(self, question: str, answer: str, reference: str) -> dict[str, Any] | None:
        system = (
            "你是客服质检员。按参考答案给待评答案打 1-5 分："
            "5 完全正确且完整，3 基本正确有遗漏，1 答错或答非所问。"
            '只输出 JSON：{"score":1-5,"reason":"..."}'
        )
        prompt = f"用户问题：{question}\n待评答案：{answer}\n参考答案：{reference}"
        return _extract_json(self._run("judge", system, prompt))

    def summarize(self, turns: list[str]) -> str:
        system = "把售后对话压缩成不超过 80 字的摘要，必须保留订单号、金额、诉求。只输出摘要。"
        return self._run("summary", system, "对话：\n" + "\n".join(turns)).strip()[:200]


def build_llm(settings: Settings) -> LLMService:
    client: ChatClient = (
        HeuristicChatClient() if settings.llm_offline else OpenAIChatClient(settings)
    )
    return LLMService(client, settings)
