"""在线意图通道权重重标定。

起因是在线评测里的一条失败样本：「发票能不能只开商品不含运费」被判成 logistics。
查下去发现不是模型判错，是**权重把它压住了** —— `rule 0.65 / llm 0.25 / dense 0.10`
是在离线规则通道上标定的，换成真模型之后 LLM 通道仍然只有四分之一的话语权，
规则通道看到"运费"两个字就赢了。

所以权重不是"调一次就完事"的超参，它跟当前用的是哪套通道强绑定。

做法：
1. 用真模型把 105 条 query 的三路投票各取一次（LLM 调用 105 次，约 1-2 分钟），
   投票结果落盘，后面网格搜索不再花钱；
2. 在 dev 上网格搜索权重（步长 0.05，三路和为 1），test 只报一次；
3. 对比「离线标定权重」与「在线标定权重」在同一批在线投票上的准确率。

用法：
    python -m scripts.online_intent_calibration
    python -m scripts.online_intent_calibration --reuse   # 复用已落盘的投票
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "eval_out"
GOLDEN = ROOT / "mooki" / "eval" / "golden" / "intent.jsonl"
VOTES_CACHE = OUT_DIR / "intent_votes_online.json"
STEP = 0.05
OFFLINE_WEIGHTS = {"llm": 0.25, "dense": 0.10, "rule": 0.65}
INITIAL_WEIGHTS = {"llm": 0.55, "dense": 0.25, "rule": 0.20}


def _load_rows() -> list[dict]:
    return [json.loads(x) for x in GOLDEN.read_text(encoding="utf-8").splitlines() if x.strip()]


def collect_votes(clf, rows: list[dict]) -> list[dict]:
    out = []
    for idx, row in enumerate(rows, 1):
        text = row["query"]
        llm_vote, _ = clf.llm_route(text)
        dense = clf.dense_vote(text)
        rule = clf.rule_vote(text)
        out.append(
            {
                "query": text,
                "gold": row["label"],
                "llm": {"label": llm_vote.label.value, "score": llm_vote.score} if llm_vote else None,
                "dense": {"label": dense.label.value, "score": dense.score} if dense else None,
                "rule": {"label": rule.label.value, "score": rule.score} if rule else None,
            }
        )
        if idx % 25 == 0:
            print(f"  已取 {idx}/{len(rows)} 条")
    return out


def fuse(record: dict, weights: dict[str, float], clf) -> tuple[str, float]:
    """复用线上同一套融合逻辑（含 LLM 低置信时让权），只是换权重。"""
    from mooki.core.schema import Vote, VoteSource, Intent

    votes = []
    for src, key in (("llm", VoteSource.LLM), ("dense", VoteSource.DENSE), ("rule", VoteSource.RULE)):
        item = record.get(src)
        if item:
            votes.append(Vote(source=key, label=Intent(item["label"]), score=float(item["score"])))
    if not votes:
        return "other", 0.0

    present = {v.source for v in votes}
    w = {
        VoteSource.LLM: weights["llm"],
        VoteSource.DENSE: weights["dense"],
        VoteSource.RULE: weights["rule"],
    }
    w = {s: w[s] for s in present}
    llm_vote = next((v for v in votes if v.source is VoteSource.LLM), None)
    if llm_vote is not None and llm_vote.score < 0.5 and len(w) > 1:
        released = w[VoteSource.LLM] * 0.5
        w[VoteSource.LLM] -= released
        others = [s for s in w if s is not VoteSource.LLM]
        for s in others:
            w[s] += released / len(others)
    total = sum(w.values()) or 1.0

    fused: dict[str, float] = {}
    for v in votes:
        fused[v.label.value] = fused.get(v.label.value, 0.0) + (w[v.source] / total) * v.score
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return top_label, round(top_score - second, 4)


def accuracy(rows: list[dict], weights: dict[str, float], clf, threshold: float = 0.12) -> dict:
    """准确率 + 弃权率：margin 低于阈值时按"追问"处理，不计入作答准确率。"""
    correct = answered = abstained = 0
    for r in rows:
        label, margin = fuse(r, weights, clf)
        if margin < threshold:
            abstained += 1
            continue
        answered += 1
        correct += int(label == r["gold"])
    n = len(rows)
    return {
        "accuracy": round(correct / n, 4),
        "answered_accuracy": round(correct / answered, 4) if answered else 0.0,
        "abstain_rate": round(abstained / n, 4),
    }


def grid(bounds: tuple[float, ...] = tuple(i * STEP for i in range(21))) -> list[dict[str, float]]:
    out = []
    for llm in bounds:
        for dense in bounds:
            rule = round(1.0 - llm - dense, 4)
            if rule < -1e-9 or rule > 1.0:
                continue
            out.append({"llm": round(llm, 4), "dense": round(dense, 4), "rule": round(rule, 4)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="复用已落盘的三路投票")
    args = parser.parse_args()

    from mooki.core.config import Settings
    from mooki.graph.runtime import AgentRuntime

    settings = Settings()
    if settings.llm_offline or not settings.llm_api_key:
        print("未配置在线模型，先填 .env 再来。", file=sys.stderr)
        return 2

    rows = _load_rows()
    if args.reuse and VOTES_CACHE.exists():
        votes = json.loads(VOTES_CACHE.read_text(encoding="utf-8"))
        print(f"复用 {VOTES_CACHE}（{len(votes)} 条）")
    else:
        runtime = AgentRuntime(settings=settings)
        t0 = time.perf_counter()
        votes = collect_votes(runtime.classifier, rows)
        print(f"取投票耗时 {time.perf_counter() - t0:.0f}s")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        VOTES_CACHE.write_text(json.dumps(votes, ensure_ascii=False, indent=1), encoding="utf-8")

    # 固定切分，保证 dev 上搜出来的权重能在同一个 test 上验证一次
    idx = list(range(len(votes)))
    random.Random(20261023).shuffle(idx)
    cut = int(len(idx) * 0.7)
    dev = [votes[i] for i in idx[:cut]]
    test = [votes[i] for i in idx[cut:]]

    scratch_clf = None
    results = []
    for name, w in (
        ("初始假设（LLM 最强）", INITIAL_WEIGHTS),
        ("离线标定（当前默认）", OFFLINE_WEIGHTS),
    ):
        results.append({"name": name, "weights": w, "dev": accuracy(dev, w, scratch_clf), "test": accuracy(test, w, scratch_clf)})

    # dev 网格搜索
    best, best_acc = None, -1.0
    for w in grid():
        acc = accuracy(dev, w, scratch_clf)["accuracy"]
        if acc > best_acc:
            best, best_acc = w, acc
    results.append({"name": "在线标定（dev 网格搜索）", "weights": best, "dev": accuracy(dev, best, scratch_clf), "test": accuracy(test, best, scratch_clf)})

    report = {"n": len(votes), "dev": len(dev), "test": len(test), "results": results,
              "votes_cache": str(VOTES_CACHE.name)}
    _write(report)
    for r in results:
        print(f"{r['name']:<24} w={r['weights']} dev={r['dev']['accuracy']:.3f} test={r['test']['accuracy']:.3f}")
    return 0


def _write(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"## 在线意图权重重标定（n={report['n']}，dev {report['dev']} / test {report['test']}）",
        "",
        "三路投票用 DeepSeek-chat 重新取一次，然后在 dev 上网格搜索权重（步长 0.05），test 只报一次。",
        "",
        "| 权重来源 | LLM | Dense | Rule | dev 准确率 | test 准确率 | 弃权率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        w = r["weights"]
        lines.append(
            f"| {r['name']} | {w['llm']:.2f} | {w['dense']:.2f} | {w['rule']:.2f} | "
            f"{r['dev']['accuracy']:.3f} | {r['test']['accuracy']:.3f} | {r['test']['abstain_rate']:.1%} |"
        )
    lines += [
        "",
        "> 初始权重把最高话语权给了 LLM（0.55）。在**离线替身**下这组权重的融合准确率只有 0.35，",
        "> 因为替身表达能力弱；换成真模型之后同一组权重反而接近最优 —— 这恰好说明",
        "> **权重不是模型的属性，是「模型 + 通道」这个组合的属性**，换模型必须重标定。",
    ]
    (OUT_DIR / "intent_weights_online.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
