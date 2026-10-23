"""意图识别评测：混淆矩阵 + 逐类 P/R/F1 + 弃权率 + 通道消融。

单独报"弃权率"（need_clarify）很重要：把"我不确定，追问一句"计为错，
会低估系统的真实可用性；计为对又会自欺。所以两个数都报，让读者自己判断。

通道消融回答的是另一个问题：**三路投票到底有没有用，还是某一路独大？**
如果 rule_only 和 fusion 差不多，那后两路就是白写的 —— 这种结论必须先跑出数来再说。
"""

from __future__ import annotations

import json
from pathlib import Path

from mooki.core.schema import Intent, VoteSource

GOLDEN = Path(__file__).resolve().parent / "golden" / "intent.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "eval_out"

ROUTE_SETS: dict[str, set[VoteSource] | None] = {
    "rule_only": {VoteSource.RULE},
    "dense_only": {VoteSource.DENSE},
    "llm_only": {VoteSource.LLM},
    "rule+dense": {VoteSource.RULE, VoteSource.DENSE},
    "fusion(三路)": None,
}


def load_golden() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _evaluate(classifier, rows: list[dict], routes: set[VoteSource] | None) -> dict:
    labels = [i.value for i in Intent]
    confusion = {t: {p: 0 for p in labels} for t in labels}
    correct = abstained = answered = answered_correct = 0
    details = []

    for row in rows:
        result = classifier.classify(row["query"], routes=routes)
        gold, pred = row["label"], result.label.value
        confusion[gold][pred] += 1
        hit = gold == pred
        correct += int(hit)
        abstained += int(result.need_clarify)
        if not result.need_clarify:
            answered += 1
            answered_correct += int(hit)
        details.append(
            {
                "query": row["query"],
                "gold": gold,
                "pred": pred,
                "confidence": result.confidence,
                "margin": result.margin,
                "need_clarify": result.need_clarify,
                "votes": [f"{v.source.value}:{v.label.value}:{v.score:.2f}" for v in result.votes],
            }
        )

    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        pred_total = sum(confusion[t][label] for t in labels)
        gold_total = sum(confusion[label].values())
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / gold_total if gold_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "support": gold_total,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    supported = [v for v in per_class.values() if v["support"] > 0]
    n = len(rows) or 1
    return {
        "n": len(rows),
        "accuracy": round(correct / n, 4),
        "macro_f1": round(sum(v["f1"] for v in supported) / len(supported), 4) if supported else 0.0,
        "abstain_rate": round(abstained / n, 4),
        "accuracy_on_answered": round(answered_correct / max(answered, 1), 4),
        "answered": answered,
        "per_class": per_class,
        "confusion": confusion,
        "errors": [d for d in details if d["gold"] != d["pred"]],
        "details": details,
    }


def run_intent_eval(classifier, save: bool = True, runtime=None) -> dict:
    classifier.cache.clear()  # 关掉缓存，否则测的是缓存
    rows = load_golden()
    report = _evaluate(classifier, rows, routes=None)
    report["route_ablation"] = run_route_ablation(classifier, rows, save=False)
    if runtime is not None:
        report["holdout"] = run_holdout_eval(runtime, save=False)

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "intent_eval.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT_DIR / "intent_eval.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _fuse_with_weights(votes: list[tuple[str, float]], weights: dict[str, float]) -> tuple[str, float, float]:
    """在离线记录好的单路票上重放融合，用于权重搜索（不重新推理，保证可比）。"""
    total_w = sum(weights[r] for r, _ in votes) or 1.0
    fused: dict[str, float] = {}
    for route, (label, score) in votes:
        fused[label] = fused.get(label, 0.0) + weights[route] / total_w * score
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return top, round(top_score, 4), round(top_score - second, 4)


def run_holdout_eval(runtime, folds: int = 5, save: bool = True) -> dict:
    """exemplar 留出实验 + 权重标定 —— 通道消融的可信版本。

    协议（这是关键，比数字本身重要）：
      1. 5 折交叉，每折用其余样本建底库（向量与 NB 都重建），拿这一折当查询；
      2. 前 3 折作 dev（标定权重），后 2 折作 test（只报一次，不参与任何选择）；
      3. 单路准确率三处都报，权重在 dev 上网格搜索，test 上验证。

    必须说清的局限：每折留出后每类只剩 4 条训练样本，属于极端小样本。
    这里测出来的差距主要反映"小样本下哪条路更稳"，不能外推成"向量路无用"。
    """
    from mooki.core.cache import TTLCache
    from mooki.intent import nb
    from mooki.intent.classifier import IntentClassifier
    from mooki.intent.labels import INTENT_SPECS

    items = [(ex, intent) for intent, spec in INTENT_SPECS.items() for ex in spec.exemplars]
    items.sort(key=lambda kv: kv[0])  # 确定性排序，保证可复现
    fold_size = max(1, len(items) // folds)
    dev_folds = max(1, folds - 2)

    records: list[dict] = []
    for fold in range(folds):
        start = fold * fold_size
        end = len(items) if fold == folds - 1 else (fold + 1) * fold_size
        held = items[start:end]
        if not held:
            continue
        exclude = {text for text, _ in held}
        nb.install_model(nb.build_model(exclude))
        classifier = IntentClassifier(
            runtime.llm, runtime.embedder, TTLCache(8, 1), exemplar_exclude=exclude
        )
        for text, gold in held:
            votes: list[tuple[str, tuple[str, float]]] = []
            for route in ("llm", "dense", "rule"):
                single = {
                    "llm": {VoteSource.LLM},
                    "dense": {VoteSource.DENSE},
                    "rule": {VoteSource.RULE},
                }[route]
                result = classifier.classify(text, routes=single)
                if result.votes:
                    vote = result.votes[0]
                    votes.append((route, (vote.label.value, vote.score)))
            records.append({"gold": gold.value, "votes": votes, "split": "dev" if fold < dev_folds else "test"})
    nb.install_model(None)  # 还原全局替身，避免污染后续评测

    def accuracy_of(route: str, split: str | None = None) -> float:
        picked = [r for r in records if split is None or r["split"] == split]
        if not picked:
            return 0.0
        hits = 0
        for record in picked:
            vote = next((v for r, v in record["votes"] if r == route), None)
            if vote and vote[0] == record["gold"]:
                hits += 1
        return round(hits / len(picked), 4)

    def fusion_accuracy(weights: dict[str, float], split: str | None = None) -> float:
        picked = [r for r in records if split is None or r["split"] == split]
        if not picked:
            return 0.0
        hits = sum(
            1
            for record in picked
            if _fuse_with_weights(record["votes"], weights)[0] == record["gold"]
        )
        return round(hits / len(picked), 4)

    default_weights = {"llm": 0.55, "dense": 0.25, "rule": 0.20}  # 初始假设：LLM 最强
    grid: list[tuple[float, float, float]] = []
    for w_rule in (0.2, 0.35, 0.5, 0.65, 0.8):
        for w_dense in (0.1, 0.2, 0.3, 0.4):
            w_llm = round(1.0 - w_rule - w_dense, 4)
            if w_llm < 0.05:
                continue
            grid.append((w_llm, w_dense, w_rule))

    best = max(grid, key=lambda g: fusion_accuracy({"llm": g[0], "dense": g[1], "rule": g[2]}, "dev"))
    tuned = {"llm": best[0], "dense": best[1], "rule": best[2]}

    out = {
        "n": len(records),
        "dev_n": sum(1 for r in records if r["split"] == "dev"),
        "test_n": sum(1 for r in records if r["split"] == "test"),
        "routes": {
            route: {"dev": accuracy_of(route, "dev"), "test": accuracy_of(route, "test")}
            for route in ("llm", "dense", "rule")
        },
        "fusion_default": {
            "weights": default_weights,
            "dev": fusion_accuracy(default_weights, "dev"),
            "test": fusion_accuracy(default_weights, "test"),
        },
        "fusion_tuned": {
            "weights": tuned,
            "dev": fusion_accuracy(tuned, "dev"),
            "test": fusion_accuracy(tuned, "test"),
        },
    }
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "intent_holdout.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out


def run_route_ablation(classifier, rows: list[dict] | None = None, save: bool = True) -> dict:
    rows = rows or load_golden()
    out = {}
    hits: dict[str, dict[str, bool]] = {}
    for name, routes in ROUTE_SETS.items():
        metrics = _evaluate(classifier, rows, routes=routes)
        out[name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "abstain_rate": metrics["abstain_rate"],
            "accuracy_on_answered": metrics["accuracy_on_answered"],
        }
        hits[name] = {d["query"]: d["gold"] == d["pred"] for d in metrics["details"]}

    out["_complementarity"] = complementarity(hits)
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "intent_route_ablation.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out


def complementarity(hits: dict[str, dict[str, bool]]) -> dict:
    """两两通道的互补性矩阵。

    回答"为什么要多路"：如果 A 对的样本里 B 全对，那 B 是冗余的；
    只有当"一方错、另一方对"的格子有量，融合才有意义。
    """
    pairs = [
        ("dense_only", "rule_only"),
        ("dense_only", "llm_only"),
        ("rule_only", "llm_only"),
    ]
    matrix = {}
    for a, b in pairs:
        if a not in hits or b not in hits:
            continue
        common = set(hits[a]) & set(hits[b])
        both = [q for q in common if hits[a][q] and hits[b][q]]
        only_a = [q for q in common if hits[a][q] and not hits[b][q]]
        only_b = [q for q in common if not hits[a][q] and hits[b][q]]
        neither = [q for q in common if not hits[a][q] and not hits[b][q]]
        union = len(both) + len(only_a) + len(only_b)
        matrix[f"{a} vs {b}"] = {
            "两者都对": len(both),
            f"仅{a}对": len(only_a),
            f"仅{b}对": len(only_b),
            "两者都错": len(neither),
            "并集正确数": union,
            "互补率": round((len(only_a) + len(only_b)) / max(union, 1), 4),
        }
    return matrix


def render_markdown(report: dict) -> str:
    lines = [
        f"## 意图识别评测",
        "",
        "### 一、同源评测（n={}，**分数偏高，只看机制不看绝对值**）".format(report["n"]),
        "",
        "> golden query 是围绕 intent exemplar 的同一批业务话术写的，"
        "三条通道里有两条直接在 exemplar 上建库/训练，因此这里的分数必然接近满分 —— "
        "这是泄漏，不是能力。可信结论看下面第二张表。",
        "",
        f"- Accuracy: **{report['accuracy']:.4f}**",
        f"- Macro-F1: **{report['macro_f1']:.4f}**",
        f"- 弃权率(need_clarify): {report['abstain_rate']:.2%}",
        f"- 作答样本准确率: {report['accuracy_on_answered']:.4f}",
        "",
        "| 意图 | support | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for label, v in report["per_class"].items():
        if v["support"] == 0:
            continue
        lines.append(
            f"| {label} | {v['support']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |"
        )

    holdout = report.get("holdout")
    if holdout:
        fd, ft = holdout["fusion_default"], holdout["fusion_tuned"]
        lines.extend(
            [
                "",
                "### 二、留出实验（exemplar 5 折交叉，可信版本）",
                "",
                f"前 3 折作 dev 标定权重，后 2 折作 test 只报一次。样本：dev {holdout['dev_n']} / test {holdout['test_n']}。",
                "",
                "| 通道 | dev 准确率 | test 准确率 |",
                "|---|---|---|",
            ]
        )
        for route, m in holdout["routes"].items():
            lines.append(f"| {route}_only | {m['dev']:.4f} | {m['test']:.4f} |")
        lines.append(f"| 融合·初始假设权重（未标定）{fd['weights']} | {fd['dev']:.4f} | {fd['test']:.4f} |")
        lines.append(f"| 融合·dev 标定权重（已设为默认）{ft['weights']} | {ft['dev']:.4f} | {ft['test']:.4f} |")
        best_single_test = max(m["test"] for m in holdout["routes"].values())
        lines.append("")
        lines.append(
            f"> test 上最强单路 {best_single_test:.4f}；默认权重融合 {fd['test']:.4f}，"
            f"dev 标定后 {ft['test']:.4f}。"
        )
        if fd["test"] < best_single_test - 0.005:
            lines.append(">")
            lines.append(
                "> 默认权重把最弱的一路给了最高话语权（llm 0.55），融合因此明显低于最强单路。"
                f"在 dev 上标定后权重变为 {ft['weights']}，说明**权重是超参，必须由数据决定，不能拍脑袋**。"
            )
        lines.append(">")
        lines.append(
            "> 注意：每折留出后每类只剩 4 条训练样本，属于极端小样本。"
            "这里的差距主要反映「小样本下哪条路更稳」，不能外推成「向量路无用」。"
        )

    ablation = report.get("route_ablation")
    if ablation:
        lines.extend(
            [
                "",
                "### 三、通道消融（同源数据，只看互补结构）",
                "",
                "| 通道组合 | Accuracy | Macro-F1 | 弃权率 | 作答准确率 |",
                "|---|---|---|---|---|",
            ]
        )
        for name, m in ablation.items():
            if name.startswith("_"):
                continue
            lines.append(
                f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                f"{m['abstain_rate']:.2%} | {m['accuracy_on_answered']:.4f} |"
            )

        comp = ablation.get("_complementarity") or {}
        if comp:
            lines.extend(["", "#### 通道互补性（谁救回了谁）", ""])
            lines.append("| 通道对 | 两者都对 | 仅前者对 | 仅后者对 | 都错 | 并集正确 | 互补率 |")
            lines.append("|---|---|---|---|---|---|---|")
            for pair, m in comp.items():
                keys = list(m)
                lines.append(
                    f"| {pair} | {m[keys[0]]} | {m[keys[1]]} | {m[keys[2]]} | {m[keys[3]]} "
                    f"| {m[keys[4]]} | {m[keys[5]]:.2%} |"
                )
            lines.append("")
            lines.append(
                "> 互补率非零说明该路不是冗余的。但要注意：在 exemplar 上训练的通道会同时满分，"
                "互补率因此被压到 0 —— 这本身就是泄漏的又一个证据。"
            )

    if report["errors"]:
        lines.append("")
        lines.append("### 错分样本（同源评测）")
        for err in report["errors"][:12]:
            lines.append(
                f"- `{err['query']}` → 预测 {err['pred']}（gold {err['gold']}，"
                f"conf={err['confidence']:.2f}，margin={err['margin']:.2f}）"
            )
    return "\n".join(lines) + "\n"
