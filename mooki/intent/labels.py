"""意图体系定义：意图 -> 任务类型 -> 工具白名单。

这张表是整个"受控"设计的收敛点：Agent 能做哪些动作，完全由意图查表决定，
不依赖模型自觉。新增能力必须先进这张表，工具本身改不了自己的可达性。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mooki.core.schema import Intent, TaskType, Urgency


@dataclass(frozen=True)
class IntentSpec:
    intent: Intent
    label_zh: str
    task_type: TaskType
    tools: tuple[str, ...]
    keywords: tuple[str, ...]
    exemplars: tuple[str, ...]
    clarify: str
    extra: dict = field(default_factory=dict)


INTENT_SPECS: dict[Intent, IntentSpec] = {
    Intent.ORDER_STATUS: IntentSpec(
        intent=Intent.ORDER_STATUS,
        label_zh="订单状态",
        task_type=TaskType.QUERY,
        tools=("order.query", "kb.search"),
        keywords=("订单", "单号", "什么时候发", "还没发", "查订单", "订单状态", "出库"),
        exemplars=(
            "帮我查一下我的订单到哪一步了",
            "订单 SO20260512001 现在是什么状态",
            "我昨天下单的，怎么还没发货",
            "看看这个单子有没有出库",
            "订单显示已支付，但一直没动静",
        ),
        clarify="你是想查订单进度，还是要取消订单？",
    ),
    Intent.REFUND: IntentSpec(
        intent=Intent.REFUND,
        label_zh="退款",
        task_type=TaskType.TRANSACTION,
        tools=("order.query", "refund.apply", "kb.search"),
        keywords=("退款", "退钱", "钱没退", "退款到账", "退回来", "原路退回", "申请退款", "退多少", "退我多少"),
        exemplars=(
            "我要退款，东西不想要了",
            "退款申请提交三天了钱还没到账",
            "这个订单能退多少钱",
            "退款什么时候原路返回",
            "取消订单以后钱退到哪里了",
        ),
        clarify="这笔订单你是想仅退款，还是退货后再退款？",
    ),
    Intent.RETURN_EXCHANGE: IntentSpec(
        intent=Intent.RETURN_EXCHANGE,
        label_zh="退换货",
        task_type=TaskType.TRANSACTION,
        tools=("order.query", "return.create", "kb.search"),
        keywords=("退货", "换货", "换一个", "换大", "换小", "尺码", "大一码", "小一号", "退回商品", "七天无理由", "寄回", "换新"),
        exemplars=(
            "收到的东西有质量问题，我要退货",
            "尺码不合适想换一个大一码的",
            "七天无理由怎么操作退货",
            "这个可以换货吗，颜色发错了",
            "退货要自己付运费吗",
        ),
        clarify="你是想退货退款，还是换一件同款？",
    ),
    Intent.LOGISTICS: IntentSpec(
        intent=Intent.LOGISTICS,
        label_zh="物流配送",
        task_type=TaskType.QUERY,
        tools=("order.query", "logistics.track", "kb.search"),
        keywords=("物流", "快递", "配送", "到哪了", "签收", "派送", "运费", "发货时间"),
        exemplars=(
            "快递一直停在揽收，是不是丢件了",
            "物流信息三天没更新了",
            "什么时候能送到，我着急用",
            "显示已签收但我没收到",
            "发的是什么快递，能改地址吗",
        ),
        clarify="你是想查物流轨迹，还是想改收货地址？",
    ),
    Intent.COMPLAINT: IntentSpec(
        intent=Intent.COMPLAINT,
        label_zh="投诉",
        task_type=TaskType.ESCALATE,
        tools=("order.query", "ticket.create", "kb.search"),
        keywords=("投诉", "12315", "曝光", "起诉", "报警", "消协", "欺诈", "骗人", "态度差", "举报", "虚假宣传"),
        exemplars=(
            "我要投诉你们的客服态度",
            "再不处理我就去12315投诉",
            "这是欺诈，我要曝光你们",
            "连续三次都没解决，我要找媒体",
            "你们这是虚假宣传，我要举报",
        ),
        clarify="方便说下具体是哪一单、哪个环节的问题吗？我优先帮你升级处理。",
    ),
    Intent.PRODUCT_CONSULT: IntentSpec(
        intent=Intent.PRODUCT_CONSULT,
        label_zh="商品咨询",
        task_type=TaskType.SEARCH,
        tools=("kb.search",),
        keywords=("怎么用", "参数", "材质", "保修", "适配", "支持吗", "有什么区别", "规格"),
        exemplars=(
            "这个耳机支持主动降噪吗",
            "保修期是多久，坏了怎么修",
            "材质是什么，会不会过敏",
            "这个能适配 iPhone 吗",
            "两款有什么区别，推荐哪个",
        ),
        clarify="你想了解这款商品的哪个方面？参数、保修还是适配性？",
    ),
    Intent.INVOICE: IntentSpec(
        intent=Intent.INVOICE,
        label_zh="发票",
        task_type=TaskType.TRANSACTION,
        tools=("order.query", "invoice.issue", "kb.search"),
        keywords=("发票", "开票", "增值税", "报销", "抬头", "税号"),
        exemplars=(
            "我要开发票，怎么开",
            "发票抬头写错了能重开吗",
            "电子发票在哪里下载",
            "能不能开增值税专用发票",
            "发票什么时候能开出来，我着急报销",
        ),
        clarify="你需要的是电子普通发票还是增值税专用发票？",
    ),
    Intent.PROMO: IntentSpec(
        intent=Intent.PROMO,
        label_zh="优惠活动",
        task_type=TaskType.QUERY,
        tools=("promo.check", "kb.search"),
        keywords=("优惠券", "满减", "活动", "折扣", "百亿补贴", "为什么不能用", "价格保护"),
        exemplars=(
            "优惠券为什么下单用不了",
            "这个商品能叠加满减吗",
            "刚买完就降价了，能价保吗",
            "活动什么时候结束",
            "我的券过期了能补吗",
        ),
        clarify="你是想确认券的使用条件，还是想申请价保？",
    ),
    Intent.ACCOUNT: IntentSpec(
        intent=Intent.ACCOUNT,
        label_zh="账户问题",
        task_type=TaskType.QUERY,
        tools=("kb.search", "ticket.create"),
        keywords=("登录", "账号", "密码", "手机号", "实名", "解绑", "被盗", "注销"),
        exemplars=(
            "账号登不上去了，验证码收不到",
            "我要换绑手机号",
            "账号好像被盗了，订单不是我下的",
            "怎么注销账号",
            "实名认证一直失败",
        ),
        clarify="你遇到的是登录问题，还是想变更绑定手机号？",
    ),
    Intent.OTHER: IntentSpec(
        intent=Intent.OTHER,
        label_zh="其他",
        task_type=TaskType.ESCALATE,
        tools=("kb.search", "ticket.create"),
        keywords=("人工", "转人工", "客服", "在吗", "你好"),
        exemplars=(
            "转人工客服",
            "我要找真人",
            "你们几点上班",
            "在吗，问个事",
            "这个怎么处理",
        ),
        clarify="能具体说下你遇到的问题吗？我好帮你转对部门。",
    ),
}


def tools_for(intent: Intent) -> list[str]:
    return list(INTENT_SPECS[intent].tools)


def task_for(intent: Intent) -> TaskType:
    return INTENT_SPECS[intent].task_type


# ---------- 紧急度：独立判定，不参与意图投票 ----------

URGENCY_RULES: dict[Urgency, tuple[str, ...]] = {
    Urgency.P0: ("投诉", "12315", "曝光", "起诉", "报警", "消协", "欺诈", "骗", "媒体", "举报", "律师"),
    Urgency.P1: ("急", "尽快", "马上", "立刻", "严重", "一直没", "等了", "三天", "一周", "又出问题"),
    Urgency.P3: ("咨询", "想问", "了解一下", "顺便问", "在吗", "你好"),
}


def rule_urgency(text: str) -> Urgency | None:
    for urgency in (Urgency.P0, Urgency.P1, Urgency.P3):
        if any(kw in text for kw in URGENCY_RULES[urgency]):
            return urgency
    return None


ALL_INTENTS: list[Intent] = list(INTENT_SPECS)

# 意图 -> 知识域软先验。映射不上（如 product_consult / other）就返回 None，
# 不做硬过滤 —— 跨域问题是常态，硬过滤直接漏答案。
DOMAIN_HINT: dict[Intent, str] = {
    Intent.ORDER_STATUS: "order",
    Intent.REFUND: "refund",
    Intent.RETURN_EXCHANGE: "return",
    Intent.LOGISTICS: "logistics",
    Intent.COMPLAINT: "complaint",
    Intent.INVOICE: "invoice",
    Intent.PROMO: "promo",
    Intent.ACCOUNT: "account",
}


def domain_for(intent: Intent | None) -> str | None:
    return DOMAIN_HINT.get(intent) if intent else None
