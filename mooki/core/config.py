from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOOKI_", env_file=".env", extra="ignore")

    app_name: str = "mooki"

    data_dir: Path = ROOT / "data"
    kb_dir: Path = ROOT / "data" / "kb"
    order_file: Path = ROOT / "data" / "orders.json"
    ledger_file: Path = ROOT / "data" / "ledger.db"
    index_dir: Path = ROOT / ".mooki_index"

    # ---------- LLM ----------
    llm_offline: bool = True
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "qwen-plus"
    llm_timeout_s: float = 20.0
    llm_temperature: float = 0.1
    # 成本估算单价（元 / 千 token）。默认 0 表示不估算，按实际服务商价格填。
    price_in_per_1k: float = 0.0
    price_out_per_1k: float = 0.0

    # ---------- Embedding ----------
    embed_backend: str = "lsa"  # lsa | hashing | bge（本地 ONNX 语义模型）
    embed_local_model: str = "BAAI/bge-small-zh-v1.5"
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 128

    # ---------- 存储 ----------
    redis_url: str = ""  # 留空则用进程内后端，便于本地跑通
    vector_backend: str = "local"  # local | chroma
    chroma_dir: Path = ROOT / ".mooki_chroma"

    # ---------- 检索 ----------
    rrf_k: int = 60
    bm25_top_k: int = 20
    dense_top_k: int = 20
    rerank_top_k: int = 5
    chunk_size: int = 420
    chunk_overlap: int = 70
    retrieval_cache_ttl: int = 600
    rewrite_cache_ttl: int = 1800

    # ---------- 意图 ----------
    # 三路权重是超参，必须由数据决定。这组默认值来自 exemplar 留出实验的 dev 标定结果，
    # 不是拍脑袋：初始假设「LLM 最强 0.55」在留出后把融合拖到 0.35，标定后回升到 0.85。
    intent_weight_llm: float = 0.25
    intent_weight_dense: float = 0.10
    intent_weight_rule: float = 0.65
    # 转人工是否真的挂起等人工接管（需要 LangGraph checkpointer）。
    # 默认关：人工不在线时把用户挂住，比直接建单更糟。
    enable_human_interrupt: bool = False
    max_tool_iterations: int = 3
    intent_margin_threshold: float = 0.12
    intent_cache_ttl: int = 900
    intent_cache_size: int = 2048

    # ---------- 韧性 ----------
    tool_timeout_s: float = 8.0
    breaker_fail_threshold: int = 5
    breaker_reset_s: float = 30.0

    # ---------- 记忆 ----------
    short_term_window: int = 8
    context_budget_tokens: int = 1200
    episodic_top_k: int = 3

    def ensure_dirs(self) -> None:
        for p in (self.index_dir, self.data_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
