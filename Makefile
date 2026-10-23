# 解释器优先级：显式 PY= > .venv > PATH 里的 python3
VENV ?= .venv
PY ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,$(shell command -v python3 2>/dev/null || command -v python))

.PHONY: help venv install index serve smoke test eval eval-retrieval eval-intent \
        eval-dialogue eval-multiturn eval-online eval-online-retrieval bench scale embedder concurrency all clean

help:
	@echo "venv            创建 .venv（可选，未创建时直接用 python3）"
	@echo "install         安装依赖"
	@echo "index           重建知识库索引（含向量模型训练）"
	@echo "serve           启动 API + 轨迹面板 (http://127.0.0.1:8000)"
	@echo "smoke           6 条典型链路冒烟"
	@echo "test            单测"
	@echo "eval            全部离线评测（检索 / 意图 / 端到端 / 多轮 / 规模）"
	@echo "eval-online     在线模式端到端评测（需 .env 配好 LLM），产出延迟与 token 成本"
	@echo "eval-online-retrieval  在线检索对照（只换改写/重排，向量不动）"
	@echo "bench           记忆压缩量化 + 延迟分解"
	@echo "concurrency     退款幂等并发压测（16 并发撞同一请求）"
	@echo "scale           规模敏感性实验（近重复干扰注入）"
	@echo "embedder        向量后端对比（hashing / lsa / api）"
	@echo "all             一键跑完 test + eval + bench + concurrency + embedder"

venv:
	$(PY) -m venv $(VENV)
	@echo "已创建 $(VENV)：$(VENV)/bin/python -m pip install -r requirements.txt"

install:
	$(PY) -m pip install -r requirements.txt

index:
	$(PY) -m scripts.build_index

serve:
	$(PY) -m uvicorn mooki.api.main:app --host 127.0.0.1 --port 8000 --reload

smoke:
	$(PY) -m scripts.smoke

test:
	$(PY) -m unittest discover -s tests -t .

eval:
	$(PY) -m scripts.run_eval all

eval-online:
	$(PY) -m scripts.run_online_eval

eval-online-retrieval:
	$(PY) -m scripts.online_retrieval_ablation

bench:
	$(PY) -m scripts.bench all

concurrency:
	$(PY) -m scripts.concurrency_check

scale:
	$(PY) -m scripts.run_eval scale

embedder:
	$(PY) -m scripts.embedder_ablation

eval-retrieval:
	$(PY) -m scripts.run_eval retrieval

eval-intent:
	$(PY) -m scripts.run_eval intent

eval-dialogue:
	$(PY) -m scripts.run_eval dialogue

eval-multiturn:
	$(PY) -m scripts.run_eval multiturn

all: test eval bench concurrency embedder

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .mooki_index
