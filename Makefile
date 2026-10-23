PY := /Users/cqr/.workbuddy/binaries/python/envs/default/bin/python

.PHONY: help install index serve smoke test eval eval-retrieval eval-intent eval-dialogue

help:
	@echo "install        安装依赖"
	@echo "index          重建知识库索引（含 LSA 训练）"
	@echo "serve          启动 API + 轨迹面板 (http://127.0.0.1:8000)"
	@echo "smoke          6 条典型链路冒烟"
	@echo "test           单测"
	@echo "eval           全部评测（检索 / 意图 / 端到端 / 规模）"
	@echo "bench          记忆压缩量化 + 延迟分解"
	@echo "scale          规模敏感性实验（近重复干扰注入）"
	@echo "embedder       向量后端对比（hashing / lsa / api）"
	@echo "all            一键跑完 test + eval + bench + embedder"

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

bench:
	$(PY) -m scripts.bench all

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

all: test eval bench embedder
