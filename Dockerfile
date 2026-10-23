FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MOOKI_LLM_OFFLINE=true

WORKDIR /app

# 依赖单独一层：改代码不用重装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 镜像内预建索引（只依赖 data/kb，离线可跑）；接入真实 embedding 后删掉这行
RUN python -m scripts.build_index

# 非 root 运行：容器里跑的是对外服务，别用 root
RUN useradd -m -u 10001 mooki && chown -R mooki:mooki /app
USER mooki

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).status==200 else 1)"

CMD ["uvicorn", "mooki.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
