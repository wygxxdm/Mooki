"""Mooki — 电商售后智能客服 Agent 平台。

分层：
    core    基础设施（配置 / LLM / Embedding / 缓存 / 韧性 / 可观测）
    intent  多路融合意图识别
    rag     混合检索链路
    tools   MCP 风格工具层
    memory  三层记忆
    graph   LangGraph 受控单 Agent
    api     FastAPI 接入
    eval    离线评测
"""

__version__ = "0.1.0"
