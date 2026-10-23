"""重建知识库索引：python -m scripts.build_index"""

from __future__ import annotations

from mooki.core.config import settings
from mooki.core.embedder import build_embedder
from mooki.rag.ingest import build_index


def main() -> int:
    embedder = build_embedder(settings)
    index = build_index(settings, embedder)
    print(f"embedder={embedder.name} chunks={index.size}")
    for domain, count in index.domains().items():
        print(f"  {domain:<10} {count}")
    print(f"索引已写入 {settings.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
