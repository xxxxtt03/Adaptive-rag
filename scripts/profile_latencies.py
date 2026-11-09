"""Micro-benchmark and Latency Profiling for Adaptive-RAG Sub-components.

Measures exact millisecond execution times on current hardware:
1. Prototypical Intent Router (local centroid cosine similarity)
2. Dense Embedding (BAAI/bge-small-zh-v1.5)
3. Lexical BM25 Retrieval
4. Reciprocal Rank Fusion (RRF k=60)
5. Cross-Encoder Cascade Reranker (BAAI/bge-reranker-base)
6. Graph-RAG in-memory 1-hop constraint traversal
"""

import asyncio
import sys
import time

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.config.settings import get_settings
from app.retrieval.graph_rag import build_ecommerce_policy_graph
from app.services.container import build_container


async def profile():
    settings = get_settings()
    print("=" * 75)
    print("⏱️ Adaptive-RAG 核心算法底层毫秒级耗时实测 (Latency Profiling)")
    print("=" * 75)

    container = await build_container(settings)
    try:
        retriever = container.retriever
        reranker = container.chat_service._reranker
        embedder = getattr(retriever, "embedder", None) or retriever._vector._embedder

        query = "定制马克杯过了七天支持退换货吗？"
        print(f"\n🔍 测试样本 Query: '{query}'")
        print("-" * 75)

        # -------------------------------------------------------------
        # 1. Graph-RAG 内存图遍历耗时
        # -------------------------------------------------------------
        graph = build_ecommerce_policy_graph()
        # Warmup
        graph.query(query)
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            res = graph.query(query)
            times.append((time.perf_counter() - t0) * 1000)
        avg_graph = sum(times) / len(times)
        min_graph = min(times)
        print(f"1️⃣ 【Graph-RAG 内存有向图遍历】")
        print(f"   • 平均耗时: {avg_graph:.3f} ms | 最快: {min_graph:.3f} ms")
        print(f"   • 命中实体: {res.matched_entities}")
        print(f"   • 规则约束数: {len(res.constraints)} 条确定性边")

        # -------------------------------------------------------------
        # 2. Dense Embedding 稠密向量编码耗时
        # -------------------------------------------------------------
        # Warmup
        await embedder.embed([query])
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            vecs = await embedder.embed([query])
            times.append((time.perf_counter() - t0) * 1000)
        avg_embed = sum(times) / len(times)
        min_embed = min(times)
        print(f"\n2️⃣ 【Dense Embedding (bge-small-zh-v1.5, 512-dim)】")
        print(f"   • 平均耗时: {avg_embed:.2f} ms | 最快: {min_embed:.2f} ms")
        print(f"   • 向量维度: {len(vecs[0])} 维")

        # -------------------------------------------------------------
        # 3. BM25 词法检索分词与打分耗时
        # -------------------------------------------------------------
        # Warmup
        retriever._bm25.search(query, top_k=8)
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            bm25_hits = retriever._bm25.search(query, top_k=8)
            times.append((time.perf_counter() - t0) * 1000)
        avg_bm25 = sum(times) / len(times)
        min_bm25 = min(times)
        print(f"\n3️⃣ 【BM25 词法检索 (含中文分词 & 130 篇文档打分)】")
        print(f"   • 平均耗时: {avg_bm25:.3f} ms | 最快: {min_bm25:.3f} ms")
        print(f"   • 召回切片数: {len(bm25_hits)} 个候选")

        # -------------------------------------------------------------
        # 4. Vector 余弦检索耗时
        # -------------------------------------------------------------
        # Warmup
        await retriever._vector.search(query, top_k=8)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            vec_hits = await retriever._vector.search(query, top_k=8)
            times.append((time.perf_counter() - t0) * 1000)
        avg_vec = sum(times) / len(times)
        min_vec = min(times)
        print(f"\n4️⃣ 【Vector 稠密余弦检索 (含 Embedding 编码 + 相似度排序)】")
        print(f"   • 平均耗时: {avg_vec:.2f} ms | 最快: {min_vec:.2f} ms")
        print(f"   • 召回切片数: {len(vec_hits)} 个候选")

        # -------------------------------------------------------------
        # 5. RRF 倒数排名融合算法耗时
        # -------------------------------------------------------------
        from app.retrieval.fusion import reciprocal_rank_fusion
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            fused = reciprocal_rank_fusion([bm25_hits, vec_hits], k=60)
            times.append((time.perf_counter() - t0) * 1000)
        avg_rrf = sum(times) / len(times)
        min_rrf = min(times)
        print(f"\n5️⃣ 【RRF 倒数排名融合 (k=60, 双路候选合并与单调序去重)】")
        print(f"   • 平均耗时: {avg_rrf:.4f} ms | 最快: {min_rrf:.4f} ms")
        print(f"   • 融合后候选数: {len(fused)} 个切片")

        # -------------------------------------------------------------
        # 6. Cross-Encoder 交叉注意力重排耗时
        # -------------------------------------------------------------
        # Warmup
        await reranker.rerank(query, fused[:8], top_n=5)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            reranked = await reranker.rerank(query, fused[:8], top_n=5)
            times.append((time.perf_counter() - t0) * 1000)
        avg_ce = sum(times) / len(times)
        min_ce = min(times)
        print(f"\n6️⃣ 【Cross-Encoder 精排 (bge-reranker-base, 对 Top-8 全词注意力打分)】")
        print(f"   • 平均耗时: {avg_ce:.2f} ms | 最快: {min_ce:.2f} ms")
        print(f"   • 精排 Top-1: {reranked[0].title} (chunk_id={reranked[0].chunk_id})")

        # -------------------------------------------------------------
        # 总结
        # -------------------------------------------------------------
        total_retrieval_pipeline = avg_bm25 + avg_vec + avg_rrf + avg_ce + avg_graph
        print("\n" + "=" * 75)
        print(f"📊 【混合检索与图谱全链路总耗时】: 约 {total_retrieval_pipeline:.2f} ms (CPU 单机实测)")
        print(f"   💡 其中 Cross-Encoder 耗时占比: {(avg_ce / total_retrieval_pipeline) * 100:.1f}%")
        print("=" * 75)

    finally:
        await container.aclose()


if __name__ == "__main__":
    asyncio.run(profile())
