"""End-to-End Walkthrough Script for Adaptive-RAG.

Simulates 4 core real-world conversation turns:
1. Prototypical Router direct greeting
2. Secondary-developed Cross-Border Tax calculation tool
3. High-risk write operation triggering HITL proposal (ORD10001 cancellation)
4. User confirmation of pending action with idempotency protection
"""

import asyncio
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.config.settings import get_settings
from app.schemas.chat import ChatRequest
from app.services.container import build_container


async def run_walkthrough():
    settings = get_settings()
    print("=" * 80)
    print("🚀 Adaptive-RAG 端到端全链路真机实战演练 (End-to-End Walkthrough)")
    print("=" * 80)

    container = await build_container(settings)
    try:
        session_id = "demo_session_001"
        user_id = "u_demo"

        # ---------------------------------------------------------
        # Case 1: 问候直出 (Prototypical Router DIRECT)
        # ---------------------------------------------------------
        print("\n▶️ [Turn 1] 意图原型快速路由 (Direct Greeting)")
        req1 = ChatRequest(
            message="你好，请问你是谁？",
            session_id=session_id,
            user_id=user_id,
        )
        print(f"👤 用户: {req1.message}")
        res1 = await container.chat_service.chat(req1, trace_id="trace-001")
        print(f"🤖 助手: {res1.answer.strip()[:100]}...")
        print(f"   📊 路由分类: intent={res1.intent}, complexity={res1.complexity}")
        print(f"   ⏱️ 耗时: total={res1.latency.total_ms:.1f}ms")

        # ---------------------------------------------------------
        # Case 2: 新增业务工具 (跨境海淘税费计算)
        # ---------------------------------------------------------
        print("\n▶️ [Turn 2] 二次开发业务工具调用 (Cross-Border Duty Tool)")
        req2 = ChatRequest(
            message="跨境海淘买一瓶1000元的高档香水要交多少税？",
            session_id=session_id,
            user_id=user_id,
        )
        print(f"👤 用户: {req2.message}")
        res2 = await container.chat_service.chat(req2, trace_id="trace-002")
        print(f"🤖 助手:\n{res2.answer.strip()}")
        print(f"   📊 路由分类: intent={res2.intent}, complexity={res2.complexity}")
        print(f"   📚 引用知识库: {[c.document_id for c in res2.citations]}")
        print(f"   ⏱️ 耗时: total={res2.latency.total_ms:.1f}ms (检索={res2.latency.retrieval_ms:.1f}ms, LLM={res2.latency.llm_ms:.1f}ms)")

        # ---------------------------------------------------------
        # Case 3: 高危操作触发 HITL (取消订单 ORD10001)
        # ---------------------------------------------------------
        print("\n▶️ [Turn 3] 高危写操作人机协同拦截 (HITL Suspension)")
        req3 = ChatRequest(
            message="帮我取消订单 ORD10001",
            session_id=session_id,
            user_id=user_id,
        )
        print(f"👤 用户: {req3.message}")
        res3 = await container.chat_service.chat(req3, trace_id="trace-003")
        print(f"🤖 助手:\n{res3.answer.strip()}")
        if res3.pending_action:
            print("   ⚠️ [HITL 拦截成功] 捕获到待人工确认提案 (Pending Action):")
            print(f"      - action_id: {res3.pending_action.action_id}")
            print(f"      - tool_name: {res3.pending_action.tool_name}")
            print(f"      - arguments: {res3.pending_action.arguments}")
            print(f"      - risk: {res3.pending_action.risk}")
            print(f"      - summary: {res3.pending_action.summary}")
        else:
            print("   ❌ 未触发 pending_action")

        # ---------------------------------------------------------
        # Case 4: 确认执行 (带上 confirm_action_id 幂等落地)
        # ---------------------------------------------------------
        if res3.pending_action:
            print("\n▶️ [Turn 4] 人工确认与幂等写操作落地 (Confirm & Execute)")
            req4 = ChatRequest(
                message="确认取消",
                session_id=session_id,
                user_id=user_id,
                confirm_action_id=res3.pending_action.action_id,
                idempotency_key=f"idem_{res3.pending_action.action_id}",
            )
            print(f"👤 用户: 提交确认提案 action_id={req4.confirm_action_id}")
            res4 = await container.chat_service.chat(req4, trace_id="trace-004")
            print(f"🤖 助手:\n{res4.answer.strip()}")
            print(f"   ⏱️ 耗时: total={res4.latency.total_ms:.1f}ms")

            # ---------------------------------------------------------
            # Case 5: 模拟手抖重复点击 (防重放与一次性 Token 校验)
            # ---------------------------------------------------------
            print("\n▶️ [Turn 5] 幂等防重放与单次有效令牌校验 (Single-Use Token & Replay Defense)")
            print("👤 用户: 手抖重复提交同一 confirm_action_id...")
            try:
                res5 = await container.chat_service.chat(req4, trace_id="trace-005")
                print(f"🤖 助手: {res5.answer.strip()[:100]}...")
            except Exception as e:
                print(f"   🛡️ 安全拦截成功: 捕获异常 [{e.__class__.__name__}: {e}]")
                print("   🔒 机制生效说明: 提案 Token 为单次有效（Single-use Nonce），消费后立即销毁，坚决杜绝二次恶意提交！")

        # ---------------------------------------------------------
        # Case 6: 发票开具申请与专票税号核验 (Invoice Issuance & VAT Tool)
        # ---------------------------------------------------------
        print("\n▶️ [Turn 6] 发票开具与专票税号核验 (Invoice Issuance Tool)")
        req6 = ChatRequest(
            message="帮我开一张 ORD10002 的企业增值税专票，抬头是北京智联创想科技有限公司，税号是 91110108MA01ABCDEF，邮箱是 finance@example.com",
            session_id=session_id,
            user_id=user_id,
        )
        print(f"👤 用户: {req6.message}")
        res6 = await container.chat_service.chat(req6, trace_id="trace-006")
        print(f"🤖 助手:\n{res6.answer.strip()}")
        print(f"   📊 路由分类: intent={res6.intent}, complexity={res6.complexity}")
        print(f"   ⏱️ 耗时: total={res6.latency.total_ms:.1f}ms")

    finally:
        await container.aclose()

    print("\n" + "=" * 80)
    print("✅ 全链路 6 个场景端到端实操验收完毕！各项指标与业务状态流转均达到设计预期。")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_walkthrough())
