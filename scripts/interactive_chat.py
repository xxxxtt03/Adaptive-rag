"""Interactive Terminal Chat REPL for Adaptive-RAG.

Allows testing conversations interactively from PowerShell with streaming output,
node progress tracking, citation display, and HITL action approval.
"""

import asyncio
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.config.settings import get_settings
from app.core.ids import new_id
from app.schemas.chat import ChatRequest
from app.services.container import build_container


async def main():
    settings = get_settings()
    print("=" * 70)
    print("💬 Adaptive-RAG 交互式终端终端体验 (Interactive Console)")
    print("   输入问题与智能客服对话；输入 'exit' 或 'quit' 退出。")
    print("=" * 70)

    container = await build_container(settings)
    session_id = new_id("sess_cli_")
    user_id = "u_demo"
    pending_action_id = None

    try:
        while True:
            try:
                prompt_prefix = "\n[待确认操作挂起，输入 'y' 确认 / 'n' 取消]\n" if pending_action_id else "\n"
                query = input(f"{prompt_prefix}👤 您: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 已退出对话。")
                break

            if not query:
                continue
            query = query.strip('"\' ')
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("👋 体验结束，感谢使用 Adaptive-RAG！")
                break

            req_confirm_id = None
            if pending_action_id:
                if query.lower() in {"y", "yes", "确认", "确定", "是"}:
                    req_confirm_id = pending_action_id
                    query = "确认执行"
                    print(f"⚡ [HITL] 已提交操作确认凭证: {req_confirm_id}")
                else:
                    print("❌ [HITL] 用户已放弃执行待确认操作。")
                    pending_action_id = None
                    continue

            request = ChatRequest(
                message=query,
                session_id=session_id,
                user_id=user_id,
                confirm_action_id=req_confirm_id,
            )

            print("🤖 助手: ", end="", flush=True)
            current_trace = new_id("trace_")
            final_response = None

            try:
                async for event in container.chat_service.stream(request, trace_id=current_trace):
                    kind = event.get("event")
                    if kind == "node":
                        node_name = event.get("node")
                        # Light indicator of state transitions
                        # print(f"[{node_name}]", end=" ", flush=True)
                    elif kind == "token":
                        print(event.get("text", ""), end="", flush=True)
                    elif kind == "answer":
                        final_response = event.get("data")
                    elif kind == "done":
                        pass
                print()  # newline
            except Exception as e:
                print(f"\n⚠️ 请求处理异常: {e}")
                continue

            if final_response:
                pending_act = final_response.get("pending_action")
                if pending_act:
                    pending_action_id = pending_act.get("action_id")
                    print("\n" + "-" * 50)
                    print(f"⚠️ [HITL 人工确认卡片]")
                    print(f"  • 操作摘要: {pending_act.get('summary')}")
                    print(f"  • 工具名称: {pending_act.get('tool_name')}")
                    print(f"  • 风险级别: {pending_act.get('risk')}")
                    print(f"  • 凭据 Token: {pending_action_id} (15分钟有效)")
                    print("-" * 50)
                else:
                    pending_action_id = None

                citations = final_response.get("citations") or []
                if citations:
                    print("\n📚 [引用来源]:")
                    for idx, c in enumerate(citations, 1):
                        sec = f" · {c.get('section')}" if c.get("section") else ""
                        print(f"  [{idx}] {c.get('title')}{sec} ({c.get('document_id')})")

    finally:
        await container.aclose()


if __name__ == "__main__":
    asyncio.run(main())
