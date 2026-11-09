"""Interactive Server Launcher with Health Check and Auto-Browser Opener.

Usage:
    python scripts/start_server.py [--port 8000] [--no-browser]
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys
import time
import webbrowser
from pathlib import Path

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import httpx
import uvicorn

BANNER = r"""
    ___       __             __  _               ____  ___   ______
   /   | ____/ /___ _____   / /_(_)   _____     / __ \/   | / ____/
  / /| |/ __  / __ `/ __ \ / __/ / | / / _ \   / /_/ / /| |/ / __  
 / ___ / /_/ / /_/ / /_/ // /_/ /| |/ /  __/  / _, _/ ___ / /_/ /  
/_/  |_\__,_/\__,_/ .___/ \__/_/ |___/\___/  /_/ |_/_/  |_\____/   
                 /_/                                               
"""


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


async def wait_for_server_ready(url: str, timeout: float = 15.0) -> bool:
    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        while time.monotonic() - t0 < timeout:
            try:
                resp = await client.get(f"{url}/api/v1/ready", timeout=1.0)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.4)
    return False


async def launch_browser_when_ready(url: str) -> None:
    ready = await wait_for_server_ready(url)
    if ready:
        print(f"\n🎉 服务健康就绪！正在自动为您打开浏览器: {url}")
        webbrowser.open(url)
    else:
        print(f"\n⚠️  等待服务就绪超时，请手动打开浏览器访问: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive-RAG Server Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    print(BANNER)
    print("=" * 80)
    print(f"🚀 Adaptive-RAG 智能客服工作台启动中...")
    print(f"🌐 监听地址: http://{args.host}:{args.port}")
    print("=" * 80)
    print("可用核心入口导航:")
    print(f"  • 💻 Web 对话工作台:   http://{args.host}:{args.port}")
    print(f"  • 📖 Swagger API 文档: http://{args.host}:{args.port}/docs")
    print(f"  • 📚 ReDoc 离线文档:   http://{args.host}:{args.port}/redoc")
    print(f"  • 📊 监控指标端点:     http://{args.host}:{args.port}/metrics")
    print(f"  • 💓 健康检查探针:     http://{args.host}:{args.port}/api/v1/ready")
    print("=" * 80)
    print("按 Ctrl + C 可优雅关闭服务。\n")

    if is_port_in_use(args.port, args.host):
        print(f"❌ 端口 {args.port} 已被占用，请更换端口运行 (例如: python scripts/start_server.py --port 8002)")
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        # Schedule browser opening in a background thread or async loop
        import threading
        def _bg_wait():
            asyncio.run(launch_browser_when_ready(url))
        threading.Thread(target=_bg_wait, daemon=True).start()

    # Launch uvicorn
    uvicorn.run("app.asgi:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
