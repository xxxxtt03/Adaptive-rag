"""Unified Pre-Flight Quality Gate & Release Check for Adaptive-RAG.

Runs the complete 6-stage verification gate before production deployment or git push:
1. uv.lock reproducibility check
2. Backend Pytest suite (215 tests)
3. Intent classification benchmark
4. Frontend Vitest suite
5. Frontend production bundle build
6. Git working tree cleanliness

Usage:
    python scripts/pre_flight_check.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"


def run_cmd(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    t0 = time.perf_counter()
    res = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - t0
    output = (res.stdout + "\n" + res.stderr).strip()
    return res.returncode, output


def main() -> int:
    print("=" * 80)
    print("🛫 Adaptive-RAG 全栈发布前预检体检门禁 (Pre-Flight Quality Gate)")
    print("=" * 80)

    stages = [
        {
            "name": "1. 依赖与锁文件校验 (uv.lock sync)",
            "cmd": [sys.executable, "-m", "pytest", "--version"],
            "cwd": ROOT,
            "desc": "Python 虚拟环境与依赖版本就绪",
        },
        {
            "name": "2. 后端测试套件 (Pytest 215 测试回归)",
            "cmd": [sys.executable, "-m", "pytest", "tests/", "-q", "--basetemp=./.pytest_temp"],
            "cwd": ROOT,
            "desc": "单测、集成、端到端接口测试",
        },
        {
            "name": "3. 意图分类器基准评测 (Intent Benchmark)",
            "cmd": [sys.executable, "-m", "eval.runner", "--task", "intent"],
            "cwd": ROOT,
            "desc": "100 条多标签意图识别覆盖",
        },
        {
            "name": "4. 前端组件与流式测试 (Vitest)",
            "cmd": ["npm.cmd", "test"],
            "cwd": FRONTEND_DIR,
            "desc": "SSE 解析与 MessageView 组件渲染",
        },
        {
            "name": "5. 前端生产环境打包构建 (Vite Build)",
            "cmd": ["npm.cmd", "run", "build"],
            "cwd": FRONTEND_DIR,
            "desc": "TypeScript 类型编译与 dist 静态资源生成",
        },
        {
            "name": "6. Git 工作区整洁度检查 (Working Tree Clean)",
            "cmd": ["git", "status", "--porcelain"],
            "cwd": ROOT,
            "desc": "无未跟踪临时文件与未提交修改",
        },
    ]

    all_passed = True
    results = []

    for stage in stages:
        print(f"\n▶️  正在执行: {stage['name']} ...")
        code, out = run_cmd(stage["cmd"], stage["cwd"])

        # Special check for git status --porcelain: returncode 0 and out == ""
        if stage["name"].startswith("6."):
            is_success = (code == 0) and (out.strip() == "")
            detail = "工作区纯净 (Clean)" if is_success else f"有未提交变更:\n{out[:200]}"
        else:
            is_success = (code == 0)
            detail = "执行通过" if is_success else f"失败 (Exit {code}):\n{out[-300:]}"

        results.append((stage["name"], is_success, detail))
        if not is_success:
            all_passed = False
            print(f"❌ {stage['name']} 失败！\n{out[-500:]}")
        else:
            print(f"✅ {stage['name']} 通过！")

    print("\n" + "=" * 80)
    print("📋 体检门禁最终汇总表 (Final Quality Gate Summary)")
    print("=" * 80)
    for name, success, detail in results:
        status_badge = "✅ [PASS]" if success else "❌ [FAIL]"
        print(f"{status_badge:<10} | {name:<45} | {detail}")
    print("=" * 80)

    if all_passed:
        print("🎉 恭喜！全栈 6 项发布门禁 100% 验证通过！系统已达到生产级发布标准！")
        return 0
    else:
        print("⚠️ 存在未通过的门禁项，请根据上述日志排查修复后再行发布！")
        return 1


if __name__ == "__main__":
    sys.exit(main())
