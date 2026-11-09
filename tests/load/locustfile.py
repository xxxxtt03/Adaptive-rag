"""Locust load testing suite for Adaptive-RAG.

Simulates distributed enterprise traffic with realistic weighted user behaviors:
- 40% Lightweight Fast-Path Greeting / Chitchat
- 30% Hybrid RAG Policy FAQ (BM25 + Dense + RRF + Cross-Encoder)
- 15% Domain Secondary Development Tools (Customs Tax / Invoices)
- 10% Loyalty Profile & Points Redemption
- 5%  Streaming SSE Real-Time Token Generation

Usage:
    pip install locust
    locust -f tests/load/locustfile.py --host http://localhost:8000
"""

from __future__ import annotations

import random
from locust import HttpUser, between, task

GREETINGS = [
    "你好",
    "您好，在吗？",
    "客服你好",
    "hello",
    "早上好",
]

POLICY_FAQS = [
    "7天无理由退货的运费谁承担？",
    "定制马克杯过了七天，杯口开裂能不能退货？",
    "偏远地区（新疆/西藏）包邮吗？",
    "商品有轻微划痕可以换货吗？",
]

TAX_AND_INVOICES = [
    "跨境海淘买一瓶1000元的高档香水要交多少税？",
    "帮我开一张 ORD10002 的企业增值税专票，税号 91110108MA01ABCDEF",
    "个人发票怎么在订单详情里下载？",
]

LOYALTY_QUERIES = [
    "我当前账户有多少积分？能兑换什么优惠券？",
    "金牌会员兑换 50 元无门槛券需要多少积分？",
    "我的积分什么时候过期？",
]


class AdaptiveRAGCustomerUser(HttpUser):
    wait_time = between(0.5, 2.5)

    @task(40)
    def test_greeting_fast_path(self) -> None:
        """40% traffic: Fast-path prototypical router short-circuit (<20ms)."""
        query = random.choice(GREETINGS)
        self.client.post(
            "/api/v1/chat",
            json={
                "message": query,
                "session_id": "locust_session",
                "user_id": "u_demo",
            },
            name="/api/v1/chat [Fast-Path Greeting]",
            timeout=10.0,
        )

    @task(30)
    def test_policy_faq_hybrid_retrieval(self) -> None:
        """30% traffic: Hybrid retrieval + Graph-RAG + Cross-Encoder."""
        query = random.choice(POLICY_FAQS)
        self.client.post(
            "/api/v1/chat",
            json={
                "message": query,
                "session_id": "locust_session",
                "user_id": "u_demo",
            },
            name="/api/v1/chat [Hybrid RAG Policy]",
            timeout=20.0,
        )

    @task(15)
    def test_secondary_dev_tools(self) -> None:
        """15% traffic: Domain calculation tools (Customs tax / Invoices)."""
        query = random.choice(TAX_AND_INVOICES)
        self.client.post(
            "/api/v1/chat",
            json={
                "message": query,
                "session_id": "locust_session",
                "user_id": "u_demo",
            },
            name="/api/v1/chat [Domain Tools]",
            timeout=20.0,
        )

    @task(10)
    def test_loyalty_points(self) -> None:
        """10% traffic: Loyalty points ledger & coupon redemption."""
        query = random.choice(LOYALTY_QUERIES)
        self.client.post(
            "/api/v1/chat",
            json={
                "message": query,
                "session_id": "locust_session",
                "user_id": "u_demo",
            },
            name="/api/v1/chat [Loyalty Points]",
            timeout=20.0,
        )

    @task(5)
    def test_streaming_sse(self) -> None:
        """5% traffic: Real-time SSE streaming typewriter connection."""
        self.client.post(
            "/api/v1/chat/stream",
            json={
                "message": "请简要介绍一下平台的售后服务承诺",
                "session_id": "locust_stream_session",
                "user_id": "u_demo",
            },
            name="/api/v1/chat/stream [SSE Streaming]",
            stream=True,
            timeout=30.0,
        )
