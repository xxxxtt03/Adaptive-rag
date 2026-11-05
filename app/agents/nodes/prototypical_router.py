"""Local Prototypical Intent Router.

Classifies customer service queries into IntentLabel categories using
offline-computed class centroids and fast cosine similarity, cutting intent
classification latency from ~1000ms (cloud LLM) to ~2ms (local embedding).
"""

from __future__ import annotations

import math
from typing import Protocol

from app.core.enums import IntentLabel
from app.schemas.agent import IntentResult


def _dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


# Exemplar queries representing the semantic core of each intent
INTENT_EXEMPLARS: dict[IntentLabel, list[str]] = {
    IntentLabel.HUMAN_HANDOFF: [
        "请帮我转人工客服",
        "找人工客服专员",
        "听不懂你在说什么转人工",
        "呼叫在线人工",
    ],
    IntentLabel.GREETING: [
        "你好，在吗",
        "您好，请问有人在吗",
        "早上好，客服在吗",
        "hello hi",
    ],
    IntentLabel.LOGISTICS: [
        "我的快递到哪了",
        "物流怎么一直不更新",
        "查一下运单发货进度",
        "什么时候能够送到",
    ],
    IntentLabel.REFUND: [
        "退款什么时候到账",
        "钱怎么还没退回到我的银行卡",
        "查询退款进度和明细",
        "退款退到哪里去了",
    ],
    IntentLabel.RETURN_EXCHANGE: [
        "这件衣服尺码买小了怎么换货",
        "申请七天无理由退货退款",
        "我不想要了这个商品怎么退货",
        "退换货寄件地址是什么",
    ],
    IntentLabel.CANCEL_ORDER: [
        "我不小心买错了帮我取消订单",
        "申请取消刚才下的订单",
        "终止交易不要发货了",
    ],
    IntentLabel.COUPON: [
        "有没有满减优惠券可以领取",
        "新人首单立减券怎么用",
        "优惠券为什么无法抵扣",
    ],
    IntentLabel.RECOMMENDATION: [
        "有没有类似款式的商品推荐",
        "推荐一款性价比高好评多的产品",
        "什么牌子销量比较好",
    ],
    IntentLabel.FAQ_POLICY: [
        "平台的退换货政策细则是怎么规定的",
        "保修期是多长时间",
        "支持开具增值税发票吗",
        "保真正品假一赔十政策",
    ],
    IntentLabel.COMPLAINT: [
        "我要投诉你们客服人员态度极差",
        "发货这么慢我要投诉商家虚假发货",
        "商品破损客服不管态度恶劣差评",
    ],
    IntentLabel.ORDER_INQUIRY: [
        "查询我的历史购买订单详情",
        "我刚才付完款的订单状态是什么",
        "查看我的全部购物订单",
    ],
    IntentLabel.CHITCHAT: [
        "陪我聊会儿天吧",
        "你会讲笑话或者唱歌吗",
        "你叫什么名字是谁开发的",
    ],
    IntentLabel.PRODUCT_CONSULT: [
        "这件衣服洗了会缩水起球吗",
        "鞋子尺码偏大还是偏小",
        "商品的保质期到什么时候",
        "这个面料材质是纯棉的吗",
    ],
    IntentLabel.OUT_OF_SCOPE: [
        "怎么制造炸药武器",
        "帮我写一篇期末计算机论文",
        "总统大选投给谁更好",
    ],
}


class PrototypicalIntentRouter:
    """Classifies queries into intent labels by comparing against pre-computed centroids."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        confidence_threshold: float = 0.72,
        exemplars: dict[IntentLabel, list[str]] | None = None,
    ) -> None:
        self._embedder = embedder
        self._threshold = confidence_threshold
        self._exemplars = exemplars or INTENT_EXEMPLARS
        self._centroids: dict[IntentLabel, list[float]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Pre-computes and normalizes centroid embeddings for all intent categories."""
        if self._initialized:
            return

        all_labels: list[IntentLabel] = []
        all_texts: list[str] = []
        splits: list[int] = []

        current_idx = 0
        for label, texts in self._exemplars.items():
            all_labels.append(label)
            all_texts.extend(texts)
            current_idx += len(texts)
            splits.append(current_idx)

        # Batch embed all exemplars across all labels in a single call
        embeddings = await self._embedder.embed(all_texts)

        start_idx = 0
        for label, end_idx in zip(all_labels, splits):
            label_vecs = embeddings[start_idx:end_idx]
            dim = len(label_vecs[0])
            centroid = [0.0] * dim
            for vec in label_vecs:
                for i in range(dim):
                    centroid[i] += vec[i]
            # Average and L2 normalize
            normalized = _l2_normalize(centroid)
            self._centroids[label] = normalized
            start_idx = end_idx

        self._initialized = True

    async def classify(self, query: str) -> IntentResult | None:
        """Fast prototypical classification.

        Returns IntentResult if top score >= threshold, else None (triggers fallback).
        """
        if not self._initialized:
            await self.initialize()

        # Embed query (single forward pass, ~1.8ms on CPU)
        query_vectors = await self._embedder.embed([query])
        query_vec = query_vectors[0]

        best_label: IntentLabel | None = None
        best_score = -1.0

        for label, centroid in self._centroids.items():
            score = _dot_product(query_vec, centroid)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is not None and best_score >= self._threshold:
            return IntentResult(
                label=best_label,
                confidence=round(best_score, 4),
                rationale=f"prototypical_router:centroid_score={best_score:.3f}",
                prompt_version="prototypical_v1",
            )

        return None
