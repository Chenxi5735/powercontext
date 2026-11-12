# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP score-service adapter for identity-preserving Memory reranking."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score complete-pool ranks in bounded HTTP batches, then select globally."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer")  # noqa: TRY003
        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if limit < 1 or limit > len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003
        scores: dict[int, float] = {}

        for offset in range(0, len(candidates), self._batch_size):
            batch = candidates[offset : offset + self._batch_size]
            expected_ranks = set(range(offset + 1, offset + len(batch) + 1))
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text}
                    for rank, candidate in enumerate(batch, start=offset + 1)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            try:
                body = response.json()
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError("rerank service response must be valid JSON") from error  # noqa: TRY003
            scores.update(_validate_batch_scores(body, expected_ranks))

        selected = tuple(rank for rank, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=math.ceil(len(candidates) / self._batch_size)),
        )


def _validate_batch_scores(body: Any, expected_ranks: set[int]) -> dict[int, float]:
    if not isinstance(body, dict) or set(body) != {"scores"} or not isinstance(body["scores"], list):
        raise ValueError("rerank service response must contain only a scores array")  # noqa: TRY003

    scores: dict[int, float] = {}
    for item in body["scores"]:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("each rerank score must contain only rank and score")  # noqa: TRY003
        rank = item["rank"]
        score = item["score"]
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError("rerank ranks must be integers")  # noqa: TRY003
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("rerank scores must be finite numbers")  # noqa: TRY003
        try:
            numeric_score = float(score)
        except (OverflowError, ValueError) as error:
            raise ValueError("rerank scores must be finite numbers") from error  # noqa: TRY003
        if not math.isfinite(numeric_score):
            raise ValueError("rerank scores must be finite numbers")  # noqa: TRY003
        if rank not in expected_ranks:
            raise ValueError("rerank response contains a missing, extra, or cross-batch rank")  # noqa: TRY003
        if rank in scores:
            raise ValueError("rerank response contains duplicate ranks")  # noqa: TRY003
        scores[rank] = numeric_score

    if set(scores) != expected_ranks:
        raise ValueError("rerank response must cover every rank in its batch exactly once")  # noqa: TRY003
    return scores


__all__ = ["HTTPMemoryReranker"]
