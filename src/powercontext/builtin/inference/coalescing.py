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

"""Bounded cross-call batching and caching for embedding models."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(eq=False, slots=True)
class _Batch:
    usage: InferenceUsage | None = None


@dataclass(eq=False, slots=True)
class _Work:
    text: str
    future: asyncio.Future[tuple[EmbeddingVector, _Batch]]


class CoalescingEmbeddingModel:
    """Merge concurrent text requests, cache successful vectors, and preserve call usage."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        try:
            valid_coalesce_ms = (
                not isinstance(coalesce_ms, bool)
                and isinstance(coalesce_ms, (int, float))
                and math.isfinite(float(coalesce_ms))
                and coalesce_ms >= 0
            )
        except OverflowError:
            valid_coalesce_ms = False
        if not valid_coalesce_ms:
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, _Work] = {}
        self._pending: list[_Work] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        texts = tuple(texts)
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0, input_tokens=0, output_tokens=0))

        references: dict[str, EmbeddingVector | _Work] = {}
        async with self._lock:
            for text in dict.fromkeys(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    self._cache.move_to_end(text)
                    references[text] = cached
                    continue
                work = self._inflight.get(text)
                if work is None:
                    work = _Work(text=text, future=asyncio.get_running_loop().create_future())
                    self._inflight[text] = work
                    self._pending.append(work)
                references[text] = work
            if self._pending and self._flush_task is None:
                self._flush_task = asyncio.create_task(self._flush_pending())

        works = tuple(dict.fromkeys(value for value in references.values() if isinstance(value, _Work)))
        results: dict[str, EmbeddingVector] = {}
        batches: set[_Batch] = set()
        if works:
            gathered = await asyncio.shield(asyncio.gather(*(work.future for work in works)))
            for work, (vector, batch) in zip(works, gathered, strict=True):
                results[work.text] = vector
                batches.add(batch)
        vectors = tuple(references[text] if isinstance(references[text], tuple) else results[text] for text in texts)
        usages = tuple(batch.usage for batch in batches if batch.usage is not None)
        usage = _sum_usage(usages)
        return EmbeddingResult(vectors=vectors, usage=usage)

    async def _flush_pending(self) -> None:
        if self._coalesce_seconds:
            await asyncio.sleep(self._coalesce_seconds)
        else:
            await asyncio.sleep(0)
        async with self._lock:
            pending, self._pending = self._pending, []
            self._flush_task = None
            for offset in range(0, len(pending), self._max_batch_size):
                batch_work = pending[offset : offset + self._max_batch_size]
                batch = _Batch()
                asyncio.create_task(self._run_batch(batch_work, batch))
            if self._pending and self._flush_task is None:
                self._flush_task = asyncio.create_task(self._flush_pending())

    async def _run_batch(self, works: list[_Work], batch: _Batch) -> None:
        try:
            result = await self._model.embed(tuple(work.text for work in works))
            if len(result.vectors) != len(works):
                raise ValueError("embedding model returned a vector count different from the text count")  # noqa: TRY003
            async with self._lock:
                batch.usage = result.usage
                if self._cache_size:
                    for work, vector in zip(works, result.vectors, strict=True):
                        self._cache[work.text] = vector
                        self._cache.move_to_end(work.text)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                for work, vector in zip(works, result.vectors, strict=True):
                    if self._inflight.get(work.text) is work:
                        del self._inflight[work.text]
                    if not work.future.done():
                        work.future.set_result((vector, batch))
        except BaseException as error:
            async with self._lock:
                for work in works:
                    if self._inflight.get(work.text) is work:
                        del self._inflight[work.text]
                    if not work.future.done():
                        work.future.set_exception(error)


def _sum_usage(usages: tuple[InferenceUsage, ...]) -> InferenceUsage:
    if not usages:
        return InferenceUsage(requests=0, input_tokens=0, output_tokens=0)
    return InferenceUsage(
        requests=sum(usage.requests for usage in usages),
        input_tokens=_sum_optional(value.input_tokens for value in usages),
        output_tokens=_sum_optional(value.output_tokens for value in usages),
    )


def _sum_optional(values: Iterable[int | None]) -> int | None:
    values = tuple(values)
    if all(value is None for value in values):
        return None
    return sum(value or 0 for value in values)


__all__ = ["CoalescingEmbeddingModel"]
