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

"""Canonical context packs and prefix-reuse deltas."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(ValueError):
    """Invalid pack, delta, or chunk boundary."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete canonical pack of UTF-8 byte chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        normalized = _validate_items(items)
        digest = _pack_digest(normalized)
        byte_count = 0
        while True:
            encoded = _pack_wire_bytes(normalized, digest, byte_count)
            if len(encoded) == byte_count:
                return cls(normalized, digest, byte_count)
            byte_count = len(encoded)

    @classmethod
    def from_content(cls, content: str | None, *, chunk_bytes: int = 512) -> ContextPack:
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ContextPackError("content must be text or None")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        try:
            encoded = content.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ContextPackError("content must be valid UTF-8 text") from error
        if not encoded:
            return cls.build(())

        preferred = _preferred_boundaries(encoded)
        chunks: list[bytes] = []
        start = 0
        while start < len(encoded):
            end = min(start + chunk_bytes, len(encoded))
            while end > start and end < len(encoded) and encoded[end] & 0xC0 == 0x80:
                end -= 1
            candidates = [boundary for boundary in preferred if start < boundary <= end]
            if candidates:
                end = candidates[-1]
            if end == start:
                raise ContextPackError("chunk_bytes cannot fit one UTF-8 code point")
            chunk = encoded[start:end]
            try:
                chunk.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ContextPackError("chunk boundary splits a UTF-8 code point") from error
            chunks.append(chunk)
            start = end
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        return _pack_wire_bytes(self.items, self.digest, self.byte_count)

    @classmethod
    def from_bytes(cls, value: bytes) -> ContextPack:
        payload = _decode_object(value)
        _require_fields(payload, {"schema", "items", "digest", "byte_count"})
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("unsupported context pack schema")
        items = _decode_items(payload["items"])
        pack = cls.build(items)
        if payload["digest"] != pack.digest:
            raise ContextPackError("context pack digest does not match its items")
        byte_count = payload["byte_count"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count != len(value):
            raise ContextPackError("context pack byte_count does not match its wire size")
        if pack.byte_count != byte_count or pack.to_bytes() != value:
            raise ContextPackError("context pack is not canonically encoded")
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A target suffix and digest tied to one exact base pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta endpoints must be ContextPack values")
        base = _validate_pack(base)
        target = _validate_pack(target)
        prefix = 0
        for base_item, target_item in zip(base.items, target.items):
            if base_item != target_item:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        _validate_digest(self.base_digest)
        _validate_digest(self.target_digest)
        _validate_prefix(self.prefix_items)
        appended = _validate_items(self.appended)
        payload = {
            "schema": DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [_b64encode(item) for item in appended],
        }
        return _canonical_json(payload)

    @classmethod
    def from_bytes(cls, value: bytes) -> ContextPackDelta:
        payload = _decode_object(value)
        _require_fields(payload, {"schema", "base_digest", "target_digest", "prefix_items", "appended"})
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("unsupported context pack delta schema")
        base_digest = _validate_digest(payload["base_digest"])
        target_digest = _validate_digest(payload["target_digest"])
        prefix_items = _validate_prefix(payload["prefix_items"])
        appended = _decode_items(payload["appended"])
        return cls(base_digest, target_digest, prefix_items, appended)


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Validate a delta against its base and reconstruct the full target pack."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("base and delta must be context pack values")
    base = _validate_pack(base)
    _validate_digest(delta.base_digest)
    _validate_digest(delta.target_digest)
    prefix_items = _validate_prefix(delta.prefix_items)
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the supplied pack")
    if prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base pack item count")
    target = ContextPack.build(base.items[:prefix_items] + _validate_items(delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest does not match the reconstructed pack")
    return target


def _pack_digest(items: tuple[bytes, ...]) -> str:
    payload = {"schema": PACK_SCHEMA, "items": [_b64encode(item) for item in items]}
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


def _pack_wire_bytes(items: tuple[bytes, ...], digest: str, byte_count: int) -> bytes:
    return _canonical_json(
        {
            "schema": PACK_SCHEMA,
            "items": [_b64encode(item) for item in items],
            "digest": digest,
            "byte_count": byte_count,
        }
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_items(items: object) -> tuple[bytes, ...]:
    if not isinstance(items, tuple):
        raise ContextPackError("items must be a tuple of bytes")
    for item in items:
        if not isinstance(item, bytes):
            raise ContextPackError("each item must be bytes")
        try:
            item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ContextPackError("each item must contain valid UTF-8") from error
    return items


def _validate_pack(pack: ContextPack) -> ContextPack:
    canonical = ContextPack.build(_validate_items(pack.items))
    if pack.digest != canonical.digest or pack.byte_count != canonical.byte_count:
        raise ContextPackError("context pack metadata does not match its items")
    return canonical


def _decode_object(value: bytes) -> dict[str, Any]:
    if not isinstance(value, bytes):
        raise ContextPackError("wire value must be bytes")
    try:
        decoded = json.loads(value.decode("utf-8", errors="strict"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateFieldError) as error:
        raise ContextPackError("wire value must be valid UTF-8 JSON without duplicate keys") from error
    if not isinstance(decoded, dict):
        raise ContextPackError("wire value must be a JSON object")
    return decoded


class _DuplicateFieldError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateFieldError(key)
        result[key] = value
    return result


def _require_fields(payload: dict[str, Any], expected: set[str]) -> None:
    if set(payload) != expected:
        raise ContextPackError("wire object has missing or extra fields")


def _decode_items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("wire chunks must be an array")
    decoded: list[bytes] = []
    for item in value:
        if not isinstance(item, str):
            raise ContextPackError("wire chunks must be Base64 strings")
        try:
            chunk = base64.b64decode(item.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as error:
            raise ContextPackError("wire chunk is not valid Base64") from error
        if _b64encode(chunk) != item:
            raise ContextPackError("wire chunk is not canonically Base64 encoded")
        try:
            chunk.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ContextPackError("wire chunk is not valid UTF-8") from error
        decoded.append(chunk)
    return tuple(decoded)


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _validate_digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise ContextPackError("digest must use sha256:<hex>")
    if any(character not in "0123456789abcdef" for character in value[7:]):
        raise ContextPackError("digest must contain lowercase hexadecimal")
    return value


def _validate_prefix(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContextPackError("prefix_items must be a non-negative integer")
    return value


def _preferred_boundaries(encoded: bytes) -> list[int]:
    boundaries: set[int] = set()
    for marker in (b'"items":[', b"},{", b"\n"):
        position = 0
        while (position := encoded.find(marker, position)) >= 0:
            boundaries.add(position + (len(marker) - 1 if marker == b"},{" else len(marker)))
            position += len(marker)
    return sorted(boundaries)


__all__ = ["ContextPack", "ContextPackDelta", "ContextPackError", "apply_context_pack_delta"]
