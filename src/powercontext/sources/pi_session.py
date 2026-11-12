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

"""Lossless visible evidence and incremental ingestion for Pi v3 sessions."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import SourceProjection
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
_CHECKPOINT_SCHEMA = "powercontext.pi-session-checkpoint.v1"


class PiSessionInput(BaseModel):
    """Complete Pi v3 JSONL bytes and optional selected branch leaf."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, retaining its original record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured, immutable event evidence from a Pi session."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]
    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED

    @model_validator(mode="after")
    def validate_identity(self) -> PiEventSource:
        identity = pi_event_source_id(self.session_id, self.event_id)
        if self.name != identity or self.source_id != identity:
            raise ValueError("Pi event name and source_id must match its stable identity")  # noqa: TRY003
        return self


def pi_event_source_id(session_id: str, event_id: str) -> str:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session id must be a non-empty string")  # noqa: TRY003
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event id must be a non-empty string")  # noqa: TRY003
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"Pi event identity exceeds {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


class PiEventDefinition:
    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    version = "1"
    source_class = PiEventSource

    def __init__(self) -> None:
        self.projections: tuple[SourceProjection[PiEventSource], ...] = (PiEventTextEvidenceProjection(),)

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=identity,
            source_id=identity,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
            materialization=SourceMaterialization.CAPTURED,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = PiEventDefinition()


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Validate the full archive and return visible events on the chosen branch."""

    if not isinstance(value, PiSessionInput):
        raise ValueError("value must be a PiSessionInput")  # noqa: TRY003
    session_id, nodes, leaf_id = _parse_archive(value)
    chain: list[dict[str, Any]] = []
    current: str | None = leaf_id
    while current is not None:
        node = nodes[current]
        chain.append(node)
        parent_id = node["parentId"]
        current = parent_id if parent_id not in (None, "") else None
    chain.reverse()

    events: list[PiEventInput] = []
    for record in chain:
        visible = _visible_event(record)
        if visible is None:
            continue
        role, body = visible
        content = f"{role}: {body}"
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=record["id"],
                parent_id=record["parentId"] or None,
                event_type=record["type"],
                role=role,
                content=content,
                record=record,
            )
        )
    return tuple(events)


def _parse_archive(value: PiSessionInput) -> tuple[str, dict[str, dict[str, Any]], str | None]:
    try:
        text = value.content.decode("utf-8-sig", errors="strict")
    except (AttributeError, UnicodeDecodeError) as error:
        raise ValueError("Pi session must be valid UTF-8 JSONL") from error  # noqa: TRY003
    records: list[Any] = []
    try:
        for line in text.splitlines():
            if line.strip():
                records.append(json.loads(line, parse_constant=_reject_json_constant))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Pi session contains invalid JSONL") from error  # noqa: TRY003
    if not records or not isinstance(records[0], dict):
        raise ValueError("Pi session must begin with a session header")  # noqa: TRY003
    header = records[0]
    session_id = header.get("id")
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("Pi session header must have type=session and version=3")  # noqa: TRY003
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session id must be a non-empty string")  # noqa: TRY003
    nodes: dict[str, dict[str, Any]] = {}
    for record in records[1:]:
        if not isinstance(record, dict):
            raise ValueError("Pi session records must be JSON objects")  # noqa: TRY003
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Pi event id must be a non-empty string")  # noqa: TRY003
        if not isinstance(record.get("type"), str) or not record["type"]:
            raise ValueError("Pi event type must be a non-empty string")  # noqa: TRY003
        if event_id in nodes:
            raise ValueError(f"duplicate Pi event id: {event_id!r}")  # noqa: TRY003
        if "parentId" not in record:
            raise ValueError("Pi event is missing parentId")  # noqa: TRY003
        parent_id = record["parentId"]
        if parent_id not in (None, ""):
            if not isinstance(parent_id, str) or parent_id not in nodes:
                raise ValueError(f"Pi event parent must refer to a preceding event: {parent_id!r}")  # noqa: TRY003
        nodes[event_id] = record
        pi_event_source_id(session_id, event_id)
    if value.leaf_id is not None:
        if not isinstance(value.leaf_id, str) or value.leaf_id not in nodes:
            raise ValueError("selected Pi leaf does not exist")  # noqa: TRY003
        leaf_id: str | None = value.leaf_id
    else:
        leaf_id = next(reversed(nodes), None)
    return session_id, nodes, leaf_id


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")  # noqa: TRY003


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    pieces = [block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)]
    return "\n".join(pieces)


def _visible_event(record: dict[str, Any]) -> tuple[str, str] | None:
    event_type = record.get("type")
    if event_type in {"compaction", "branch_summary"}:
        role = event_type
        body = record.get("summary")
        return (role, body) if isinstance(body, str) and body else None
    if event_type == "custom_message":
        body = _text_content(record.get("content"))
        return ("custom", body) if body else None

    message = record.get("message")
    if not isinstance(message, dict):
        message = {}
    role = message.get("role")
    if role == "bashExecution" or event_type == "bashExecution":
        command = message.get("command", record.get("command"))
        output = message.get("output", record.get("output"))
        if not isinstance(command, str):
            return None
        body = f"$ {command}"
        if isinstance(output, str) and output:
            body += f"\n{output}"
        return "bashExecution", body
    if role == "toolResult":
        text = _text_content(message.get("content"))
        tool_name = message.get("toolName", "")
        call_id = message.get("toolCallId", "")
        body = f"[tool_result {tool_name} {call_id}] {text}".rstrip()
        return ("toolResult", body) if text else None
    if not isinstance(role, str) or not role:
        return None
    content = message.get("content")
    if isinstance(content, str):
        return (role, content) if content else None
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block.get("type") == "toolCall":
            name = block.get("name", "")
            call_id = block.get("id", "")
            arguments = json.dumps(
                block.get("arguments"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            parts.append(f"[tool_call {name} {call_id}] {arguments}")
    body = "\n".join(part for part in parts if part)
    return (role, body) if body else None


class PiSessionConnector:
    """Ingest newly selected branch events with a durable lifecycle checkpoint."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str, leaf_id: str | None = None) -> None:
        self.path = path
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        from pathlib import Path

        raw = Path(self.path).read_bytes()
        value = PiSessionInput(content=raw, leaf_id=self.leaf_id)
        session_id, nodes, _ = _parse_archive(value)
        selected = parse_pi_session(value)
        checkpoint = _decode_checkpoint(session.checkpoint)
        if checkpoint is not None and checkpoint[0] != session_id:
            raise ValueError("Pi session id changed since the previous import")  # noqa: TRY003
        imported = {} if checkpoint is None else checkpoint[1]
        for event_id, expected_digest in imported.items():
            record = nodes.get(event_id)
            if record is None or _record_digest(record) != expected_digest:
                raise ValueError(f"an imported Pi event was removed or changed: {event_id!r}")  # noqa: TRY003

        updated = dict(imported)
        for event in selected:
            digest = _record_digest(event.record)
            if event.event_id in imported:
                continue
            await session.submit(
                pi_event_source_id(event.session_id, event.event_id),
                PI_EVENT_SOURCE_NAME,
                event,
            )
            updated[event.event_id] = digest
        proposed: JsonValue = {
            "schema": _CHECKPOINT_SCHEMA,
            "session_id": session_id,
            "events": updated,
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=proposed)


def _record_digest(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _decode_checkpoint(value: JsonValue | None) -> tuple[str, dict[str, str]] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"schema", "session_id", "events"}:
        raise ValueError("invalid Pi connector checkpoint")  # noqa: TRY003
    session_id, events = value["session_id"], value["events"]
    if value["schema"] != _CHECKPOINT_SCHEMA or not isinstance(session_id, str) or not isinstance(events, dict):
        raise ValueError("invalid Pi connector checkpoint")  # noqa: TRY003
    if any(not isinstance(key, str) or not isinstance(digest, str) or len(digest) != 64 for key, digest in events.items()):
        raise ValueError("invalid Pi connector checkpoint events")  # noqa: TRY003
    return session_id, events


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]
