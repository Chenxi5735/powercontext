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

"""Offline CSV parsing into stable, row-level Source evidence."""

from __future__ import annotations

import csv
import io
import json
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import SourceProjection
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"


class CSVRowInput(BaseModel):
    """One parsed CSV record with its exact document and schema identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_row(self) -> CSVRowInput:
        if not self.document_id or self.document_id != self.document_id.strip():
            raise ValueError("document_id must be non-empty and have no surrounding whitespace")  # noqa: TRY003
        if not self.primary_key:
            raise ValueError("primary_key must be non-empty")  # noqa: TRY003
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be non-empty and unique")  # noqa: TRY003
        if any(not column.strip() for column in self.columns):
            raise ValueError("columns must be non-blank")  # noqa: TRY003
        if len(self.columns) != len(self.values):
            raise ValueError("columns and values must have equal lengths")  # noqa: TRY003
        try:
            key_index = self.columns.index(self.primary_key)
        except ValueError as error:
            raise ValueError("primary_key must exactly match a column") from error  # noqa: TRY003
        if not self.primary_key_value or not self.primary_key_value.strip():
            raise ValueError("primary_key_value must not be blank")  # noqa: TRY003
        if self.values[key_index] != self.primary_key_value:
            raise ValueError("primary_key_value must match the row value")  # noqa: TRY003
        csv_row_source_id(self.document_id, self.primary_key_value)
        return self


class CSVDocumentInput(BaseModel):
    """Complete offline CSV content and its document-level identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    primary_key: str
    content: bytes


class CSVRowSource(Source):
    """Captured raw fields for one stable document/key pair."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]
    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED

    @model_validator(mode="after")
    def validate_row_identity(self) -> CSVRowSource:
        if not self.document_id or self.document_id != self.document_id.strip():
            raise ValueError("document_id must be non-empty and have no surrounding whitespace")  # noqa: TRY003
        if not self.primary_key or self.primary_key not in self.columns:
            raise ValueError("primary_key must exactly match a column")  # noqa: TRY003
        if not self.primary_key_value or not self.primary_key_value.strip():
            raise ValueError("primary_key_value must not be blank")  # noqa: TRY003
        if len(self.columns) != len(self.values):
            raise ValueError("columns and values must have equal lengths")  # noqa: TRY003
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be unique")  # noqa: TRY003
        if not self.columns or any(not column.strip() for column in self.columns):
            raise ValueError("columns must be non-empty and non-blank")  # noqa: TRY003
        key_index = self.columns.index(self.primary_key)
        if self.values[key_index] != self.primary_key_value:
            raise ValueError("primary_key_value must match the row value")  # noqa: TRY003
        expected = csv_row_source_id(self.document_id, self.primary_key_value)
        if self.name != expected or self.source_id != expected:
            raise ValueError("name and source_id must match the CSV row identity")  # noqa: TRY003
        return self

    @property
    def row(self) -> dict[str, str]:
        return dict(zip(self.columns, self.values, strict=True))


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate a complete UTF-8 CSV document atomically."""

    if not isinstance(value, CSVDocumentInput):
        raise ValueError("value must be a CSVDocumentInput")  # noqa: TRY003
    content = value.content
    document_id = value.document_id
    primary_key = value.primary_key
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty and have no surrounding whitespace")  # noqa: TRY003
    if not isinstance(primary_key, str) or not primary_key:
        raise ValueError("primary_key must be non-empty")  # noqa: TRY003
    try:
        text = content.decode("utf-8-sig", errors="strict")
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=",", quotechar='"', strict=True)
        header = next(reader, None)
        if header is None or not header:
            raise ValueError("CSV document must contain a non-empty header")  # noqa: TRY003
        if any(not column.strip() for column in header):
            raise ValueError("CSV headers must be non-empty and non-blank")  # noqa: TRY003
        if len(set(header)) != len(header):
            raise ValueError("CSV headers must be unique")  # noqa: TRY003
        try:
            key_index = header.index(primary_key)
        except ValueError as error:
            raise ValueError("primary_key must exactly match a CSV header") from error  # noqa: TRY003

        rows: list[CSVRowInput] = []
        seen: set[str] = set()
        for values in reader:
            if not values:
                continue
            if len(values) != len(header):
                raise ValueError("CSV record width must match the header")  # noqa: TRY003
            key_value = values[key_index]
            if not key_value or not key_value.strip():
                raise ValueError("CSV primary key values must not be blank")  # noqa: TRY003
            if key_value in seen:
                raise ValueError("CSV primary key values must be unique")  # noqa: TRY003
            seen.add(key_value)
            csv_row_source_id(document_id, key_value)
            rows.append(
                CSVRowInput(
                    document_id=document_id,
                    primary_key=primary_key,
                    primary_key_value=key_value,
                    columns=tuple(header),
                    values=tuple(values),
                )
            )
        return tuple(rows)
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("CSV document must be valid UTF-8 and well-formed CSV") from error  # noqa: TRY003


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the bounded, delimiter-safe identity for one CSV row."""

    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty and have no surrounding whitespace")  # noqa: TRY003
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must not be blank")  # noqa: TRY003
    identity = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row source identity exceeds {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


class CSVRowDefinition:
    """Source Definition for resolving and projecting captured CSV rows."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    version = "1"
    source_class = CSVRowSource

    def __init__(self) -> None:
        self.projections: tuple[SourceProjection[CSVRowSource], ...] = (CSVRowTextEvidenceProjection(),)

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Project a row as lossless ordered JSON text and metadata."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = source.row
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_DEFINITION = CSVRowDefinition()
CSV_ROW_SOURCE_DEFINITION = CSV_ROW_DEFINITION

__all__ = [
    "CSV_ROW_DEFINITION",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowDefinition",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
