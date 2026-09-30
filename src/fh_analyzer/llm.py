"""LLM access behind a tiny interface, so the pipeline is testable without an API key.

Structured output uses *forced tool use*: the Pydantic model's JSON schema becomes the
tool's input schema and `tool_choice` forces the model to call it. The returned
arguments are then validated by Pydantic; on a validation error we retry once with the
error message fed back to the model.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    retries: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, u) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += getattr(u, "input_tokens", 0) or 0
            self.output_tokens += getattr(u, "output_tokens", 0) or 0
            self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
            self.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class LLM(Protocol):
    usage: Usage
    model: str

    def extract(self, system: str, user: str, schema: type[T]) -> T: ...


_PARAM = re.compile(r'<parameter name="(\w+)">(.*?)(?:</parameter>|$)', re.S)
_JUNK = re.compile(r"\s*(</?parameter[^>]*>|</?record_findings>|</?invoke[^>]*>).*", re.S)


def repair_tool_input(data: dict) -> dict:
    """Recover from a known model failure mode where remaining tool arguments are
    emitted as XML-ish text *inside* a string field, e.g.
    {"summary": "...</parameter><parameter name=\"reasons\">[]"}.
    Pull those arguments back out into the dict and strip the junk from the string.
    Only fills keys that are missing; never overwrites real values."""
    if not isinstance(data, dict):
        return data
    out = dict(data)
    for key, val in data.items():
        if not isinstance(val, str) or "<parameter" not in val and "</record_findings>" not in val:
            continue
        for name, raw in _PARAM.findall(val):
            if name in out:
                continue
            raw = _JUNK.sub("", raw).strip()
            try:
                out[name] = json.loads(raw)
            except ValueError:
                out[name] = raw
        out[key] = _JUNK.sub("", val).strip()
    return out


def _require_all(schema: dict) -> dict:
    """Mark every property as required in the schema the MODEL sees.

    Pydantic leaves fields with defaults (e.g. `rejections: list = []`) out of
    `required`. Models treat non-required fields as optional and, in practice, skip
    them - producing a good summary but empty lists. Validation on our side stays
    lenient (defaults still apply); only the tool schema is tightened.
    """
    schema = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


def _strict_schema(schema: dict) -> dict:
    """Prepare a JSON schema for strict tool use: objects must declare
    additionalProperties: false."""
    schema = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node.setdefault("additionalProperties", False)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_tokens: int = 8000,
                 strict: bool = True, cost_log=None) -> None:
        import anthropic

        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=4)
        self.model = model
        self.max_tokens = max_tokens
        # Strict tool use makes the API enforce the schema (structured outputs). If the
        # model or schema isn't supported we fall back to validate-and-retry.
        self.strict = strict
        # Forced tool use (tool_choice "tool"). Some models reject it (seen with
        # claude-opus-5-5: 'tool_choice: type "tool" and "any" are not supported'); for
        # those we switch to tool_choice "auto" plus an instruction to call the tool.
        self.force_tool = True
        self.usage = Usage()
        # costs.CostLog: one record per API call (tokens, $, latency, stage, document).
        self.cost_log = cost_log

    def _tool(self, schema: type[BaseModel]) -> dict:
        tool = {
            "name": "record_findings",
            "description": f"Record the extracted {schema.__name__} for this document.",
            "input_schema": _require_all(schema.model_json_schema()),
        }
        if self.strict:
            tool["input_schema"] = _strict_schema(tool["input_schema"])
            tool["strict"] = True
        return tool

    AUTO_TOOL_NOTE = ("\n\nRespond ONLY by calling the record_findings tool, exactly once. "
                      "Do not answer in plain text.")

    def _kwargs(self, system: str, messages: list[dict]) -> dict:
        if not self.force_tool:
            system += self.AUTO_TOOL_NOTE
        return dict(
            model=self.model,
            max_tokens=self.max_tokens,
            # The system prompt is identical across documents of a type -> cache it.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tool_choice=({"type": "tool", "name": "record_findings"} if self.force_tool
                         else {"type": "auto"}),
            messages=messages,
        )

    def _create(self, system: str, schema: type[BaseModel], messages: list[dict]):
        for _ in range(3):          # at most: forced+strict -> auto -> non-strict
            try:
                return self._client.messages.create(tools=[self._tool(schema)],
                                                    **self._kwargs(system, messages))
            except self._anthropic.BadRequestError as e:
                if self.force_tool and "tool_choice" in str(e):
                    log.warning("%s rejects forced tool use; using tool_choice=auto.",
                                self.model)
                    self.force_tool = False
                elif self.strict:
                    log.warning("Strict tool use rejected (%s); falling back to non-strict.", e)
                    self.strict = False
                else:
                    raise
        raise AssertionError("unreachable")

    def _record(self, usage, t0: float, attempt: int, *, ok: bool,
                error: str | None = None) -> None:
        if self.cost_log is not None:
            self.cost_log.record(model=self.model, usage=usage, retry=attempt > 0, ok=ok,
                                 error=error,
                                 latency_ms=int((time.perf_counter() - t0) * 1000))

    def extract(self, system: str, user: str, schema: type[T]) -> T:
        messages: list[dict] = [{"role": "user", "content": user}]
        for attempt in range(2):
            if self.cost_log is not None:
                self.cost_log.check_budget()
            t0 = time.perf_counter()
            try:
                resp = self._create(system, schema, messages)
            except Exception as e:
                self._record(None, t0, attempt, ok=False, error=f"{type(e).__name__}: {e}")
                raise
            self.usage.add(resp.usage)
            block = next((b for b in resp.content if b.type == "tool_use"), None)
            if block is None:           # only possible with tool_choice=auto
                self._record(resp.usage, t0, attempt, ok=False, error="NoToolCall")
                if attempt == 1:
                    raise ValueError(f"{self.model} answered without calling the tool")
                self.usage.retries += 1
                messages += [{"role": "assistant", "content": resp.content},
                             {"role": "user", "content": "Call the record_findings tool now "
                                                         "with your findings."}]
                continue
            try:
                out = schema.model_validate(repair_tool_input(block.input))
                self._record(resp.usage, t0, attempt, ok=True)
                return out
            except ValidationError as e:
                self._record(resp.usage, t0, attempt, ok=False,
                             error=f"ValidationError: {str(e)[:300]}")
                if attempt == 1:
                    raise
                self.usage.retries += 1
                log.warning("Schema validation failed, retrying: %s", e)
                messages += [
                    {"role": "assistant", "content": resp.content},
                    {"role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": block.id, "is_error": True,
                        "content": f"Output failed validation:\n{e}\nCall the tool again "
                                   "with corrected input. Pass every field as a separate "
                                   "JSON argument; do not embed XML in string values.",
                    }]},
                ]
        raise AssertionError("unreachable")
