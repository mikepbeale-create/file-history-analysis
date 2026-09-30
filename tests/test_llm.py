from types import SimpleNamespace as NS

from fh_analyzer.llm import AnthropicLLM
from fh_analyzer.models import InterviewExtraction


class _Msgs:
    def __init__(self, outputs):
        self.outputs, self.kwargs = list(outputs), []

    def create(self, **kw):
        self.kwargs.append(kw)
        block = NS(type="tool_use", id="t1", input=self.outputs.pop(0))
        return NS(content=[block], usage=NS(input_tokens=10, output_tokens=5,
                                            cache_read_input_tokens=0))


def _llm(outputs):
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.model, llm.max_tokens, llm.strict, llm.cost_log = "m", 100, True, None
    llm.force_tool = True
    import anthropic
    llm._anthropic = anthropic
    from fh_analyzer.llm import Usage
    llm.usage = Usage()
    llm._client = NS(messages=_Msgs(outputs))
    return llm


GOOD = {"agreement_reached": True, "summary": "s",
        "summary_citation": {"doc_id": "D", "page": 1, "quote": "q"}}


def test_forced_tool_use_and_schema():
    llm = _llm([GOOD])
    out = llm.extract("sys", "user", InterviewExtraction)
    assert out.agreement_reached is True
    kw = llm._client.messages.kwargs[0]
    assert kw["tool_choice"] == {"type": "tool", "name": "record_findings"}
    assert kw["tools"][0]["input_schema"]["title"] == "InterviewExtraction"
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_retries_once_on_validation_error():
    llm = _llm([{"summary": "missing fields"}, GOOD])
    assert llm.extract("sys", "user", InterviewExtraction).summary == "s"
    assert llm.usage.retries == 1 and llm.usage.calls == 2
    retry_msgs = llm._client.messages.kwargs[1]["messages"]
    assert retry_msgs[-1]["content"][0]["is_error"] is True


def test_kwargs_match_installed_sdk():
    """Guard against SDK signature changes (e.g. anthropic 1.x dropped `temperature`)."""
    import inspect

    import anthropic

    allowed = set(inspect.signature(
        anthropic.Anthropic(api_key="x").messages.create).parameters)
    llm = _llm([GOOD])
    llm.extract("sys", "user", InterviewExtraction)
    assert set(llm._client.messages.kwargs[0]) <= allowed


def test_strict_tool_schema():
    llm = _llm([GOOD])
    llm.extract("sys", "user", InterviewExtraction)
    tool = llm._client.messages.kwargs[0]["tools"][0]
    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False
    assert tool["input_schema"]["$defs"]["Citation"]["additionalProperties"] is False


def test_falls_back_when_strict_rejected():
    import anthropic
    import httpx

    class _Picky(_Msgs):
        def create(self, **kw):
            if kw["tools"][0].get("strict"):
                self.kwargs.append(kw)
                req = httpx.Request("POST", "https://x")
                raise anthropic.BadRequestError(
                    "strict not supported", response=httpx.Response(400, request=req),
                    body=None)
            return super().create(**kw)

    llm = _llm([GOOD])
    llm._client = NS(messages=_Picky([GOOD]))
    assert llm.extract("sys", "user", InterviewExtraction).summary == "s"
    assert llm.strict is False


def test_repair_xml_leaked_into_string():
    from fh_analyzer.llm import repair_tool_input

    bad = {"summary": 'This document is a notice.</parameter>\n<parameter name="reasons">[]'
                      '</parameter>\n<parameter name="summary_citation">{"doc_id": "D", '
                      '"page": 1, "quote": "q"}</record_findings>\n\n'}
    fixed = repair_tool_input(bad)
    assert fixed["summary"] == "This document is a notice."
    assert fixed["reasons"] == []
    assert fixed["summary_citation"]["page"] == 1
    assert repair_tool_input({"summary": "clean"}) == {"summary": "clean"}


def test_model_sees_every_field_as_required():
    from fh_analyzer.models import OfficeActionExtraction

    llm = _llm([GOOD])
    tool = llm._tool(OfficeActionExtraction)
    sch = tool["input_schema"]
    assert set(sch["required"]) == set(sch["properties"])
    assert "rejections" in sch["required"]
    # ...while our own validation stays lenient.
    assert OfficeActionExtraction.model_validate(
        {"action_type": "other", "summary": "s"}).rejections == []


def test_falls_back_to_auto_tool_choice_when_forced_tool_rejected():
    """claude-opus-5-5 answered 400 'tool_choice: type "tool" and "any" are not supported'
    for every call in the model sweep. The client should switch to tool_choice=auto, tell
    the model to call the tool, and re-ask once if it answers in plain text."""
    import anthropic
    import httpx

    class _NoForced(_Msgs):
        def __init__(self, outputs):
            super().__init__(outputs)
            self.text_first = True

        def create(self, **kw):
            self.kwargs.append(kw)
            if kw["tool_choice"]["type"] == "tool":
                req = httpx.Request("POST", "https://x")
                raise anthropic.BadRequestError(
                    'tool_choice: type "tool" and "any" are not supported for this model.',
                    response=httpx.Response(400, request=req), body=None)
            if self.text_first:                     # first auto reply is prose
                self.text_first = False
                return NS(content=[NS(type="text", text="Here is my analysis...")],
                          usage=NS(input_tokens=10, output_tokens=5))
            block = NS(type="tool_use", id="t1", input=self.outputs.pop(0))
            return NS(content=[block], usage=NS(input_tokens=10, output_tokens=5))

    llm = _llm([GOOD])
    llm._client = NS(messages=_NoForced([GOOD]))
    assert llm.extract("sys", "user", InterviewExtraction).summary == "s"
    assert llm.force_tool is False and llm.strict is True        # strict kept
    auto = [k for k in llm._client.messages.kwargs if k["tool_choice"]["type"] == "auto"]
    assert "ONLY by calling the record_findings tool" in auto[0]["system"][0]["text"]
    assert auto[-1]["messages"][-1]["content"].startswith("Call the record_findings tool")
    assert llm.usage.retries == 1
