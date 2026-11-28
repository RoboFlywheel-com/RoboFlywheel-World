# SPDX-License-Identifier: Apache-2.0
"""The two request shapes the client speaks, and what happens when the wrong one is used.

The wire is the one place in this client where a mistake is **silent**. Everything else fails
loudly: a missing credential raises, an HTTP error raises, a non-JSON body raises. But a
chat-completions body posted to a Responses endpoint can be accepted and answered from the text
alone -- which yields a critique that scored seven criteria having never received an image, and
that critique satisfies the gate. So the tests here are about what must *not* be quietly tolerated:

* a payload in the other wire's shape must raise, not read as empty text;
* a response truncated at the token ceiling must name that cause, because "no output" and
  "max_output_tokens" send the reader to two different fixes;
* the wire name must be one of the two, and a base_url that already spells its own path must not
  have the path appended twice.

The helpers under test are private and are exercised directly. There is no way to observe a request
body through ``call_vision`` without a network call, and a test that needed a live endpoint would
not be run -- which is exactly the sort of gate this repo refuses to keep.
"""

from __future__ import annotations

import pytest

from rfw_w.scenes.agents.llm_client import (
    CredentialsMissing,
    LLMConfig,
    LLMError,
    WIRES,
    _read_response,
    _request_body,
)

MODEL = "gpt-5.6-sol"
DATA_URL = "data:image/png;base64,AAAA"


def config(**overrides) -> LLMConfig:
    fields = {
        "base_url": "https://example.invalid/protocol/openai/v1",
        "model": MODEL,
        "api_key_env": "TEST_KEY",
    }
    fields.update(overrides)
    return LLMConfig(**fields)


def chat_payload(text: str = "hello") -> dict:
    return {
        "model": MODEL,
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 42},
    }


def responses_payload(*, blocks: list[list[tuple[str, str]]], status: str = "completed",
                      incomplete: dict | None = None) -> dict:
    """A /responses body: `output` is a list of items, each with typed content parts."""
    return {
        "model": MODEL,
        "status": status,
        "incomplete_details": incomplete,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": kind, "text": text} for kind, text in block],
            }
            for block in blocks
        ],
        "usage": {"total_tokens": 1176},
    }


class TestWireValidation:
    def test_an_unknown_wire_is_refused(self) -> None:
        with pytest.raises(LLMError, match="wire must be one of"):
            config(wire="openai-ish")

    def test_the_two_known_wires_are_both_accepted(self) -> None:
        assert set(WIRES) == {"chat_completions", "responses"}
        for wire in WIRES:
            assert config(wire=wire).wire == wire

    def test_the_default_is_the_shape_the_existing_deployment_speaks(self) -> None:
        """No environment and no argument must reproduce the DashScope behaviour exactly."""
        assert config().wire == "chat_completions"


class TestEndpoint:
    def test_each_wire_appends_its_own_path(self) -> None:
        assert config().endpoint.endswith("/v1/chat/completions")
        assert config(wire="responses").endpoint.endswith("/v1/responses")

    def test_a_base_url_that_already_names_its_path_is_not_appended_twice(self) -> None:
        """Providers hand out both ".../v1" and ".../v1/responses"; the second is used as given.

        Appending blindly turns it into ".../v1/responses/responses", and a 404 reads like a wrong
        model name rather than a wrong URL.
        """
        full = config(base_url="https://example.invalid/protocol/openai/v1/responses",
                      wire="responses")
        assert full.endpoint == "https://example.invalid/protocol/openai/v1/responses"

    def test_a_trailing_slash_does_not_double_up(self) -> None:
        assert config(base_url="https://example.invalid/v1/").endpoint == (
            "https://example.invalid/v1/chat/completions"
        )


class TestRequestBody:
    def test_the_responses_body_carries_instructions_and_typed_input(self) -> None:
        body = _request_body(
            config(wire="responses"), system_prompt="SYS", user_prompt="USER", images=[DATA_URL]
        )
        assert body["instructions"] == "SYS"
        assert body["max_output_tokens"] == config().max_tokens
        assert body["input"][0]["role"] == "user"
        assert body["input"][0]["content"][0] == {"type": "input_text", "text": "USER"}
        assert body["input"][0]["content"][1] == {"type": "input_image", "image_url": DATA_URL}

    def test_the_responses_body_carries_none_of_the_chat_keys(self) -> None:
        """The negative half, and the one that matters: a stray `messages` key is the mistake."""
        body = _request_body(
            config(wire="responses"), system_prompt="SYS", user_prompt="USER", images=[]
        )
        assert "messages" not in body
        assert "max_tokens" not in body

    def test_the_chat_body_is_unchanged(self) -> None:
        body = _request_body(
            config(), system_prompt="SYS", user_prompt="USER", images=[DATA_URL]
        )
        assert body["max_tokens"] == config().max_tokens
        assert body["messages"][0] == {"role": "system", "content": "SYS"}
        assert body["messages"][1]["content"][1] == {
            "type": "image_url", "image_url": {"url": DATA_URL}
        }
        assert "input" not in body and "instructions" not in body

    def test_an_image_is_a_bare_url_on_one_wire_and_a_wrapped_dict_on_the_other(self) -> None:
        """A data URL wrapped in `{"url": ...}` on the responses wire is a 400, not a silent miss."""
        responses = _request_body(
            config(wire="responses"), system_prompt="S", user_prompt="U", images=[DATA_URL]
        )
        chat = _request_body(config(), system_prompt="S", user_prompt="U", images=[DATA_URL])
        assert responses["input"][0]["content"][1]["image_url"] == DATA_URL
        assert chat["messages"][1]["content"][1]["image_url"] == {"url": DATA_URL}


class TestResponseReading:
    def test_output_text_parts_are_concatenated(self) -> None:
        payload = responses_payload(blocks=[[("output_text", '{"a":'), ("output_text", "1}")]])
        assert _read_response(config(wire="responses"), payload).text == '{"a":1}'

    def test_non_text_parts_are_skipped(self) -> None:
        """A reasoning item and a refusal carry no text and must not become the critique."""
        payload = responses_payload(
            blocks=[[("reasoning_text", "thinking..."), ("output_text", '{"ok":true}')]]
        )
        assert _read_response(config(wire="responses"), payload).text == '{"ok":true}'

    def test_usage_and_status_come_back_as_the_common_fields(self) -> None:
        response = _read_response(config(wire="responses"), responses_payload(
            blocks=[[("output_text", "{}")]]
        ))
        assert response.tokens_used == 1176
        assert response.model_id == MODEL
        assert response.finish_reason == "completed"

    def test_an_empty_responses_body_raises_rather_than_returning_empty_text(self) -> None:
        """Empty text would reach `parse_critique` as a malformed reply and be recorded as the
        model's fault. It is the wire's fault, and it must say so here."""
        with pytest.raises(LLMError, match="carried no output_text"):
            _read_response(config(wire="responses"), responses_payload(blocks=[]))

    def test_a_token_ceiling_is_named_as_the_cause(self) -> None:
        payload = responses_payload(
            blocks=[], status="incomplete", incomplete={"reason": "max_output_tokens"}
        )
        with pytest.raises(LLMError, match="max_output_tokens"):
            _read_response(config(wire="responses"), payload)

    def test_provider_reason_is_not_echoed_into_errors(self) -> None:
        marker = "synthetic-sensitive-marker"
        payload = responses_payload(blocks=[], status=marker,
                                    incomplete={"reason": marker})
        with pytest.raises(LLMError) as caught:
            _read_response(config(wire="responses"), payload)
        assert marker not in str(caught.value)

    def test_a_truncated_reply_keeps_the_reason_it_was_truncated(self) -> None:
        """Text that stops mid-object is UNPARSEABLE downstream, which reads as a bad model.

        The ceiling is what actually happened, and `finish_reason` is the only place that survives
        to say so -- the first comparison run between two base models was diagnosed entirely from
        this field, after one arm returned 329 characters of a JSON object it had not finished.
        """
        response = _read_response(config(wire="responses"), responses_payload(
            blocks=[[("output_text", '{"scores": {"prompt_adherence": 3')]],
            status="incomplete", incomplete={"reason": "max_output_tokens"},
        ))
        assert response.finish_reason == "max_output_tokens"
        assert response.text.endswith("3")

    def test_a_chat_body_read_as_the_other_wire_raises(self) -> None:
        """The silent-corruption case this whole module exists to prevent.

        A chat-completions body has no `output` key, so a reader that shrugged would hand back an
        empty string, and an empty critique is one that reports nothing wrong.
        """
        with pytest.raises(LLMError, match="carried no output_text"):
            _read_response(config(wire="responses"), chat_payload("a real critique"))

    def test_a_responses_body_read_as_chat_still_raises(self) -> None:
        """The same mistake in the other direction already failed loudly, and must keep doing so."""
        with pytest.raises(LLMError, match="no choices"):
            _read_response(config(), responses_payload(blocks=[[("output_text", "{}")]]))

    def test_a_null_usage_block_does_not_become_a_token_count(self) -> None:
        payload = responses_payload(blocks=[[("output_text", "{}")]])
        payload["usage"] = None
        assert _read_response(config(wire="responses"), payload).tokens_used is None


class TestCredential:
    def test_the_config_records_the_variable_name_and_not_the_value(self, monkeypatch) -> None:
        monkeypatch.setenv("TEST_KEY", "sk-secret-value")
        rendered = repr(config().to_dict()) + repr(config())
        assert "TEST_KEY" in rendered
        assert "sk-secret-value" not in rendered

    def test_presence_is_reported_without_reading_the_key(self, monkeypatch) -> None:
        monkeypatch.delenv("TEST_KEY", raising=False)
        assert config().credential_present() is False
        monkeypatch.setenv("TEST_KEY", "x")
        assert config().credential_present() is True

    def test_the_missing_credential_error_exposes_no_configuration(self, monkeypatch) -> None:
        monkeypatch.delenv("TEST_KEY", raising=False)
        from rfw_w.scenes.agents.llm_client import call_vision

        with pytest.raises(CredentialsMissing, match="credential environment variable is not set") as caught:
            call_vision(config(), system_prompt="s", user_prompt="u")
        assert caught.value.code == "LLM_CREDENTIAL_MISSING"
        assert "TEST_KEY" not in str(caught.value)

    def test_provider_error_body_cannot_reach_exception(self, monkeypatch) -> None:
        import io
        import urllib.error
        from rfw_w.scenes.agents.llm_client import call_vision

        marker = "synthetic-sensitive-marker"
        monkeypatch.setenv("TEST_KEY", marker)
        def reject(*args, **kwargs):
            raise urllib.error.HTTPError("https://example.invalid/v1", 401, marker, {},
                                         io.BytesIO(marker.encode()))
        monkeypatch.setattr("urllib.request.urlopen", reject)
        with pytest.raises(LLMError) as caught:
            call_vision(config(), system_prompt="s", user_prompt="u")
        assert caught.value.code == "LLM_HTTP_401"
        assert marker not in str(caught.value)
        assert caught.value.__cause__ is None
