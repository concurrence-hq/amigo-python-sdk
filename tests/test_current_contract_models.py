"""Regressions for response and request shapes in the current Classic contract."""

import pytest

from amigo_sdk.generated.model import (
    LLMConfigInput,
    LLMType,
    MetricCreateMetricRequest,
    OrganizationCreateAgentVersionResponse,
)


def test_agent_version_response_does_not_require_removed_id():
    response = OrganizationCreateAgentVersionResponse.model_validate(
        {"version": 2, "created_at": "2026-09-07T12:00:00Z"}
    )
    assert response.version == 2


def test_metric_creation_accepts_initial_version_contract():
    payload = {
        "name": "Synthetic scheduling check",
        "applied_to_services": [],
        "additional_notes": None,
        "tags": {},
        "initial_version_metric_value": {
            "description": "The answer does not claim an unverified appointment change",
            "metric_value": {"type": "boolean"},
        },
    }
    request = MetricCreateMetricRequest.model_validate(payload)
    assert request.model_dump(mode="json") == payload


def test_current_model_identifier_can_be_deserialized():
    assert LLMType("openai_gpt-5.6-sol").value == "openai_gpt-5.6-sol"


@pytest.mark.parametrize(
    "llm_name", ["google_gemini-3.5-flash", "google_gemini-3.5-flash-lite"]
)
def test_llm_config_accepts_gemini_3_5_ids(llm_name):
    config = LLMConfigInput.model_validate({"llm_name": llm_name})
    assert config.model_dump(mode="json")["llm_name"] == llm_name
