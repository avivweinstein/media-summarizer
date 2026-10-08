from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from config import settings
from nvidia_inference import (
    NVIDIA_INFERENCE_ORIGIN,
    NvidiaInferenceError,
    nvidia_http_client,
    nvidia_model_catalog,
    validate_nvidia_configuration,
    validated_nvidia_base_url,
    verify_nvidia_model_access,
)


def test_validated_base_url_accepts_only_exact_inference_hub(mocker: MagicMock) -> None:
    mocker.patch.object(settings, "nvidia_inference_base_url", NVIDIA_INFERENCE_ORIGIN)

    assert validated_nvidia_base_url() == NVIDIA_INFERENCE_ORIGIN

    for unsafe in (
        "https://api.anthropic.com",
        "http://inference-api.nvidia.com",
        "https://inference-api.nvidia.com.evil.example",
        "https://inference-api.nvidia.com/v1",
        "https://inference-api.nvidia.com?redirect=evil",
    ):
        mocker.patch.object(settings, "nvidia_inference_base_url", unsafe)
        with pytest.raises(ValueError, match="must be exactly"):
            validated_nvidia_base_url()


def test_configuration_requires_internal_key_and_model(mocker: MagicMock) -> None:
    mocker.patch.object(settings, "nvidia_inference_base_url", NVIDIA_INFERENCE_ORIGIN)
    mocker.patch.object(settings, "nvidia_inference_api_key", "")
    mocker.patch.object(settings, "nvidia_inference_model", "internal-model")

    with pytest.raises(ValueError, match="API_KEY"):
        validate_nvidia_configuration()


async def test_catalog_disables_proxy_and_redirects(mocker: MagicMock) -> None:
    mocker.patch.object(settings, "nvidia_inference_base_url", NVIDIA_INFERENCE_ORIGIN)
    mocker.patch.object(settings, "nvidia_inference_api_key", "secret-value")
    mocker.patch.object(settings, "nvidia_inference_model", "internal-model")
    response = MagicMock()
    response.json.return_value = {"data": [{"id": "internal-model"}]}
    response.raise_for_status.return_value = None
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get.return_value = response
    factory = mocker.patch("nvidia_inference.nvidia_http_client", return_value=client)

    assert await nvidia_model_catalog() == {"internal-model"}

    factory.assert_called_once_with(timeout=10)
    client.get.assert_awaited_once_with(
        f"{NVIDIA_INFERENCE_ORIGIN}/v1/models",
        headers={"x-api-key": "secret-value"},
    )


async def test_sdk_cannot_forward_anthropic_auth_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anthropic import AsyncAnthropic

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "public-secret-must-not-leave")
    captured_headers: httpx.Headers | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_headers
        captured_headers = request.headers
        return httpx.Response(
            200,
            json={
                "id": "message-id",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "{}"}],
                "model": "internal-model",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    async with nvidia_http_client(
        timeout=10,
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = AsyncAnthropic(
            api_key="nvidia-key",
            base_url=NVIDIA_INFERENCE_ORIGIN,
            http_client=http_client,
        )
        await client.messages.create(
            model="internal-model",
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )

    assert captured_headers is not None
    assert captured_headers.get("x-api-key") == "nvidia-key"
    assert "authorization" not in captured_headers


async def test_model_access_error_never_contains_key(mocker: MagicMock) -> None:
    secret = "do-not-leak-this-key"
    mocker.patch.object(settings, "nvidia_inference_base_url", NVIDIA_INFERENCE_ORIGIN)
    mocker.patch.object(settings, "nvidia_inference_api_key", secret)
    mocker.patch.object(settings, "nvidia_inference_model", "internal-model")
    request = httpx.Request(
        "GET",
        f"{NVIDIA_INFERENCE_ORIGIN}/v1/models",
        headers={"x-api-key": secret},
    )
    response = httpx.Response(403, request=request)
    mocker.patch(
        "nvidia_inference.nvidia_model_catalog",
        new=AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "forbidden",
                request=request,
                response=response,
            )
        ),
    )

    with pytest.raises(NvidiaInferenceError) as caught:
        await verify_nvidia_model_access()

    assert "authentication failed" in str(caught.value)
    assert secret not in str(caught.value)


async def test_model_access_fails_when_configured_model_is_absent(mocker: MagicMock) -> None:
    mocker.patch.object(settings, "nvidia_inference_model", "missing-model")
    mocker.patch(
        "nvidia_inference.nvidia_model_catalog",
        new=AsyncMock(return_value={"another-model"}),
    )

    with pytest.raises(NvidiaInferenceError, match="unavailable"):
        await verify_nvidia_model_access()
