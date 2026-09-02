"""Fail-closed configuration and health checks for NVIDIA Inference Hub."""

from urllib.parse import urlparse

import httpx

from config import settings

NVIDIA_INFERENCE_ORIGIN = "https://inference-api.nvidia.com"


class NvidiaInferenceError(RuntimeError):
    """Safe, credential-free Inference Hub readiness error."""


def validated_nvidia_base_url() -> str:
    """Return the configured Inference Hub origin only when it is exact and safe."""
    value = settings.nvidia_inference_base_url.strip().rstrip("/")
    parsed = urlparse(value)
    if (
        value != NVIDIA_INFERENCE_ORIGIN
        or parsed.scheme != "https"
        or parsed.hostname != "inference-api.nvidia.com"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "NVIDIA_INFERENCE_BASE_URL must be exactly "
            f"{NVIDIA_INFERENCE_ORIGIN}."
        )
    return value


def validate_nvidia_configuration() -> None:
    """Reject missing or unsafe internal-provider configuration before use."""
    validated_nvidia_base_url()
    if not settings.nvidia_inference_api_key.strip():
        raise ValueError("NVIDIA_INFERENCE_API_KEY is not configured.")
    if not settings.nvidia_inference_model.strip():
        raise ValueError("NVIDIA_INFERENCE_MODEL is not configured.")


async def nvidia_model_catalog() -> set[str]:
    """Fetch the accessible Inference Hub model IDs without following redirects."""
    validate_nvidia_configuration()
    async with httpx.AsyncClient(timeout=10, trust_env=False, follow_redirects=False) as client:
        response = await client.get(
            f"{validated_nvidia_base_url()}/v1/models",
            headers={"x-api-key": settings.nvidia_inference_api_key},
        )
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("NVIDIA Inference Hub returned an invalid model catalog.")
    return {
        str(item["id"])
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


async def verify_nvidia_model_access() -> None:
    """Verify credentials, VPN reachability, and configured model availability."""
    try:
        models = await nvidia_model_catalog()
    except httpx.TimeoutException as error:
        raise NvidiaInferenceError(
            "NVIDIA Inference Hub is unreachable; connect to the NVIDIA network."
        ) from error
    except httpx.HTTPStatusError as error:
        if error.response.status_code in {401, 403}:
            message = "NVIDIA Inference Hub authentication failed."
        else:
            message = f"NVIDIA Inference Hub returned HTTP {error.response.status_code}."
        raise NvidiaInferenceError(message) from error
    except httpx.HTTPError as error:
        raise NvidiaInferenceError(
            "NVIDIA Inference Hub is unreachable; connect to the NVIDIA network."
        ) from error
    except (TypeError, ValueError) as error:
        raise NvidiaInferenceError(str(error)) from error
    if settings.nvidia_inference_model not in models:
        raise NvidiaInferenceError("The configured NVIDIA Inference Hub model is unavailable.")
