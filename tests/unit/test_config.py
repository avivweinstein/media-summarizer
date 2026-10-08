from typing import Any, cast

import pytest
from pydantic import ValidationError

from config import Settings


def test_notion_is_opt_in_by_default() -> None:
    settings = cast(Any, Settings)(_env_file=None)

    assert settings.notion_enabled is False


def test_webhooks_are_opt_in_by_default() -> None:
    settings = cast(Any, Settings)(_env_file=None)

    assert settings.webhooks_enabled is False


def test_usage_limits_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        cast(Any, Settings)(_env_file=None, max_estimated_cost_usd=0)
