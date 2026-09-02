from typing import Any, cast

from config import Settings


def test_notion_is_opt_in_by_default() -> None:
    settings = cast(Any, Settings)(_env_file=None)

    assert settings.notion_enabled is False
