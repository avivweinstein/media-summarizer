# Contributing

Thanks for your interest in contributing to Media Summarizer!

## Getting started

1. Fork the repo and clone your fork
2. Follow the [Setup instructions](README.md#setup)
3. Create a branch for your work: `git checkout -b my-feature`
4. Make your changes
5. Run lint and tests:
   ```bash
   uv run ruff check .
   uv run mypy .
   uv run pytest
   ```
6. Commit and push, then open a pull request

## Code style

- Python 3.11+, type-annotated
- Linted with [ruff](https://docs.astral.sh/ruff/) and type-checked with [mypy](https://mypy.readthedocs.io/) (strict mode)
- Line length: 100 characters
- Write docstrings for public functions
- Keep functions small and focused

## Adding a new source

Media sources follow the `BaseSource` pattern in `sources/base.py`:

1. Create `sources/your_source.py` implementing `BaseSource.fetch()`
2. Add URL detection in `pipeline.detect_source()`
3. Wire it up in `pipeline.run_job()`
4. Add unit tests in `tests/unit/`

See `sources/youtube.py` and `sources/podcast.py` for examples.

## Tests

- **Unit tests** (`tests/unit/`) run without API keys and should stay fast
- **Integration tests** (`tests/integration/`) require real API keys in `.env`
- Run unit tests: `uv run pytest`
- Run integration tests: `uv run pytest -m integration`

## Ideas for contributions

See [ENHANCEMENTS.md](ENHANCEMENTS.md) for a list of planned features that are ready to be picked up.
