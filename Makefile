.PHONY: check

check:
	uv run ruff format --check && uv run ruff check --ignore-noqa && uv run basedpyright && uv run pytest
