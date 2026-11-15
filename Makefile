.PHONY: check

check:
	uv run ruff format --check && uv run ruff check --ignore-noqa && uv run basedpyright && uv run ast-grep test --skip-snapshot-tests && uv run ast-grep scan && uv run pytest
