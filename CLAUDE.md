# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrewAI is a multi-agent automation framework for orchestrating autonomous AI agents. It provides two complementary paradigms: **Crews** (collaborative autonomous agents) and **Flows** (event-driven workflows with state management).

## Repository Structure

This is a **UV workspace** with three packages under `lib/`:

- **`lib/crewai/`** — Core framework (agents, tasks, crews, flows, memory, LLM integration)
- **`lib/crewai-tools/`** — Extended tools package (web scraping, databases, APIs, integrations)
- **`lib/devtools/`** — Internal development utilities

Source code lives at `lib/crewai/src/crewai/` and `lib/crewai-tools/src/crewai_tools/`. Tests live at `lib/crewai/tests/` and `lib/crewai-tools/tests/`.

## Build & Development Commands

```bash
# Install all dependencies (uses UV)
uv sync --all-groups --all-extras

# Run full test suite (parallel, network blocked, 60s timeout)
pytest

# Run a single test file
pytest lib/crewai/tests/path/to/test_file.py

# Run a single test
pytest lib/crewai/tests/path/to/test_file.py::TestClass::test_name

# Run tests without parallelism (useful for debugging)
pytest -n0 lib/crewai/tests/path/to/test_file.py

# Lint (ruff)
ruff check lib/
ruff format lib/

# Type check (mypy strict mode)
mypy lib/crewai/src/ lib/crewai-tools/src/

# Pre-commit hooks (ruff, mypy, uv-lock, commitizen)
pre-commit run --all-files
```

## Testing Conventions

- **Network is blocked by default** (`--block-network`). HTTP interactions must use VCR cassettes via `@pytest.mark.vcr`. Set `PYTEST_VCR_RECORD_MODE=once` to record new cassettes (requires real API keys).
- Tests marked `@pytest.mark.telemetry` are exempt from telemetry mocking.
- Auto-fixtures in `conftest.py`: event bus cleanup after each test, temporary `CREWAI_STORAGE_DIR`, VCR header filtering for sensitive data.
- CI runs tests split across 8 parallel groups on Python 3.10–3.13.

## Code Quality

- **Ruff**: Linter and formatter. Auto-fix enabled. Tests are excluded from lint rules like `S101` (assert). Relative imports are banned — use absolute imports.
- **MyPy**: Strict mode with Pydantic plugin. Covers `lib/crewai/src/` and `lib/crewai-tools/src/` (excludes tests and CLI templates).
- **Commits**: Conventional commits enforced via commitizen pre-commit hook.
- **Python**: Target version 3.10. Use `from __future__ import annotations` style (Ruff `future-annotations = true`).

## Architecture — Key Abstractions

**Core classes** (exported from `crewai`):
- `Agent` — AI agent with role, goal, backstory, tools, and optional memory
- `Task` — Unit of work assigned to an agent, with expected output and optional guardrails
- `Crew` — Orchestrates agents executing tasks (sequential or hierarchical `Process`)
- `Flow` — Event-driven workflow with typed state, decorators `@start()`, `@listen()`, `@router()`
- `LLM` — Unified LLM interface supporting OpenAI, Anthropic, Bedrock, Google, Azure, LiteLLM, etc.

**Memory system** (`crewai/memory/`): Short-term (ChromaDB), long-term (SQLite), entity memory, external (Mem0).

**Events** (`crewai/events/`): Internal event bus for lifecycle hooks and observability.

**Tools** (`crewai/tools/` and `crewai_tools/`): `BaseTool` base class, `@tool` decorator, `StructuredTool`, MCP support.

**CLI** (`crewai/cli/`): Entry point `crewai` command — scaffolding (`create crew/flow`), running, training, deploying.

## Decorator Patterns

Crews use a decorator-based class pattern:
- `@CrewBase` on the class, then `@agent`, `@task`, `@crew` on methods
- `@before_kickoff`, `@after_kickoff` for lifecycle callbacks
- Agents/tasks are configured via YAML files in `config/agents.yaml` and `config/tasks.yaml`

Flows use:
- `@start()` for entry points, `@listen()` for reactive methods, `@router()` for conditional branching
- `@persist` for state persistence
- `or_()` and `and_()` for combining conditions
