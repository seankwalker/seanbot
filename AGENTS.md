# Repository Guidelines

## Project Structure & Module Organization
- `fb/` contains Facebook message preprocessing scripts and dependency pins in `fb/requirements.txt`.
- `fb/preprocess-data.py` reads exported inbox JSON and writes training pairs to CSV format.
- `fb/performance-test.py` benchmarks text cleaning and pair-building logic.
- `imessage/` contains iMessage extraction scripts and runtime config (`imessage/pyproject.toml`, `imessage/mise.toml`).
- `imessage/main.py` queries the macOS Messages SQLite database and writes prompt/response CSV output.
- `training/` contains Unsloth fine-tuning scripts and training workflow notes.
- Generated data files (for example `imessage/data.csv`) should be treated as local artifacts unless sanitized.

## Build, Test, and Development Commands
```bash
# iMessage extraction (Python 3.12 via uv/mise)
cd imessage
uv run main.py --chat_ids "+15551234567" --limit 200 --output training_pairs.csv

# Facebook preprocessing
python -m venv .venv && source .venv/bin/activate
pip install -r fb/requirements.txt
python fb/preprocess-data.py
python fb/performance-test.py

# Unit tests
python -m unittest imessage.test_main

# Fine-tuning entry point
python training/train_unsloth.py --dataset-file training_pairs.jsonl --max-steps 60
```
- Use `uv run` in `imessage/` to keep runtime consistent with `mise.toml`.
- Run `fb/performance-test.py` when changing cleaning or pairing logic.

## Coding Style & Naming Conventions
- Python only: 4-space indentation, `snake_case` for functions/variables, `UPPER_CASE` for constants.
- Keep data pipeline code explicit and readable; prefer small helper functions for reusable transforms.
- Follow existing script naming patterns for utilities (for example `preprocess-data.py`).
- Keep CLI flags descriptive (`--chat_ids`, `--limit`, `--output`) and document defaults in code.

## Testing Guidelines
- Run `python -m unittest imessage.test_main` before changing iMessage extraction logic.
- For pipeline changes, also verify output schema and row counts in generated CSVs.
- For performance-sensitive edits, compare before/after runtime using `fb/performance-test.py`.
- If you introduce reusable modules, add `pytest` tests under a new `tests/` directory.

## Commit & Pull Request Guidelines
- Existing history uses short, lowercase subjects (for example `latest`, `new dir structure`).
- Prefer clearer imperative commit subjects with scope, such as `imessage: handle empty texts`.
- PRs should include: changed paths, commands run, output summary (rows/runtime), and sanitized samples when output format changes.
- Link related issues/tasks when available.

## Security & Configuration Tips
- Never commit raw chat exports, phone numbers, or local message databases.
- Keep `.env`, training outputs, GGUF files, and generated CSV/JSONL files out of Git.
- Review generated CSV, JSONL, and model uploads for PII before sharing.
