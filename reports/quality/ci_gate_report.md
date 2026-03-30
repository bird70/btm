# CI Quality Gate Report

## Commands
- `./.venv312/bin/ruff check src tests`
- `./.venv312/bin/ruff format --check src tests`
- `./.venv312/bin/python -m pytest -q`

## Results
- ruff check: PASS
- ruff format --check: PASS
- pytest: PASS

## Output Snippets

### ruff check

```text
All checks passed!
```

### ruff format --check

```text
41 files already formatted
```

### pytest

```text
............................                                             [100%]
28 passed in 51.68s
```
