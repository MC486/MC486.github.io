# AGENTS.md

## Cursor Cloud specific instructions

This is a single Python product: the **AI Word Strategy Game**, a terminal/CLI word game
with AI opponents. There is no web/API service and no external database server — persistence
is embedded SQLite (`data/game.db`, auto-created). The root `index.html` / `codeLoader.js`
are only a static GitHub Pages showcase and are not part of running or testing the game.

### Environment / how to run
- Dependencies are installed into a virtualenv at `.venv` by the startup update script
  (`pip install -r requirements.txt` + NLTK `words` corpus). Activate it with
  `source .venv/bin/activate` before running anything.
- `python3 -m venv` requires the system package `python3.12-venv` (already present in the
  VM image). If venv creation ever fails with an `ensurepip` error, install it via
  `sudo apt-get install -y python3.12-venv`.
- TensorFlow prints `oneDNN`/`Could not find cuda drivers` messages on import — these are
  informational, not errors (no GPU in this environment).

### Entry point gotcha
- The README says `python main.py`, but there is **no `main.py`**. The real entry point is
  `python game_app.py` (-> `engine.engine_core.main()`).

### Known-broken state on `main` (pre-existing, NOT environment issues)
- Running `python game_app.py` currently crashes at startup with
  `BaseRepository.__init__() missing 1 required positional argument: 'table_name'`
  (`NaiveBayesRepository.__init__` calls `super().__init__(db_manager)` without a table name).
  The full interactive game does not run on `main` until this is fixed.
- The test suite has substantial code/test drift: ~110 pass and ~112 fail
  (`AttributeError`/`TypeError`/`AssertionError`/`sqlite3.OperationalError`). These are
  pre-existing signature/schema mismatches in the app code, not missing dependencies
  (there are no import errors).
- The game's core gameplay modules DO work in isolation: `core.letter_pool`,
  `core.validation.word_validator` (NLTK-backed), and `core.word_scoring`.

### Testing caveats
- Do NOT run the whole suite blindly with plain `pytest`: two tests infinite-loop and
  OOM-kill the process —
  `tests/core/test_input_handler.py::test_invalid_word` and `::test_non_alphabetic_input`
  (they mock `input` with a constant `return_value` while `get_player_word` loops on
  invalid input, growing Mock call history until the VM runs out of memory).
  Always deselect them, e.g.:
  ```
  pytest --deselect "tests/core/test_input_handler.py::test_invalid_word" \
         --deselect "tests/core/test_input_handler.py::test_non_alphabetic_input"
  ```
- There is no configured linter (no ruff/flake8/pylint/pre-commit); `python -m py_compile`
  over the source tree is a reasonable syntax check.

### Config
- `config.yaml` is listed in `.gitignore` but is actually committed/tracked, so it exists
  on fresh clones. The app and `tests/test_config_loading.py` read it from the repo root.
