# Contributing to ClearView

Thanks for taking the time to contribute! Bug reports, fixes, tests, docs
and new datasets or architectures are all welcome. Participation is governed
by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and requesting features

Open an [issue](https://github.com/dronefreak/clearview/issues). For bugs,
please include:

- the exact command or code you ran, and the full error output
- what you expected to happen instead
- your environment: OS, Python, PyTorch and CUDA versions, GPU

## Development setup

ClearView requires Python 3.10 or newer.

```bash
git clone https://github.com/dronefreak/clearview.git
cd clearview
make setup-venv && source .venv/bin/activate
make install-dev          # pip install -e ".[dev,metrics,hub]"
make pre-commit-install   # ruff, mypy, prettier, markdownlint, ... on commit
make test                 # the full suite runs on CPU in a few minutes
```

Run `make help` to list every target.

## Making a change

1. Branch from `main`: `feature/...`, `fix/...`, `docs/...`, `test/...`.
2. Keep each pull request focused on one change.
3. Add or update tests. Every bug fix should come with a test that fails
   without the fix.
4. Update the docs (README, `clearview/scripts/README.md`,
   `configs/mix/README.md`) when behaviour or CLI flags change.
5. Check locally before pushing:

   ```bash
   make lint format-check test
   ```

6. Open the pull request with a short description of what changed and why,
   and link related issues (`Closes #123`).

## Coding standards

- **Formatting and linting**: [ruff](https://docs.astral.sh/ruff/)
  (`make format`), 88-character lines.
- **Type hints** on all public functions. mypy runs in pre-commit.
- **Docstrings**: Google style (`Args:`, `Returns:`, `Raises:`).
- **Comments** explain _why_, not _what_.
- **Tests** live in `tests/test_*.py`, use pytest, and must run on CPU with
  small synthetic inputs. Anything that needs a GPU, network access or an
  optional package must skip cleanly without it (`pytest.mark.skipif`,
  `pytest.importorskip`).

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
(enforced by a `commit-msg` hook): one line, `type(optional scope): summary`.

```text
fix(training): restore LR scheduler state on --resume
feat(data): add GT-RAIN dataset loader
docs: document the RGB PSNR/SSIM evaluation protocol
test: cover NumPy inputs in compute_ssim
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.

## Questions

Open a [discussion](https://github.com/dronefreak/clearview/discussions)
or an [issue](https://github.com/dronefreak/clearview/issues).
