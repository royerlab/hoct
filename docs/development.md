# Development

## Checkout and tests

```bash
git clone https://github.com/royerlab/hoct
cd hoct
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Most tests use synthetic inputs. Build the optional CTC graph fixtures with
`uv run python scripts/prepare_test_data.py`; they are stored under the ignored
`.test-data/` directory. Model-dependent tests use `HOCT_TEST_MODEL`, or
`weights/general_v1.pt`. Unavailable data/model fixtures are skipped.

## Documentation

The site uses MkDocs Material and mkdocstrings, following the structure of
[tracksdata's documentation](https://royerlab.github.io/tracksdata/).
The API reference reads source files without running inference. The correction
page includes `examples/README.md`, so the site and repository share that guide.

```bash
# Live preview at http://127.0.0.1:8000
uv run --only-group docs mkdocs serve

# Strict production build into site/
uv run --only-group docs mkdocs build --strict
```

The `docs` dependency group installs only the documentation tools, without HOCT's
tracking or GUI dependencies. Edit pages under `docs/`, the demo guide under
`examples/`, and navigation in `mkdocs.yml`. Generated `site/` files are ignored.

## CI and publishing

The main CI workflow checks formatting, runs tests, and builds the Python wheel
and source distribution. The documentation workflow builds the site in strict
mode on pull requests and pushes to `main`. Broken internal links and build
warnings fail the documentation check.

On `main` in `royerlab/hoct`, the documentation workflow publishes the build to
GitHub Pages. Enable **Settings → Pages → Build and deployment → Source → GitHub
Actions** once in the repository. The configured site URL is
`https://royerlab.github.io/hoct/`. Pull requests and forks only build; they do
not publish the site. A manual workflow run on `main` can also publish it.

Release tags use the existing release workflow to build and publish the package.
To check packaging locally:

```bash
uv build
```
