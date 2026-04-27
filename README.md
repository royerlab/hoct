# HOCT Inference

Inference API and CLI for the Higher-Order Cell Tracking Transformer (HOCT) model with JIT-compiled models.

## Installation

```bash
# From the monorepo root
uv sync

# Or install directly
pip install -e hoct_inference
```

## Usage

### Command Line

```bash
hoct-inference --help
```

### Python API

```python
import hoct_inference

# TODO: Add API usage examples
```

## Development

```bash
# Install development dependencies
cd hoct_inference
uv sync --extra dev

# Run tests
pytest

# Run linting
ruff check .

# Format code
ruff format .
```
