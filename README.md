# EET Inference

Inference API and CLI for the Edge Embedding Tracking (EET) model with JIT-compiled models.

## Installation

```bash
# From the monorepo root
uv sync

# Or install directly
pip install -e eet_inference
```

## Usage

### Command Line

```bash
eet-inference --help
```

### Python API

```python
import eet_inference

# TODO: Add API usage examples
```

## Development

```bash
# Install development dependencies
cd eet_inference
uv sync --extra dev

# Run tests
pytest

# Run linting
ruff check .

# Format code
ruff format .
```
