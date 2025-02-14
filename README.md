# Entropy-Aware Branching
Source code for paper "Entropy-Aware Branching for Improved Mathematical Reasoning", currently under review.

Thanks to [xjdr's entropix project](https://github.com/xjdr-alt/entropix) for inspiration and [Sinatras & citizenhicks smolLM implementation](https://github.com/SinatrasC/entropix-smollm) for the basis of the visualizations.

Supports `Llama-3.2-1b`, `Llama-3.2-3b` and `Llama-3.1-8b` models & easy to add more.

## Getting Started

> Using [uv](https://docs.astral.sh/uv/getting-started/installation) for package management.

1. Make a virtual environment
  ```bash
  uv venv --python 3.12
  ```
2. Activate the virtual environment
  ```bash
  source .venv/bin/activate
  ```
3. Install dependencies
  ```bash
  uv pip install --project pyproject.toml .
  ```
5. Run model inference (see `run.py` for usage)
  ```bash
  python run.py
  ```
6. Plot & experiment with responses
```bash
entropix-dashboard
```
