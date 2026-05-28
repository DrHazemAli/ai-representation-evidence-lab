# AI Representation Evidence Lab

A companion source-code repository for the article section on representation observability.

The goal of this repository is not to provide a production observability platform.
The goal is to make the article's core engineering claim reproducible:

> A prompt and a final answer are not sufficient evidence when the behavior depends on representation.
> The stronger artifact is the representation path.

This repository demonstrates how to inspect that path across:

- Unicode identity
- UTF-8 byte form
- Unicode normalization
- tokenizer evidence when available
- retrieval candidates
- context-promotion decisions
- decoding configuration
- reproducible figures
- evidence records that avoid permanent storage of raw sensitive content

## Why this repository exists

The article argues that the model does not receive text as a human object.
It receives a constructed representation.

This repository turns that argument into inspectable artifacts that engineers can run, test, modify, and extend.

## Repository structure

```text
ai-representation-evidence-lab/
  README.md
  pyproject.toml
  src/
    representation_evidence/
      __init__.py
      cli.py
      decoding.py
      evidence.py
      figures.py
      retrieval.py
      unicode_trace.py
  examples/
    run_full_demo.py
  tests/
    test_unicode_trace.py
    test_retrieval.py
    test_decoding.py
  outputs/
    sample-full-demo.json
    sample-input-evidence.json
    sample-context-evidence.json
    sample-decoding-evidence.json
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional:

```bash
pip install ".[figures,tokenizer]"
```

The code runs without tokenizer dependencies. If `tiktoken` is installed, tokenizer IDs are added to the evidence record.

## CLI examples

Inspect a text input:

```bash
rep-evidence trace-input "H\u200b"
```

Compare two visually similar inputs:

```bash
rep-evidence compare "é" "e\u0301"
```

Demonstrate retrieval promotion:

```bash
rep-evidence retrieval-demo
```

Demonstrate decoding contracts:

```bash
rep-evidence decoding-demo
```

Generate article figures:

```bash
rep-evidence generate-figures --out figures
```

Run the full demo and write evidence outputs:

```bash
python examples/run_full_demo.py
```

## Mapping to the article

| Article idea | Code artifact |
|---|---|
| A letter is not primitive inside the machine | `unicode_trace.py` |
| Unicode and normalization can change the object | `trace-input`, `compare` |
| Token IDs depend on tokenizer contract | optional tokenizer evidence in `unicode_trace.py` |
| Vector proximity is not authority | `retrieval.py` |
| Context assembly is a promotion boundary | `retrieval-demo` |
| Output is selected under a decoding contract | `decoding.py` |
| Evidence should be reproducible | `evidence.py`, `examples/run_full_demo.py` |
| Figures should be generated, not manually drawn | `figures.py` |

## Production note

This repository intentionally uses minimal local examples.

In production, the same idea should be integrated with your observability and governance stack. Sensitive content should not be stored blindly. Use hashes, redaction, scoped retention, access control, policy labels, and audit controls according to your environment.

The objective is evidentiary sufficiency: preserving enough of the representation path to prove what the system actually processed when correctness, safety, reproducibility, or auditability depends on it.
