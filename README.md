# mlenv

[![CI](https://github.com/jmweb-org/mlenv/actions/workflows/ci.yml/badge.svg)](https://github.com/jmweb-org/mlenv/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mlenv-cli.svg)](https://pypi.org/project/mlenv-cli/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Snapshot the machine learning environment to a single file, and diff two
snapshots to see exactly what changed.

"It trained fine last week" usually comes down to something in the stack
moving: a different CUDA build behind PyTorch, a minor Python bump, a driver
upgrade, a GPU swapped on a shared box. `mlenv` captures all of that into one
JSON file you can commit, share, and compare, with the changes most likely to
affect results ranked first.

```console
$ mlenv snapshot -o good.json     # when results were correct
$ mlenv snapshot -o now.json      # after something changed
$ mlenv diff good.json now.json
risk  section   key         change
high  cuda      torch_cuda  12.1 -> 11.8
high  python    version     3.11.9 -> 3.12.0
low   packages  rich        13.7.0 -> 13.8.0
```

## Install

```console
$ pip install mlenv-cli                 # from PyPI, once released
$ pip install git+https://github.com/jmweb-org/mlenv   # latest, available now          # core: Python, platform, packages, CUDA, env vars
$ pip install "mlenv-cli[gpu]"   # adds GPU model and driver capture via NVML
```

`mlenv` has no heavy dependencies and runs anywhere. GPU details are read
through NVML only when the `gpu` extra is installed and a driver is present;
without it, every other section is still captured.

## What it captures

| Section | Contents |
| --- | --- |
| `python` | Version, implementation, interpreter path |
| `platform` | OS, release, architecture, libc |
| `packages` | Every installed distribution and its version |
| `cuda` | PyTorch build, its CUDA and cuDNN versions, availability |
| `gpus` | GPU count, model per index, driver version |
| `env` | Training-relevant variables (`CUDA_*`, `NCCL_*`, `OMP_NUM_THREADS`, ...) |

## Commands

```console
$ mlenv snapshot -o env.json   # write a snapshot (omit -o to print to stdout)
$ mlenv show                   # pretty-print the current environment
$ mlenv diff a.json b.json     # show changes, highest risk first
$ mlenv diff a.json b.json --json    # machine-readable changes
$ mlenv diff a.json b.json --check   # exit non-zero on any high-risk change
```

### In CI

`--check` turns the diff into a gate. Commit a baseline snapshot and fail the
build when the runner drifts from it in a way that could change results:

```yaml
- run: mlenv snapshot -o current.json
- run: mlenv diff baseline.json current.json --check
```

## Risk levels

A change is **high** risk when it routinely moves numerical results: anything
under `cuda`, a GPU model or count or driver change, or a Python minor-version
bump. A major-version bump of a sensitive package (`torch`, `numpy`,
`tensorflow`, `jax`, `transformers`) is high; a smaller bump is **medium**.
Everything else is **low**. Only high-risk changes trip `--check`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Ran; no high-risk change (or `--check` not set) |
| 1 | `--check` found a high-risk change |
| 2 | A snapshot file was missing or invalid |

## License

MIT. See [LICENSE](LICENSE).
