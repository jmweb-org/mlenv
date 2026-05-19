# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-20

### Added
- Docker image and a published container entry point.
- Continuous integration across Python 3.10, 3.11 and 3.12 (lint, format
  check and tests).
- Expanded documentation, including a CI gating example.

## [0.1.0] - 2026-05-18

### Added
- `snapshot` command: capture Python, platform, installed packages, CUDA and
  cuDNN versions, GPU model and driver, and training-relevant environment
  variables into a single JSON file.
- `show` command: pretty-print the current environment.
- `diff` command: compare two snapshots, ranking changes by how likely they
  are to affect results, with `--json` and a `--check` gate.
- Optional GPU capture through NVML behind the `gpu` extra.

[0.2.0]: https://github.com/jmweb-org/mlenv/releases/tag/v0.2.0
[0.1.0]: https://github.com/jmweb-org/mlenv/releases/tag/v0.1.0
