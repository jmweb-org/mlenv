"""Collect the current environment into a :class:`~mlenv.models.Snapshot`.

Each collector takes its data source as an argument with a real default, so
the same functions can be driven with fakes in tests without monkeypatching
the interpreter.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from types import ModuleType

from mlenv.models import (
    SECTION_CUDA,
    SECTION_ENV,
    SECTION_GPUS,
    SECTION_PACKAGES,
    SECTION_PLATFORM,
    SECTION_PYTHON,
    Snapshot,
)

# Environment variables worth recording. Exact names plus prefixes that are
# known to change training behaviour.
ENV_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
)
ENV_PREFIXES = ("CUDA_", "NCCL_", "PYTORCH_", "HF_", "TORCH_", "XLA_")


def collect_python(info: object | None = None) -> dict[str, str]:
    impl = platform.python_implementation()
    version = platform.python_version()
    return {
        "version": version,
        "implementation": impl,
        "executable": sys.executable,
    }


def collect_platform() -> dict[str, str]:
    libc, libc_version = platform.libc_ver()
    out = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    if libc:
        out["libc"] = f"{libc} {libc_version}".strip()
    return out


def collect_packages(
    distributions: Iterable[tuple[str, str]] | None = None,
) -> dict[str, str]:
    if distributions is None:
        distributions = _installed_distributions()
    packages: dict[str, str] = {}
    for name, version in distributions:
        key = name.lower().replace("_", "-")
        packages[key] = version
    return dict(sorted(packages.items()))


def _installed_distributions() -> Iterable[tuple[str, str]]:
    from importlib.metadata import distributions

    for dist in distributions():
        name = dist.metadata["Name"]
        if name:
            yield name, dist.version


def collect_cuda(torch_module: ModuleType | None = None) -> dict[str, str]:
    torch = torch_module if torch_module is not None else _import_torch()
    if torch is None:
        return {}
    out: dict[str, str] = {"torch": _getattr_str(torch, "__version__")}
    version = getattr(torch, "version", None)
    if version is not None:
        cuda = getattr(version, "cuda", None)
        if cuda:
            out["torch_cuda"] = str(cuda)
    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None) if backends else None
    if cudnn is not None:
        try:
            value = cudnn.version()
        except Exception:  # pragma: no cover - defensive
            value = None
        if value:
            out["cudnn"] = str(value)
    cuda_runtime = getattr(torch, "cuda", None)
    if cuda_runtime is not None:
        with suppress(Exception):  # defensive: is_available can raise on bad installs
            out["cuda_available"] = str(bool(cuda_runtime.is_available()))
    return out


def collect_gpus(
    probe: Callable[[], list[tuple[int, str, str]]] | None = None,
) -> dict[str, str]:
    reader = probe if probe is not None else _nvml_gpus
    try:
        devices = reader()
    except Exception:
        return {}
    out: dict[str, str] = {}
    if not devices:
        return out
    out["count"] = str(len(devices))
    for index, name, _driver in devices:
        out[f"gpu{index}"] = name
    driver = devices[0][2]
    if driver:
        out["driver"] = driver
    return out


def collect_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = environ if environ is not None else os.environ
    out: dict[str, str] = {}
    for name, value in source.items():
        if name in ENV_NAMES or name.startswith(ENV_PREFIXES):
            out[name] = value
    return dict(sorted(out.items()))


def collect_snapshot(
    *,
    distributions: Iterable[tuple[str, str]] | None = None,
    torch_module: ModuleType | None = None,
    gpu_probe: Callable[[], list[tuple[int, str, str]]] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Snapshot:
    sections: dict[str, dict[str, str]] = {
        SECTION_PYTHON: collect_python(),
        SECTION_PLATFORM: collect_platform(),
        SECTION_PACKAGES: collect_packages(distributions),
        SECTION_CUDA: collect_cuda(torch_module),
        SECTION_GPUS: collect_gpus(gpu_probe),
        SECTION_ENV: collect_env(environ),
    }
    sections = {name: values for name, values in sections.items() if values}
    return Snapshot(sections=sections)


def _import_torch() -> ModuleType | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _nvml_gpus() -> list[tuple[int, str, str]]:
    import pynvml

    pynvml.nvmlInit()
    try:
        driver = _as_text(pynvml.nvmlSystemGetDriverVersion())
        out: list[tuple[int, str, str]] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = _as_text(pynvml.nvmlDeviceGetName(handle))
            out.append((index, name, driver))
        return out
    finally:
        pynvml.nvmlShutdown()


def _getattr_str(obj: object, name: str) -> str:
    return str(getattr(obj, name, ""))


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)
