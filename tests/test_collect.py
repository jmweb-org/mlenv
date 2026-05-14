from __future__ import annotations

from types import SimpleNamespace

from mlenv.collect import (
    collect_cuda,
    collect_env,
    collect_gpus,
    collect_packages,
    collect_python,
    collect_snapshot,
)
from mlenv.models import SECTION_CUDA, SECTION_ENV, SECTION_GPUS


def test_collect_python_reports_version_and_implementation():
    info = collect_python()
    assert "version" in info
    assert "implementation" in info


def test_collect_packages_normalizes_names_and_sorts():
    pkgs = collect_packages([("Torch_Vision", "0.18.0"), ("NumPy", "1.26.0")])
    assert pkgs == {"numpy": "1.26.0", "torch-vision": "0.18.0"}
    assert list(pkgs) == sorted(pkgs)


def test_collect_env_filters_to_relevant_names_and_prefixes():
    env = collect_env(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "NCCL_DEBUG": "INFO",
            "HF_HOME": "/data/hf",
            "PATH": "/usr/bin",
            "EDITOR": "vim",
        }
    )
    assert env == {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": "/data/hf",
        "NCCL_DEBUG": "INFO",
    }


def test_collect_cuda_without_torch_is_empty():
    assert collect_cuda(torch_module=None) == {}


def test_collect_cuda_reads_fake_torch():
    fake_torch = SimpleNamespace(
        __version__="2.3.0",
        version=SimpleNamespace(cuda="12.1"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 8907)),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    out = collect_cuda(torch_module=fake_torch)
    assert out["torch"] == "2.3.0"
    assert out["torch_cuda"] == "12.1"
    assert out["cudnn"] == "8907"
    assert out["cuda_available"] == "True"


def test_collect_gpus_from_probe():
    def probe():
        return [(0, "NVIDIA L40S", "550.54"), (1, "NVIDIA L40S", "550.54")]

    out = collect_gpus(probe=probe)
    assert out["count"] == "2"
    assert out["gpu0"] == "NVIDIA L40S"
    assert out["driver"] == "550.54"


def test_collect_gpus_handles_probe_failure():
    def probe():
        raise RuntimeError("no driver")

    assert collect_gpus(probe=probe) == {}


def test_collect_snapshot_drops_empty_sections():
    snap = collect_snapshot(
        distributions=[("numpy", "1.26.0")],
        torch_module=None,
        gpu_probe=lambda: [],
        environ={"PATH": "/usr/bin"},
    )
    assert SECTION_CUDA not in snap.sections
    assert SECTION_GPUS not in snap.sections
    assert SECTION_ENV not in snap.sections
    assert snap.section("packages") == {"numpy": "1.26.0"}
