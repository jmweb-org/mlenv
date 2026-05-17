from __future__ import annotations

from mlenv.diff import ChangeKind, Risk, diff_snapshots, has_high_risk
from mlenv.models import (
    SECTION_CUDA,
    SECTION_GPUS,
    SECTION_PACKAGES,
    SECTION_PYTHON,
    Snapshot,
)


def snap(**sections):
    return Snapshot(sections={k: dict(v) for k, v in sections.items()})


def test_identical_snapshots_have_no_changes():
    a = snap(python={"version": "3.12.3"})
    assert diff_snapshots(a, a) == []


def test_added_and_removed_keys():
    old = snap(packages={"numpy": "1.26.0"})
    new = snap(packages={"numpy": "1.26.0", "scipy": "1.13.0"})
    changes = diff_snapshots(old, new)
    assert len(changes) == 1
    assert changes[0].kind is ChangeKind.ADDED
    assert changes[0].key == "scipy"
    assert changes[0].new == "1.13.0"


def test_cuda_change_is_high_risk():
    old = snap(**{SECTION_CUDA: {"torch_cuda": "12.1"}})
    new = snap(**{SECTION_CUDA: {"torch_cuda": "11.8"}})
    changes = diff_snapshots(old, new)
    assert changes[0].risk is Risk.HIGH
    assert has_high_risk(changes)


def test_torch_major_bump_is_high_risk():
    old = snap(**{SECTION_PACKAGES: {"torch": "2.3.0"}})
    new = snap(**{SECTION_PACKAGES: {"torch": "3.0.0"}})
    changes = diff_snapshots(old, new)
    assert changes[0].risk is Risk.HIGH


def test_torch_patch_bump_is_medium_risk():
    old = snap(**{SECTION_PACKAGES: {"torch": "2.3.0"}})
    new = snap(**{SECTION_PACKAGES: {"torch": "2.3.1"}})
    changes = diff_snapshots(old, new)
    assert changes[0].risk is Risk.MEDIUM


def test_python_patch_change_is_low_risk():
    old = snap(**{SECTION_PYTHON: {"version": "3.12.3"}})
    new = snap(**{SECTION_PYTHON: {"version": "3.12.4"}})
    changes = diff_snapshots(old, new)
    assert changes[0].risk is Risk.LOW


def test_python_minor_change_is_high_risk():
    old = snap(**{SECTION_PYTHON: {"version": "3.11.9"}})
    new = snap(**{SECTION_PYTHON: {"version": "3.12.0"}})
    changes = diff_snapshots(old, new)
    assert changes[0].risk is Risk.HIGH


def test_ordinary_package_change_is_low_risk():
    old = snap(**{SECTION_PACKAGES: {"rich": "13.7.0"}})
    new = snap(**{SECTION_PACKAGES: {"rich": "13.8.0"}})
    changes = diff_snapshots(old, new)
    assert changes[0].risk is Risk.LOW
    assert not has_high_risk(changes)


def test_gpu_change_is_high_risk():
    old = snap(**{SECTION_GPUS: {"count": "1", "gpu0": "NVIDIA L40S"}})
    new = snap(**{SECTION_GPUS: {"count": "1", "gpu0": "NVIDIA A100"}})
    changes = diff_snapshots(old, new)
    assert changes[0].risk is Risk.HIGH


def test_changes_sorted_high_risk_first():
    old = snap(
        **{
            SECTION_PACKAGES: {"rich": "13.7.0", "torch": "2.3.0"},
        }
    )
    new = snap(
        **{
            SECTION_PACKAGES: {"rich": "13.8.0", "torch": "3.0.0"},
        }
    )
    changes = diff_snapshots(old, new)
    assert changes[0].key == "torch"
    assert changes[0].risk is Risk.HIGH
    assert changes[-1].risk is Risk.LOW
