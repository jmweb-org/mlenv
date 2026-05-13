from __future__ import annotations

from mlenv.models import Snapshot
from mlenv.storage import dumps, load, loads, save


def test_round_trip_through_json():
    snap = Snapshot(
        sections={
            "python": {"version": "3.12.3"},
            "packages": {"numpy": "1.26.0", "torch": "2.3.0"},
        }
    )
    restored = loads(dumps(snap))
    assert restored == snap


def test_save_and_load(tmp_path):
    snap = Snapshot(sections={"cuda": {"torch_cuda": "12.1"}})
    path = tmp_path / "env.json"
    save(snap, path)
    assert path.read_text().endswith("\n")
    assert load(path) == snap


def test_sections_serialized_in_canonical_order():
    snap = Snapshot(
        sections={
            "env": {"CUDA_VISIBLE_DEVICES": "0"},
            "python": {"version": "3.12.3"},
        }
    )
    data = snap.to_dict()
    assert list(data["sections"]) == ["python", "env"]


def test_from_dict_coerces_values_to_strings():
    snap = Snapshot.from_dict({"schema_version": 1, "sections": {"gpus": {"count": 2}}})
    assert snap.section("gpus")["count"] == "2"
