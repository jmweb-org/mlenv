from __future__ import annotations

import json

from typer.testing import CliRunner

from mlenv import __version__
from mlenv import cli as cli_module
from mlenv.models import Snapshot
from mlenv.storage import save

runner = CliRunner()


def test_version():
    result = runner.invoke(cli_module.app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_snapshot_to_stdout_is_valid_json():
    result = runner.invoke(cli_module.app, ["snapshot"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "sections" in payload
    assert "python" in payload["sections"]


def test_snapshot_to_file(tmp_path):
    out = tmp_path / "env.json"
    result = runner.invoke(cli_module.app, ["snapshot", "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "python" in json.loads(out.read_text())["sections"]


def test_show_runs():
    result = runner.invoke(cli_module.app, ["show"])
    assert result.exit_code == 0


def _write(tmp_path, name, sections):
    path = tmp_path / name
    save(Snapshot(sections=sections), path)
    return path


def test_diff_reports_changes_as_json(tmp_path):
    a = _write(tmp_path, "a.json", {"packages": {"torch": "2.3.0"}})
    b = _write(tmp_path, "b.json", {"packages": {"torch": "3.0.0"}})
    result = runner.invoke(cli_module.app, ["diff", str(a), str(b), "--json"])
    assert result.exit_code == 0
    changes = json.loads(result.stdout)
    assert changes[0]["key"] == "torch"
    assert changes[0]["risk"] == "high"


def test_diff_check_fails_on_high_risk(tmp_path):
    a = _write(tmp_path, "a.json", {"cuda": {"torch_cuda": "12.1"}})
    b = _write(tmp_path, "b.json", {"cuda": {"torch_cuda": "11.8"}})
    result = runner.invoke(cli_module.app, ["diff", str(a), str(b), "--check"])
    assert result.exit_code == cli_module.EXIT_HIGH_RISK


def test_diff_check_passes_when_low_risk(tmp_path):
    a = _write(tmp_path, "a.json", {"packages": {"rich": "13.7.0"}})
    b = _write(tmp_path, "b.json", {"packages": {"rich": "13.8.0"}})
    result = runner.invoke(cli_module.app, ["diff", str(a), str(b), "--check"])
    assert result.exit_code == 0


def test_diff_missing_file_is_bad_input(tmp_path):
    a = _write(tmp_path, "a.json", {"packages": {"rich": "13.7.0"}})
    result = runner.invoke(cli_module.app, ["diff", str(a), str(tmp_path / "missing.json")])
    assert result.exit_code == cli_module.EXIT_BAD_INPUT
