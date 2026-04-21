"""Verification tests for file-based-planning helper scripts.

Focuses on the --output-dir / output_dir parameter so planning files land
in a session-scoped location (e.g. {DOCS_ROOT}) instead of the agent's
current working directory.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = ROOT / "helix" / "builtin_skills" / "file-based-planning" / "scripts"
_TEMPLATES_DIR = ROOT / "helix" / "builtin_skills" / "file-based-planning" / "templates"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


init_planning_mod = _load_module("fbp_init_planning", _SCRIPTS_DIR / "init_planning.py")
check_complete_mod = _load_module("fbp_check_complete", _SCRIPTS_DIR / "check_complete.py")
session_catchup_mod = _load_module("fbp_session_catchup", _SCRIPTS_DIR / "session_catchup.py")


def test_init_planning_writes_to_output_dir():
    """init_planning must write planning files to output_dir, not cwd."""
    with tempfile.TemporaryDirectory() as cwd_td, tempfile.TemporaryDirectory() as out_td:
        cwd = Path(cwd_td)
        output_dir = Path(out_td)

        old_cwd = os.getcwd()
        os.chdir(cwd)
        try:
            result, code = init_planning_mod.init_planning(
                project_name="test-proj",
                templates_dir=_TEMPLATES_DIR,
                output_dir=output_dir,
            )
        finally:
            os.chdir(old_cwd)

        assert code == 0, f"unexpected exit code: {code}"
        assert result["status"] == "ok", f"unexpected status: {result}"
        # Files should be in output_dir, not cwd
        assert (output_dir / "task_plan.md").exists()
        assert (output_dir / "findings.md").exists()
        assert (output_dir / "progress.md").exists()
        assert not (cwd / "task_plan.md").exists()
        print("  init_planning writes to output_dir OK")


def test_check_complete_reads_from_output_dir():
    """check_complete must look for task_plan.md in output_dir, not cwd."""
    with tempfile.TemporaryDirectory() as cwd_td, tempfile.TemporaryDirectory() as out_td:
        cwd = Path(cwd_td)
        output_dir = Path(out_td)
        # Create a minimal plan file in output_dir (not cwd).
        (output_dir / "task_plan.md").write_text(
            "## Current Phase\n**Status:** complete\n", encoding="utf-8"
        )

        old_cwd = os.getcwd()
        os.chdir(cwd)
        try:
            result, code = check_complete_mod.check_complete(output_dir=output_dir)
        finally:
            os.chdir(old_cwd)

        assert code == 0, f"unexpected exit code: {code}"
        # Should find the plan, not report no_plan.
        assert result["status"] != "no_plan", f"did not find plan: {result}"
        print("  check_complete reads from output_dir OK")


def test_session_catchup_reads_from_output_dir():
    """analyze_session must look for planning files in output_dir, not cwd."""
    with tempfile.TemporaryDirectory() as cwd_td, tempfile.TemporaryDirectory() as out_td:
        cwd = Path(cwd_td)
        output_dir = Path(out_td)
        (output_dir / "task_plan.md").write_text(
            "## Current Phase\nPhase 1\n**Status:** in_progress\n", encoding="utf-8"
        )
        (output_dir / "findings.md").write_text("# Findings\n", encoding="utf-8")
        (output_dir / "progress.md").write_text("# Progress\n", encoding="utf-8")

        old_cwd = os.getcwd()
        os.chdir(cwd)
        try:
            result, code = session_catchup_mod.analyze_session(output_dir=output_dir)
        finally:
            os.chdir(old_cwd)

        assert code == 0, f"unexpected exit code: {code}"
        assert result["status"] == "active_session", f"unexpected status: {result}"
        print("  session_catchup reads from output_dir OK")


if __name__ == "__main__":
    print("=== file-based-planning scripts ===")
    test_init_planning_writes_to_output_dir()
    test_check_complete_reads_from_output_dir()
    test_session_catchup_reads_from_output_dir()
    print("\n✅ All file-based-planning script tests passed!")
