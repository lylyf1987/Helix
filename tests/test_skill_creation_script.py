"""Verification tests for the built-in skill-creation scaffold script."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = (
    ROOT
    / "helix"
    / "builtin_skills"
    / "all-agents"
    / "skill-creation"
    / "scripts"
    / "skill_creation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("skill_creation_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


skill_creation = _load_module()


def test_parse_args_defaults_to_multi():
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "skill_creation.py",
            "--action", "scaffold",
            "--skill-id", "demo-skill",
            "--scope", "all-agents",
        ]
        args = skill_creation.parse_args()
        assert args.script_mode == "multi"
        print("  parse_args default script_mode=multi OK")
    finally:
        sys.argv = old_argv


def test_multi_scaffold_creates_phase_scripts():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        out = skill_creation.run_scaffold(
            workspace=workspace,
            skill_id="demo-skill",
            scope="all-agents",
            description="Demo multi-step skill",
            overwrite=False,
            script_mode="multi",
            handler="",
            dependencies=[],
        )
        assert out["status"] == "ok"

        skill_dir = workspace / "skills" / "all-agents" / "demo-skill"
        expected_files = [
            skill_dir / "SKILL.md",
            skill_dir / "scripts" / "gather_context.py",
            skill_dir / "scripts" / "execute_step.py",
            skill_dir / "scripts" / "verify_result.py",
            skill_dir / "scripts" / "README.md",
        ]
        for path in expected_files:
            assert path.exists(), f"missing scaffolded file: {path}"

        skill_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "script_mode: `multi`" in skill_text
        assert "scripts/gather_context.py" in skill_text
        assert "scripts/execute_step.py" in skill_text
        assert "scripts/verify_result.py" in skill_text

        validate = skill_creation.run_validate(
            workspace=workspace,
            skill_id="demo-skill",
            scope="all-agents",
        )
        assert validate["status"] == "ok"
        print("  multi scaffold creates phase scripts OK")


def test_validate_multi_requires_multiple_scripts():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        skill_dir = workspace / "skills" / "all-agents" / "partial-skill"
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            skill_creation._skill_template(
                skill_id="partial-skill",
                scope="all-agents",
                skill_name="Partial Skill",
                description="Partial multi-step skill",
                handler_path="",
                script_mode="multi",
                dependencies=[],
            ),
            encoding="utf-8",
        )

        (scripts_dir / "execute_step.py").write_text(
            skill_creation._script_template(
                "partial-skill",
                phase_name="execute-step",
                phase_purpose="Perform the main deterministic transformation.",
            ),
            encoding="utf-8",
        )

        out = skill_creation.run_validate(
            workspace=workspace,
            skill_id="partial-skill",
            scope="all-agents",
        )
        assert out["status"] == "error"
        assert "multi_mode_requires_multiple_scripts" in out["skill_created/updated"]
        print("  multi validate requires multiple scripts OK")


if __name__ == "__main__":
    print("=== Skill Creation Script ===")
    test_parse_args_defaults_to_multi()
    test_multi_scaffold_creates_phase_scripts()
    test_validate_multi_requires_multiple_scripts()
    print("\n✅ All skill-creation script tests passed!")
