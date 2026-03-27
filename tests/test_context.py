"""Phase 3 verification tests for providers, context loaders, and prompt builder."""

import json
import sys
import tempfile
from http.client import RemoteDisconnected
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.providers.ollama import OllamaProvider
from helix.providers.openai_compat import OpenAICompatProvider
from helix.core.agent import _load_skills as load_skills
from helix.core.agent import _load_knowledge_catalog as load_knowledge_catalog
from helix.core.agent import _build_system_prompt


# =========================================================================== #
# Provider tests (structural — no real LLM calls)
# =========================================================================== #


def test_ollama_provider_init():
    """Verify OllamaProvider initializes with correct defaults."""
    provider = OllamaProvider()
    assert provider.model == "llama3.1:8b"
    assert "11434" in provider.endpoint
    assert provider.timeout == 300
    print("  OllamaProvider init OK")


def test_ollama_provider_custom_init():
    """Verify OllamaProvider respects custom parameters."""
    provider = OllamaProvider(
        model="deepseek-r1:14b",
        base_url="http://myhost:8080",
        timeout=60,
        temperature=0.5,
    )
    assert provider.model == "deepseek-r1:14b"
    assert "myhost:8080" in provider.endpoint
    assert provider.timeout == 60
    print("  OllamaProvider custom init OK")


def test_openai_provider_init():
    """Verify OpenAICompatProvider initializes with correct defaults."""
    provider = OpenAICompatProvider()
    assert provider.model == "local-model"
    assert "/chat/completions" in provider.endpoint
    print("  OpenAICompatProvider init OK")


def test_openai_provider_presets():
    """Verify preset resolution for known providers."""
    dp = OpenAICompatProvider(provider="deepseek")
    assert "deepseek.com" in dp.endpoint
    assert dp.model == "deepseek-chat"

    zai = OpenAICompatProvider(provider="zai")
    assert "z.ai" in zai.endpoint

    lm = OpenAICompatProvider(provider="lmstudio", model="my-model")
    assert lm.model == "my-model"
    print("  OpenAICompatProvider presets OK")


def test_openai_provider_requires_api_key_for_zai():
    """Verify Z.AI fails fast with a clear message when the API key is missing."""
    with patch.dict("os.environ", {}, clear=True):
        provider = OpenAICompatProvider(provider="zai")
        try:
            provider.generate("hello")
            assert False, "Expected missing API key to raise"
        except RuntimeError as exc:
            assert "Missing API key" in str(exc)
            assert "ZAI_API_KEY" in str(exc)
    print("  OpenAICompatProvider missing Z.AI key OK")


def test_openai_provider_requires_api_key_for_deepseek():
    """Verify DeepSeek fails fast with a clear message when the API key is missing."""
    with patch.dict("os.environ", {}, clear=True):
        provider = OpenAICompatProvider(provider="deepseek")
        try:
            provider.generate("hello")
            assert False, "Expected missing API key to raise"
        except RuntimeError as exc:
            assert "Missing API key" in str(exc)
            assert "DEEPSEEK_API_KEY" in str(exc)
    print("  OpenAICompatProvider missing DeepSeek key OK")


def test_provider_satisfies_protocol():
    """Verify both providers have the generate() interface matching ModelProvider."""
    import inspect
    for cls in [OllamaProvider, OpenAICompatProvider]:
        assert hasattr(cls, "generate"), f"{cls.__name__} missing generate()"
        sig = inspect.signature(cls.generate)
        params = list(sig.parameters.keys())
        assert "prompt" in params, f"{cls.__name__}.generate() missing prompt param"
        assert "stream" in params, f"{cls.__name__}.generate() missing stream param"
        assert "chunk_callback" in params, f"{cls.__name__}.generate() missing chunk_callback param"
    print("  Protocol compliance OK")


class _MockHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_MockHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_openai_provider_stream_timeout_wrapped_as_runtime_error():
    provider = OpenAICompatProvider()

    with patch(
        "helix.providers.openai_compat.urlopen",
        side_effect=TimeoutError("read timed out"),
    ):
        try:
            provider.generate("hello", stream=True)
            assert False, "Expected streaming timeout to raise RuntimeError"
        except RuntimeError as exc:
            assert "openai_compatible network error" in str(exc)
            assert "read timed out" in str(exc)
    print("  OpenAICompatProvider stream timeout wrapping OK")


def test_ollama_provider_stream_disconnect_wrapped_as_runtime_error():
    provider = OllamaProvider()

    with patch(
        "helix.providers.ollama.urlopen",
        side_effect=RemoteDisconnected("closed"),
    ):
        try:
            provider.generate("hello", stream=True)
            assert False, "Expected stream disconnect to raise RuntimeError"
        except RuntimeError as exc:
            assert "Ollama network error" in str(exc)
            assert "closed" in str(exc)
    print("  OllamaProvider stream disconnect wrapping OK")


def test_openai_provider_non_stream_invalid_json_wrapped_as_runtime_error():
    provider = OpenAICompatProvider()

    with patch(
        "helix.providers._http.urlopen",
        return_value=_MockHTTPResponse(b"not-json"),
    ):
        try:
            provider.generate("hello", stream=False)
            assert False, "Expected invalid JSON response to raise RuntimeError"
        except RuntimeError as exc:
            assert "openai_compatible invalid JSON response" in str(exc)
    print("  OpenAICompatProvider invalid JSON wrapping OK")


# =========================================================================== #
# Skill loader tests
# =========================================================================== #


def _create_skill_tree(root: Path) -> None:
    """Create a test skill directory tree."""
    # all-agents/search-web/SKILL.md
    skill_dir = root / "all-agents" / "search-web"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: search-web\n"
        "description: Search the web for information\n"
        "handler: scripts/search.py\n"
        "required_tools: bash\n"
        "---\n"
        "Full instructions here...\n",
        encoding="utf-8",
    )

    # all-agents/code-review/SKILL.md
    skill_dir2 = root / "all-agents" / "code-review"
    skill_dir2.mkdir(parents=True)
    (skill_dir2 / "SKILL.md").write_text(
        "---\n"
        "name: code-review\n"
        "description: Review code quality\n"
        "handler: scripts/review.py\n"
        "recommended_tools: python\n"
        "---\n",
        encoding="utf-8",
    )

    # all-agents/load-skill/ — should be excluded (builtin)
    builtin_dir = root / "all-agents" / "load-skill"
    builtin_dir.mkdir(parents=True)
    (builtin_dir / "SKILL.md").write_text("---\nname: load-skill\n---\n", encoding="utf-8")


def test_skill_loader():
    """Test skill loading and filtering."""
    with tempfile.TemporaryDirectory() as td:
        skills_root = Path(td) / "skills"
        _create_skill_tree(skills_root)

        skills = load_skills(skills_root)
        assert len(skills) == 2, f"Expected 2 skills, got {len(skills)}"
        ids = {s["skill_id"] for s in skills}
        assert "search-web" in ids
        assert "code-review" in ids
        assert "load-skill" not in ids  # builtin excluded
        print("  Skill loader OK")


def test_skill_loader_empty():
    """Test skill loading from non-existent directory."""
    skills = load_skills(Path("/nonexistent/path"))
    assert skills == []
    print("  Skill loader (empty) OK")

def test_skill_helpers():
    """Test helper parsing used by the skill loader."""
    from helix.core.agent import _parse_csv, _parse_frontmatter

    assert len(_parse_csv("a, b, c")) == 3
    assert _parse_frontmatter("---\nname: demo\n---\nbody\n") == {"name": "demo"}
    print("  Skill helper parsing OK")


# =========================================================================== #
# Knowledge loader tests
# =========================================================================== #


def _create_knowledge_catalog(root: Path) -> None:
    """Create a test knowledge catalog."""
    index_dir = root / "index"
    index_dir.mkdir(parents=True)
    catalog = [
        {
            "title": "RL for LLM Post-Training",
            "summary": "Overview of reinforcement learning methods commonly used in LLM post-training.",
            "path": "knowledge/docs/rl-overview.md",
            "tags": ["rl", "llm"],
        },
        {
            "title": "Agent System Design",
            "summary": "Notes on runtime architecture, orchestration, and component boundaries.",
            "path": "knowledge/docs/agent-design.md",
            "tags": "design, architecture",  # String tags
        },
    ]
    (index_dir / "catalog.json").write_text(
        json.dumps(catalog, indent=2), encoding="utf-8"
    )


def test_knowledge_loader():
    """Test knowledge catalog loading."""
    with tempfile.TemporaryDirectory() as td:
        knowledge_root = Path(td)
        _create_knowledge_catalog(knowledge_root)

        catalog = load_knowledge_catalog(knowledge_root)
        assert len(catalog) == 2
        assert catalog[0]["title"] == "Agent System Design"  # sorted by title
        assert catalog[1]["title"] == "RL for LLM Post-Training"
        assert catalog[0]["summary"].startswith("Notes on runtime architecture")
        assert isinstance(catalog[0]["tags"], list)  # string tags normalized
        print("  Knowledge loader OK")


def test_knowledge_loader_empty():
    """Test knowledge loading from non-existent directory."""
    catalog = load_knowledge_catalog(Path("/nonexistent/path"))
    assert catalog == []
    print("  Knowledge loader (empty) OK")

def test_knowledge_helpers():
    """Test helper normalization used by the knowledge loader."""
    from helix.core.agent import _normalize_tags

    assert len(_normalize_tags("a, b, c")) == 3
    assert _normalize_tags(["a", " ", "b"]) == ["a", "b"]
    print("  Knowledge helper normalization OK")


# =========================================================================== #
# Prompt builder tests
# =========================================================================== #


def _create_workspace(root: Path) -> None:
    """Create a test workspace with skills and knowledge."""
    # Create skills
    _create_skill_tree(root / "skills")

    # Create knowledge
    _create_knowledge_catalog(root / "knowledge")


def test_prompt_builder():
    """Test full prompt assembly with placeholder injection."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        _create_workspace(workspace)
        session_root = workspace / "sessions" / "demo-01"
        project_root = session_root / "project"
        docs_root = session_root / "docs"
        state_root = session_root / ".state"

        prompt = _build_system_prompt(
            workspace,
            "core_agent",
            session_id="demo-01",
            session_root=session_root,
            project_root=project_root,
            docs_root=docs_root,
            state_root=state_root,
        )

        assert "Core Agent" in prompt
        assert "search-web" in prompt  # skill injected
        assert "rl-overview" in prompt  # knowledge injected
        assert "load-skill" in prompt  # builtin loader injected
        assert str(workspace) in prompt  # workspace path injected
        assert "demo-01" in prompt
        assert str(session_root) in prompt
        assert str(project_root) in prompt
        assert str(docs_root) in prompt
        assert str(state_root) in prompt
        assert "{{SKILLS_META_FROM_JSON}}" not in prompt  # placeholder replaced
        assert "{{KNOWLEDGE_META_FROM_JSON}}" not in prompt
        assert "{{BUILTIN_REFERENCE_LOADERS}}" not in prompt
        assert "{{WORKSPACE_ROOT}}" not in prompt
        assert "{{SESSION_ID}}" not in prompt
        assert "{{SESSION_ROOT}}" not in prompt
        assert "{{PROJECT_ROOT}}" not in prompt
        assert "{{DOCS_ROOT}}" not in prompt
        assert "{{STATE_ROOT}}" not in prompt
        assert "{{RUNTIME_WORKSPACE}}" not in prompt
        print("  Prompt builder OK")


def test_prompt_builder_unknown_role():
    """Test prompt builder returns empty for unknown role."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        _create_workspace(workspace)

        prompt = _build_system_prompt(workspace, "nonexistent_role")
        assert prompt == ""
        print("  Prompt builder (unknown role) OK")


def test_prompt_builder_no_prompts():
    """Test prompt builder with workspace that has no matching role."""
    with tempfile.TemporaryDirectory() as td:
        prompt = _build_system_prompt(Path(td), "nonexistent_role_xyz")
        assert prompt == ""
        print("  Prompt builder (no prompts) OK")


# =========================================================================== #
# Runner
# =========================================================================== #


if __name__ == "__main__":
    print("=== Provider Initialization ===")
    test_ollama_provider_init()
    test_ollama_provider_custom_init()
    test_openai_provider_init()
    test_openai_provider_presets()
    test_openai_provider_requires_api_key_for_zai()
    test_openai_provider_requires_api_key_for_deepseek()
    test_provider_satisfies_protocol()

    print("\n=== Skill Loader ===")
    test_skill_loader()
    test_skill_loader_empty()
    test_skill_helpers()

    print("\n=== Knowledge Loader ===")
    test_knowledge_loader()
    test_knowledge_loader_empty()
    test_knowledge_helpers()

    print("\n=== Prompt Builder ===")
    test_prompt_builder()
    test_prompt_builder_unknown_role()
    test_prompt_builder_no_prompts()

    print("\n✅ All Phase 3 tests passed!")
