from __future__ import annotations

import argparse
from pathlib import Path

from .runtime import AgentRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic System (clean runtime kernel)")
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "lmstudio", "zai", "deepseek", "openai_compatible"],
        help="LLM provider. Implemented: ollama, lmstudio, zai, deepseek, openai_compatible.",
    )
    parser.add_argument(
        "--mode",
        default="controlled",
        choices=["auto", "controlled"],
        help="auto: execute without confirmation; controlled: ask confirmation before each exec",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--workspace",
        required=True,
        help="Runtime workspace path (absolute or relative).",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Core agent model name override.",
    )
    parser.add_argument(
        "--image-analysis-provider",
        "--image_analysis_provider",
        dest="image_analysis_provider",
        default="none",
        help="Image analysis provider for image-understanding skill (default: none).",
    )
    parser.add_argument(
        "--image-analysis-model",
        "--image_analysis_model",
        dest="image_analysis_model",
        default="none",
        help="Image analysis model for image-understanding skill (default: none).",
    )
    parser.add_argument(
        "--image-generation-provider",
        "--image_generation_provider",
        dest="image_generation_provider",
        default="none",
        help="Image generation provider for image-generation skill (default: none).",
    )
    parser.add_argument(
        "--image-generation-model",
        "--image_generation_model",
        dest="image_generation_model",
        default="none",
        help="Image generation model for image-generation skill (default: none).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()

    runtime = AgentRuntime(
        workspace=workspace,
        provider=args.provider,
        mode=args.mode,
        session_id=args.session_id,
        model_name=args.model,
        image_analysis_provider=args.image_analysis_provider,
        image_analysis_model=args.image_analysis_model,
        image_generation_provider=args.image_generation_provider,
        image_generation_model=args.image_generation_model,
    )
    return runtime.start()


if __name__ == "__main__":
    raise SystemExit(main())
