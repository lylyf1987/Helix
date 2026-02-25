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
        "--model-name",
        default=None,
        help="Provider model name override.",
    )
    parser.add_argument(
        "--vision-provider",
        "--vision_provider",
        dest="vision_provider",
        default="none",
        help="Vision provider for image-understanding skill (default: none).",
    )
    parser.add_argument(
        "--vision-model",
        "--vision_model",
        dest="vision_model",
        default="none",
        help="Vision model for image-understanding skill (default: none).",
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
        model_name=args.model_name,
        vision_provider=args.vision_provider,
        vision_model=args.vision_model,
    )
    return runtime.start()


if __name__ == "__main__":
    raise SystemExit(main())
