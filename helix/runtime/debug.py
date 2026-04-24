"""Session debugging views and HTML rendering."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path

from ..core.state import Turn


def open_file_in_viewer(path: Path) -> bool:
    """Best-effort open of a file in the platform's default viewer."""
    try:
        if sys.platform == "darwin":
            subprocess.run(
                ["open", str(path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        if sys.platform.startswith("linux"):
            subprocess.run(
                ["xdg-open", str(path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
    except Exception:
        return False
    return False


def render_session_view_html(
    *,
    session_id: str,
    field: str,
    session_path: Path,
    value: object,
) -> str:
    """Render a field-specific session JSON view as a readable HTML page."""
    def _render_text_view(*, eyebrow: str, body_text: str) -> str:
        title = escape(f"Session View - {session_id} - {field}")
        escaped_body = escape(body_text)
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f3efe6;
        --panel: rgba(255, 252, 245, 0.92);
        --border: rgba(46, 58, 89, 0.14);
        --text: #1f2430;
        --muted: #6a7280;
        --accent: #1e6aa8;
        --shadow: 0 20px 50px rgba(31, 36, 48, 0.10);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Menlo, Monaco, "SFMono-Regular", Consolas, monospace;
        background:
          radial-gradient(circle at top left, rgba(30, 106, 168, 0.12), transparent 34%),
          linear-gradient(180deg, #f8f5ee 0%, var(--bg) 100%);
        color: var(--text);
      }}
      main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 32px 24px 48px;
      }}
      .header {{
        margin-bottom: 20px;
      }}
      .eyebrow {{
        color: var(--accent);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 28px;
        line-height: 1.2;
      }}
      .sub {{
        margin: 0;
        color: var(--muted);
        font-size: 14px;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }}
      .panel-head {{
        padding: 16px 18px;
        border-bottom: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.72);
        font-size: 13px;
        color: var(--muted);
      }}
      pre {{
        margin: 0;
        padding: 22px 24px 28px;
        font-size: 13px;
        line-height: 1.55;
        overflow: auto;
        white-space: pre-wrap;
        word-break: break-word;
      }}
      .turn {{
        margin: 0 0 12px;
        padding: 10px 14px;
        border-radius: 8px;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 13px;
        line-height: 1.55;
      }}
      .turn-user {{
        background: rgba(46, 58, 89, 0.06);
        border-left: 3px solid #6a7280;
      }}
      .turn-core_agent {{
        background: transparent;
        border-left: 3px solid var(--accent);
      }}
      .turn-runtime {{
        background: rgba(46, 58, 89, 0.10);
        border-left: 3px solid #8b6914;
      }}
      .turn-sub_agent {{
        background: rgba(30, 106, 168, 0.06);
        border-left: 3px solid #5b8abf;
      }}
      .turn-role {{
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
        color: var(--muted);
      }}
    </style>
  </head>
  <body>
    <main>
      <div class="header">
        <div class="eyebrow">{escape(eyebrow)}</div>
        <h1>{escape(field)}</h1>
        <p class="sub">Session <strong>{escape(session_id)}</strong></p>
      </div>
      <section class="panel">
        <div class="panel-head">{escape(str(session_path))}</div>
        <pre>{escaped_body}</pre>
      </section>
    </main>
  </body>
</html>
"""

    if field == "last_prompt":
        if isinstance(value, list):
            # New format: list of message dicts with role/content
            parts: list[str] = []
            for msg in value:
                if isinstance(msg, dict):
                    role = str(msg.get("role", "unknown"))
                    content = str(msg.get("content", ""))
                    parts.append(f"=== {role.upper()} ===\n{content}")
            prompt_text = "\n\n".join(parts) if parts else "(none yet)"
        else:
            prompt_text = str(value) if str(value) else "(none yet)"
        return _render_text_view(
            eyebrow="Agentic System Prompt View",
            body_text=prompt_text,
        )

    if field == "workflow_summary":
        summary_text = str(value) if str(value) else "(empty)"
        return _render_text_view(
            eyebrow="Agentic System Workflow Summary View",
            body_text=summary_text,
        )

    if field in {"full_history", "observation"}:
        turns = value if isinstance(value, list) else []
        blocks: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "unknown") or "unknown")
            content = str(turn.get("content", "") or "")
            timestamp = str(turn.get("timestamp", "") or "")
            role_label = f"[{timestamp}] {role}" if timestamp else role
            css_class = f"turn turn-{escape(role)}"
            blocks.append(
                f'<div class="{css_class}">'
                f'<span class="badge">{escape(role_label)}</span>'
                f'{escape(content)}</div>'
            )
        body_html = "\n".join(blocks) if blocks else '<div class="empty">(empty)</div>'
        title = escape(f"Session View - {session_id} - {field}")
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Menlo, Monaco, "SFMono-Regular", Consolas, monospace;
        background: #1a1a2e;
        color: #e8e8e8;
      }}
      main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 32px 24px 48px;
      }}
      .header {{ margin-bottom: 24px; }}
      .eyebrow {{
        color: #5b9bd5;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
      }}
      h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.2; color: #ffffff; }}
      .sub {{ margin: 0; color: #888; font-size: 14px; }}
      .turn {{
        margin: 0 0 8px;
        padding: 12px 16px;
        border-radius: 6px;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 13px;
        line-height: 1.55;
        color: #ffffff;
      }}
      .badge {{
        display: inline-block;
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 2px 8px;
        border-radius: 3px;
        margin-bottom: 6px;
        color: #ffffff;
      }}
      .turn-user        {{ background: #585858; }}
      .turn-user .badge  {{ background: #6e6e6e; }}
      .turn-core_agent        {{ background: #005faf; }}
      .turn-core_agent .badge  {{ background: #0073cc; }}
      .turn-runtime        {{ background: #af5f00; }}
      .turn-runtime .badge  {{ background: #c87000; }}
      .turn-sub_agent        {{ background: #008700; }}
      .turn-sub_agent .badge  {{ background: #00a000; }}
      .empty {{ color: #888; padding: 20px; }}
    </style>
  </head>
  <body>
    <main>
      <div class="header">
        <div class="eyebrow">OpenHelix Timeline View</div>
        <h1>{escape(field)}</h1>
        <p class="sub">Session <strong>{escape(session_id)}</strong></p>
      </div>
      {body_html}
    </main>
  </body>
</html>
"""

    payload = {
        "session_id": session_id,
        "field": field,
        "source_session_file": str(session_path),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "value": value,
    }
    pretty_json = json.dumps(payload, indent=2, ensure_ascii=False)
    escaped_json = escape(pretty_json)
    title = escape(f"Session View - {session_id} - {field}")

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f3efe6;
        --panel: rgba(255, 252, 245, 0.92);
        --border: rgba(46, 58, 89, 0.14);
        --text: #1f2430;
        --muted: #6a7280;
        --accent: #1e6aa8;
        --shadow: 0 20px 50px rgba(31, 36, 48, 0.10);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Menlo, Monaco, "SFMono-Regular", Consolas, monospace;
        background:
          radial-gradient(circle at top left, rgba(30, 106, 168, 0.12), transparent 34%),
          linear-gradient(180deg, #f8f5ee 0%, var(--bg) 100%);
        color: var(--text);
      }}
      main {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 32px 24px 48px;
      }}
      .header {{
        margin-bottom: 20px;
      }}
      .eyebrow {{
        color: var(--accent);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 28px;
        line-height: 1.2;
      }}
      .sub {{
        margin: 0;
        color: var(--muted);
        font-size: 14px;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }}
      .panel-head {{
        padding: 16px 18px;
        border-bottom: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.72);
        font-size: 13px;
        color: var(--muted);
      }}
      pre {{
        margin: 0;
        padding: 22px 24px 28px;
        font-size: 13px;
        line-height: 1.55;
        overflow: auto;
        white-space: pre-wrap;
        word-break: break-word;
      }}
    </style>
  </head>
  <body>
    <main>
      <div class="header">
        <div class="eyebrow">Agentic System Session View</div>
        <h1>{escape(field)}</h1>
        <p class="sub">Session <strong>{escape(session_id)}</strong></p>
      </div>
      <section class="panel">
        <div class="panel-head">{escape(str(session_path))}</div>
        <pre>{escaped_json}</pre>
      </section>
    </main>
  </body>
</html>
"""
