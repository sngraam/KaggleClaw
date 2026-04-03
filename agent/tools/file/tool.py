"""
agent/tools/file_tool.py — File system tool for the Kaggle working directory.
Sandboxed to /kaggle/working/ and /kaggle/input/ (read-only for input).
"""

import os
import shutil
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from openai_harmony import Author, Message, Role, TextContent

from ..tool import Tool

ALLOWED_WRITE_ROOTS = ["/kaggle/working", "/tmp"]
ALLOWED_READ_ROOTS = ["/kaggle/working", "/kaggle/input", "/tmp"]

# Fallback to local paths when not running on Kaggle
if not os.path.exists("/kaggle"):
    ALLOWED_WRITE_ROOTS = [".", "/tmp"]
    ALLOWED_READ_ROOTS = [".", "/tmp"]


def _resolve_and_check(path: str, allowed_roots: list[str]) -> Path:
    resolved = Path(path).resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(Path(root).resolve())
            return resolved
        except ValueError:
            continue
    raise PermissionError(
        f"Path '{path}' is outside allowed directories: {allowed_roots}"
    )


class FileTool(Tool):

    @property
    def name(self) -> str:
        return "file"

    def instruction(self) -> str:
        return """\
Use this tool to read, write, list, and manage files in the Kaggle working directory.

Available commands:
  read <path>              — Read the contents of a file
  write <path>\\n<content>  — Write content to a file (creates parent dirs)
  list [path]              — List files in a directory (default: /kaggle/working/)
  delete <path>            — Delete a file
  mkdir <path>             — Create a directory
  exists <path>            — Check if a file or directory exists
  move <src> <dst>         — Move/rename a file

Allowed write paths: /kaggle/working/, /tmp/
Allowed read paths:  /kaggle/working/, /kaggle/input/, /tmp/

Examples:
  read /kaggle/working/submission.csv
  write /kaggle/working/model.py\\nimport pandas as pd\\n...
  list /kaggle/working/
  delete /kaggle/working/run/tmp_script.py
"""

    def _make_response(self, text: str, channel: str | None = None) -> Message:
        return Message(
            id=uuid4(),
            author=Author(role=Role.TOOL, name=self.name),
            content=[TextContent(text=text)],
            channel=channel,
        ).with_recipient("assistant")

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        raw = message.content[0].text.strip() if message.content else ""
        channel = message.channel

        try:
            result = self._dispatch(raw)
        except PermissionError as e:
            result = f"[PERMISSION ERROR] {e}"
        except FileNotFoundError as e:
            result = f"[NOT FOUND] {e}"
        except Exception as e:
            result = f"[ERROR] {type(e).__name__}: {e}"

        yield self._make_response(result, channel=channel)

    def _dispatch(self, raw: str) -> str:
        if raw.startswith("read "):
            path = raw[5:].strip()
            p = _resolve_and_check(path, ALLOWED_READ_ROOTS)
            content = p.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            # truncate very large files
            if len(lines) > 500:
                preview = "\n".join(lines[:500])
                return f"{preview}\n\n[... truncated — {len(lines)} total lines. File at {path}]"
            return content

        elif raw.startswith("write "):
            # format: "write <path>\n<content>"
            rest = raw[6:]
            newline_idx = rest.find("\n")
            if newline_idx == -1:
                return "[ERROR] write command needs content after a newline: write <path>\\n<content>"
            path = rest[:newline_idx].strip()
            content = rest[newline_idx + 1:]
            p = _resolve_and_check(path, ALLOWED_WRITE_ROOTS)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"[OK] Written {len(content)} bytes to {path}"

        elif raw.startswith("list"):
            path = raw[4:].strip() or "/kaggle/working"
            if not os.path.exists("/kaggle"):
                path = path.replace("/kaggle/working", ".").replace("/kaggle/input", ".")
            p = _resolve_and_check(path, ALLOWED_READ_ROOTS)
            if not p.exists():
                return f"[NOT FOUND] Directory does not exist: {path}"
            entries = []
            for item in sorted(p.iterdir()):
                if item.is_dir():
                    count = sum(1 for _ in item.rglob("*"))
                    entries.append(f"📁 {item.name}/  ({count} items)")
                else:
                    size = item.stat().st_size
                    size_str = f"{size:,} bytes" if size < 1024 else f"{size//1024:,} KB"
                    entries.append(f"📄 {item.name}  [{size_str}]")
            return "\n".join(entries) if entries else "(empty directory)"

        elif raw.startswith("delete "):
            path = raw[7:].strip()
            p = _resolve_and_check(path, ALLOWED_WRITE_ROOTS)
            if p.is_dir():
                shutil.rmtree(p)
                return f"[OK] Deleted directory {path}"
            p.unlink()
            return f"[OK] Deleted {path}"

        elif raw.startswith("mkdir "):
            path = raw[6:].strip()
            p = _resolve_and_check(path, ALLOWED_WRITE_ROOTS)
            p.mkdir(parents=True, exist_ok=True)
            return f"[OK] Created directory {path}"

        elif raw.startswith("exists "):
            path = raw[7:].strip()
            # exists check doesn't need write perms
            p = Path(path).resolve()
            exists = p.exists()
            kind = "directory" if p.is_dir() else "file" if p.is_file() else "path"
            return f"{'✓' if exists else '✗'} {kind} {'exists' if exists else 'does not exist'}: {path}"

        elif raw.startswith("move "):
            parts = raw[5:].strip().split()
            if len(parts) < 2:
                return "[ERROR] Usage: move <src> <dst>"
            src, dst = parts[0], parts[1]
            sp = _resolve_and_check(src, ALLOWED_WRITE_ROOTS)
            dp = _resolve_and_check(dst, ALLOWED_WRITE_ROOTS)
            dp.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sp), str(dp))
            return f"[OK] Moved {src} → {dst}"

        else:
            return (
                f"[ERROR] Unknown command. Available: read, write, list, delete, mkdir, exists, move\n"
                f"Got: {raw[:80]}"
            )
