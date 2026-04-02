"""
agent/apply_patch_tool.py — Wraps apply_patch.py as a Tool the model can call.
"""

from typing import AsyncIterator
from uuid import uuid4

from openai_harmony import Author, Message, Role, TextContent

from .tool import Tool
from .apply_patch import apply_patch, DiffError


class ApplyPatchTool(Tool):
    """
    Tool for applying structured diffs to files.
    The model sends a patch block starting with *** Begin Patch and ending with *** End Patch.
    """

    @property
    def name(self) -> str:
        return "apply_patch"

    def instruction(self) -> str:
        return """\
Use this tool to edit existing files or create new files using a structured diff format.

Send a patch block in the following format:
*** Begin Patch
*** Add File: path/to/new_file.py
+line 1
+line 2
*** Update File: path/to/existing_file.py
@@ def my_function():
-    old_line
+    new_line
*** Delete File: path/to/obsolete_file.py
*** End Patch

Rules:
- Use *** Add File: to create new files (all lines prefixed with +)
- Use *** Update File: to edit existing files (@@-anchored hunks with - for removed, + for added)
- Use *** Delete File: to remove files
- Context lines in hunks have a leading space
- Paths are relative to the current working directory or absolute
"""

    def _make_response(self, text: str, channel: str | None = None) -> Message:
        return Message(
            id=uuid4(),
            author=Author(role=Role.TOOL, name=self.name),
            content=[TextContent(text=text)],
            channel=channel,
        ).with_recipient("assistant")

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        patch_text = message.content[0].text.strip() if message.content else ""
        channel = message.channel

        try:
            result = apply_patch(patch_text)
            yield self._make_response(f"[OK] {result}", channel=channel)
        except DiffError as e:
            yield self._make_response(f"[PATCH ERROR] {e}", channel=channel)
        except Exception as e:
            yield self._make_response(f"[ERROR] {type(e).__name__}: {e}", channel=channel)
