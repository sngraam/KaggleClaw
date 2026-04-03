from .python.tool import Tool as PythonTool, Sandbox
from .file.tool import FileTool
from .apply_patch.tool import ApplyPatchTool
from .web_search.tool import WebSearchTool
from .plan_follow.tool import PlanFollowTool

__all__ = ['PythonTool', 'Sandbox', 'FileTool', 'ApplyPatchTool', 'WebSearchTool', 'PlanFollowTool']


def get_all_tools(sandbox=None):
    """Return all instantiated tools ready for use by AgentRunner."""
    python_tool = PythonTool(
        local_jupyter_timeout=120.0,
        tool_prompt=None,
        sandbox=sandbox or Sandbox(timeout=120.0),
    )
    return [
        python_tool,
        FileTool(),
        ApplyPatchTool(),
        WebSearchTool(),
        PlanFollowTool(),
    ]