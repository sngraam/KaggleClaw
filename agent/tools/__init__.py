"""
agent/tools/__init__.py — Tool registry for KaggleClaw agent.
"""

from .file_tool import FileTool

__all__ = ["FileTool", "get_all_tools"]


def get_all_tools(
    *,
    browser_backend=None,
    jupyter_connection_file: str | None = None,
    jupyter_timeout: float = 120.0,
):
    """
    Returns the full list of initialized tools for the agent.

    Args:
        browser_backend: An instance of ExaBackend or YouComBackend.
                         If None, browser tool is skipped.
        jupyter_connection_file: Path to Jupyter connection file for python tool.
                                 If None, uses docker backend.
        jupyter_timeout: Timeout in seconds for Jupyter kernel execution.
    """
    tools = []

    # 1. Browser tool
    if browser_backend is not None:
        from ..browser import SimpleBrowserTool
        tools.append(SimpleBrowserTool(backend=browser_backend))

    # 2. Python / Jupyter tool
    from ..python.tool import PythonTool
    if jupyter_connection_file:
        tools.append(PythonTool(
            execution_backend="dangerously_use_local_jupyter",
            local_jupyter_connection_file=jupyter_connection_file,
            local_jupyter_timeout=jupyter_timeout,
        ))
    else:
        # fallback to uv (faster than docker on Kaggle)
        tools.append(PythonTool(execution_backend="dangerously_use_uv"))

    # 3. File tool
    tools.append(FileTool())

    # 4. Apply patch tool
    from ..apply_patch import ApplyPatchTool
    tools.append(ApplyPatchTool())

    return tools
