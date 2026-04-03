
import contextlib
import os
import re
import threading
import time
import queue

from openai_harmony import Message, Role, TextContent, ToolNamespaceConfig, Author
from jupyter_client import KernelManager

class Sandbox:

    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:

        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count

            return ports

    def __init__(self, timeout: float):

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None
        
        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def _format_error(self, traceback: list[str]) -> str:

        clean_lines = []

        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)

            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue

            clean_lines.append(clean_frame)

        return ''.join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:

        client = self._client
        effective_timeout = timeout or self._default_timeout
        
        msg_id = client.execute(
            code, 
            store_history=True, 
            allow_stdin=False, 
            stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > effective_timeout:
                self._km.interrupt_kernel()

                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

            try:
                msg = client.get_iopub_msg(timeout=1.0)

            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')

                if content.get('name') == 'stdout':
                    stdout_parts.append(text)

                else:
                    stderr_parts.append(text)

            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])

                stderr_parts.append(self._format_error(traceback_list))

            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')

                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')

            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr

        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def close(self):

        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self):
        
        self.execute(
            '%reset -f\n'
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def __del__(self):

        self.close()

class Tool:

    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):

        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        
        self._owns_session = sandbox is None
        
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

        self._execution_backend = "dangerously_use_local_jupyter" if sandbox is None else "provided_sandbox"

    def _ensure_session(self):

        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:

        lines = code.strip().split('\n')

        if not lines:
            return code

        last_line = lines[-1].strip()

        if 'print' in last_line or 'import' in last_line:
            return code

        if not last_line:
            return code

        if last_line.startswith('#'):
            return code

        lines[-1] = 'print(' + last_line + ')'

        return '\n'.join(lines)

    @property
    def name(self) -> str:
        return "python"

    @property
    def instruction(self) -> str:
        if self._execution_backend == "dangerously_use_local_jupyter":
            return """
Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. Internet access for this session is UNKNOWN. Depends on the cluster.
            """.strip()

        return """
Use this tool to execute STATELESS Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you. You have to use print statements to access the output.

IMPORTANT: Your python environment is not shared between calls. You will have to pass your entire code each time.
        """.strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:

        return ToolNamespaceConfig(
            name='python', 
            description=self.instruction, 
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:

        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')

        if channel:
            message = message.with_channel(channel)

        return message

    def process_sync_plus(self, message: Message) -> list[Message]:

        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)

            except TimeoutError as exc:
                output = f'[ERROR] {exc}'

        return [self._make_response(output, channel=message.channel)]

    async def process(self, message: Message):
        import asyncio
        # Run synchronous Jupyter execution in a thread pool to avoid blocking the FastAPI event loop.
        # We do NOT call get_event_loop() inside the thread — process_sync_plus is purely synchronous.
        try:
            responses = await asyncio.to_thread(self.process_sync_plus, message)
            for resp in responses:
                yield resp
        except Exception as e:
            yield self._make_response(f'[ERROR] Python execution failed: {e}', channel=message.channel)
