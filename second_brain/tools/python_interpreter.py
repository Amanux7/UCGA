"""
UCGA Second Brain - Python Interpreter Tool

Provides safe, sandboxed Python code execution with restricted builtins,
stdout capture, and timeout protection.

Author: Aman Singh
"""

import io
import sys
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Restricted set of builtins allowed during code execution
_ALLOWED_BUILTINS = {
    "math": __import__("math"),
    "len": len,
    "range": range,
    "print": print,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "type": type,
    "isinstance": isinstance,
    "True": True,
    "False": False,
    "None": None,
}


class _ExecutionResult:
    """Container for threaded execution results."""

    def __init__(self) -> None:
        self.output: Optional[str] = None
        self.error: Optional[str] = None


class PythonInterpreter:
    """Executes Python code safely with restricted builtins and timeouts.

    Uses a restricted set of builtins to prevent dangerous operations,
    captures stdout, and enforces a maximum execution time of 10 seconds.
    """

    MAX_EXECUTION_TIME: int = 10  # seconds

    def execute(self, code: str) -> str:
        """Execute Python code in a sandboxed environment.

        Args:
            code: The Python source code string to execute.

        Returns:
            The captured stdout output, a success message if no output
            was produced, or an error message if execution failed.
        """
        logger.info("Executing code snippet (%d chars)", len(code))

        result = _ExecutionResult()

        def _run() -> None:
            stdout_capture = io.StringIO()
            restricted_globals = {"__builtins__": _ALLOWED_BUILTINS}

            try:
                old_stdout = sys.stdout
                sys.stdout = stdout_capture
                try:
                    exec(code, restricted_globals)  # noqa: S102
                finally:
                    sys.stdout = old_stdout

                output = stdout_capture.getvalue()
                if output.strip():
                    result.output = output
                else:
                    result.output = "Code executed successfully (no output)"

            except Exception as exc:
                # Restore stdout in case it wasn't restored above
                sys.stdout = sys.__stdout__
                result.error = f"Error: {type(exc).__name__}: {str(exc)}"
                logger.warning("Code execution error: %s", result.error)

        # Run in a thread with a timeout to guard against infinite loops
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.MAX_EXECUTION_TIME)

        if thread.is_alive():
            logger.warning(
                "Code execution timed out after %d seconds",
                self.MAX_EXECUTION_TIME,
            )
            return (
                f"Error: TimeoutError: Code execution exceeded "
                f"{self.MAX_EXECUTION_TIME} second limit"
            )

        if result.error is not None:
            return result.error

        return result.output or "Code executed successfully (no output)"
