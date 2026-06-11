"""
UCGA Second Brain - File Writer Tool

Provides sandboxed file-writing capability with filename sanitization
and automatic output directory creation.

Author: Aman Singh
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

# Only allow alphanumeric characters, underscores, dots, and hyphens
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_.\-]")


class FileWriter:
    """Writes content to files within a designated output directory.

    Filenames are sanitized to prevent path-traversal attacks and
    restricted to alphanumeric characters, underscores, dots, and
    hyphens. The output directory is created automatically if it
    does not already exist.
    """

    def __init__(self, output_dir: str) -> None:
        """Initialise the FileWriter with a base output directory.

        Args:
            output_dir: Absolute or relative path to the directory
                where files will be written.
        """
        self.output_dir: str = output_dir
        logger.info("FileWriter initialised with output_dir: %s", output_dir)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize a filename by removing unsafe characters.

        Strips path separators, ".." sequences, and any character that
        is not alphanumeric, underscore, dot, or hyphen.

        Args:
            filename: The raw filename to sanitize.

        Returns:
            The sanitized filename string.
        """
        # Remove any directory components
        filename = os.path.basename(filename)

        # Remove ".." to prevent directory traversal
        filename = filename.replace("..", "")

        # Strip everything except allowed characters
        filename = _SAFE_FILENAME_RE.sub("", filename)

        # Guard against empty result after sanitization
        if not filename:
            filename = "untitled"

        return filename

    def write(self, filename: str, content: str) -> str:
        """Write content to a file inside the output directory.

        Args:
            filename: Desired filename (will be sanitized).
            content: The text content to write.

        Returns:
            The absolute path of the written file, or an error message
            if the operation fails.
        """
        try:
            safe_name: str = self._sanitize_filename(filename)
            logger.info(
                "Writing file: %s (sanitized from %r)", safe_name, filename
            )

            # Ensure the output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            filepath: str = os.path.join(self.output_dir, safe_name)
            abs_path: str = os.path.abspath(filepath)

            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(content)

            logger.info("File written successfully: %s", abs_path)
            return abs_path

        except Exception as exc:
            error_msg = f"Error: {type(exc).__name__}: {str(exc)}"
            logger.error("Failed to write file %r: %s", filename, error_msg)
            return error_msg
