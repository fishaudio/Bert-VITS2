import sys
import tempfile
from typing import TextIO


class StdoutWrapper(TextIO):
    """
    `sys.stdout` wrapper for both Google Colab and local environment.
    """

    def __init__(self) -> None:
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8"
        )
        self.original_stdout = sys.stdout

    def write(self, message: str) -> int:
        result = self.temp_file.write(message)
        self.temp_file.flush()
        print(message, end="", file=self.original_stdout)
        return result

    def flush(self) -> None:
        self.temp_file.flush()

    def read(self, n: int = -1) -> str:
        self.temp_file.seek(0)
        return self.temp_file.read(n)

    def close(self) -> None:
        self.temp_file.close()

    def fileno(self) -> int:
        return self.temp_file.fileno()


try:
    import google.colab  # type: ignore

    SAFE_STDOUT = StdoutWrapper()
except ImportError:
    SAFE_STDOUT = sys.stdout
