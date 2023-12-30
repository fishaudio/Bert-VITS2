import sys
import tempfile


class StdoutWrapper:
    def __init__(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.original_stdout = sys.stdout

    def write(self, message: str):
        self.temp_file.write(message)
        self.temp_file.flush()
        print(message, end="", file=self.original_stdout)

    def flush(self):
        self.temp_file.flush()

    def read(self):
        self.temp_file.seek(0)
        return self.temp_file.read()

    def close(self):
        self.temp_file.close()

    def fileno(self):
        return self.temp_file.fileno()


try:
    import google.colab

    SAFE_STDOUT = StdoutWrapper()
except ImportError:
    SAFE_STDOUT = sys.stdout
