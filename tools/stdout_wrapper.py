import sys


class StdoutWrapper:
    def write(self, message: str):
        print(message, end="")

    def flush(self):
        pass


def get_stdout():
    # Colab 環境をチェックする
    if "google.colab" in sys.modules:
        return StdoutWrapper()
    else:
        return sys.stdout
