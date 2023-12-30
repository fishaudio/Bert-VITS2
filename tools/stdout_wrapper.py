import sys
import tempfile


class StdoutWrapper:
    def __init__(self):
        # 一時ファイルの作成とオープン
        self.temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)

    def write(self, message: str):
        # メッセージを一時ファイルに書き込む
        self.temp_file.write(message)
        self.temp_file.flush()

    def flush(self):
        # 一時ファイルのフラッシュ
        self.temp_file.flush()

    def read(self):
        # ファイルの内容を読み出す
        self.temp_file.seek(0)  # ファイルの先頭にシーク
        return self.temp_file.read()

    def close(self):
        # 一時ファイルを閉じる
        self.temp_file.close()

    def fileno(self):
        # 一時ファイルのファイルディスクリプタを返す
        return self.temp_file.fileno()


def get_stdout():
    # Colab 環境をチェックする
    if "google.colab" in sys.modules:
        return StdoutWrapper()
    else:
        return sys.stdout
