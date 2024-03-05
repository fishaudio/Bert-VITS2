from typing import Any
import socket

from .worker_common import RequestType, receive_data, send_data


class WorkerClient:
    def __init__(self, port: int) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 5: timeout
        sock.settimeout(5)
        sock.connect((socket.gethostname(), port))
        self.sock = sock

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.sock.close()

    def dispatch_pyopenjtalk(self, func: str, *args, **kwargs):
        data = {
            "request-type": RequestType.PYOPENJTALK,
            "func": func,
            "args": args,
            "kwargs": kwargs,
        }
        send_data(self.sock, data)
        return receive_data(self.sock).get("return")

    def status(self):
        send_data(self.sock, {"request-type": RequestType.STATUS})
        return receive_data(self.sock)

    def quit_server(self):
        send_data(self.sock, {"request-type": RequestType.QUIT_SERVER})
        receive_data(self.sock)
