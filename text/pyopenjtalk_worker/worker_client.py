from typing import Any
import socket

from .worker_common import RequestType, receive_data, send_data

from common.log import logger


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
        logger.trace(f"client sends request: {data}")
        send_data(self.sock, data)
        logger.trace("client sent request successfully")
        response = receive_data(self.sock)
        logger.trace(f"client received response: {response}")
        return response.get("return")

    def status(self):
        data = {"request-type": RequestType.STATUS}
        logger.trace(f"client sends request: {data}")
        send_data(self.sock, data)
        logger.trace("client sent request successfully")
        response = receive_data(self.sock)
        logger.trace(f"client received response: {response}")
        return response.get("client-count")

    def quit_server(self):
        data = {"request-type": RequestType.QUIT_SERVER}
        logger.trace(f"client sends request: {data}")
        send_data(self.sock, data)
        logger.trace("client sent request successfully")
        response = receive_data(self.sock)
        logger.trace(f"client received response: {response}")
