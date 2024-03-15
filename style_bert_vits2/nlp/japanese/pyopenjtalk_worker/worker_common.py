import json
import socket
from enum import IntEnum, auto
from typing import Any, Final


WORKER_PORT: Final[int] = 7861
HEADER_SIZE: Final[int] = 4


class RequestType(IntEnum):
    STATUS = auto()
    QUIT_SERVER = auto()
    PYOPENJTALK = auto()


class ConnectionClosedException(Exception):
    pass


# socket communication


def send_data(sock: socket.socket, data: dict[str, Any]):
    json_data = json.dumps(data).encode()
    header = len(json_data).to_bytes(HEADER_SIZE, byteorder="big")
    sock.sendall(header + json_data)


def __receive_until(sock: socket.socket, size: int):
    data = b""
    while len(data) < size:
        part = sock.recv(size - len(data))
        if part == b"":
            raise ConnectionClosedException("接続が閉じられました")
        data += part

    return data


def receive_data(sock: socket.socket) -> dict[str, Any]:
    header = __receive_until(sock, HEADER_SIZE)
    data_length = int.from_bytes(header, byteorder="big")
    body = __receive_until(sock, data_length)
    return json.loads(body.decode())
