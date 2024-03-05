"""
Run the pyopenjtalk worker in a separate process
to avoid user dictionary access error
"""

from typing import Optional, Any

from .worker_common import WOKER_PORT
from .worker_client import WorkerClient

from common.log import logger

WORKER_CLIENT: Optional[WorkerClient] = None

# pyopenjtalk interface

# g2p: not used


def run_frontend(text: str) -> list[dict[str, Any]]:
    assert WORKER_CLIENT
    ret = WORKER_CLIENT.dispatch_pyopenjtalk("run_frontend", text)
    assert isinstance(ret, list)
    return ret


def make_label(njd_features) -> list[str]:
    assert WORKER_CLIENT
    ret = WORKER_CLIENT.dispatch_pyopenjtalk("make_label", njd_features)
    assert isinstance(ret, list)
    return ret


def mecab_dict_index(path: str, out_path: str, dn_mecab: Optional[str] = None):
    assert WORKER_CLIENT
    WORKER_CLIENT.dispatch_pyopenjtalk("mecab_dict_index", path, out_path, dn_mecab)


def update_global_jtalk_with_user_dict(path: str):
    assert WORKER_CLIENT
    WORKER_CLIENT.dispatch_pyopenjtalk("update_global_jtalk_with_user_dict", path)


def unset_user_dict():
    assert WORKER_CLIENT
    WORKER_CLIENT.dispatch_pyopenjtalk("unset_user_dict")


# initialize module when imported


def initialize(port: int = WOKER_PORT):
    import time
    import socket
    import sys
    import atexit

    global WORKER_CLIENT
    logger.debug("initialize")
    if WORKER_CLIENT:
        return

    client = None
    try:
        client = WorkerClient(port)
    except (socket.timeout, socket.error):
        logger.debug("try starting pyopenjtalk worker server")
        import os
        import subprocess

        worker_pkg_path = os.path.relpath(
            os.path.dirname(__file__), os.getcwd()
        ).replace(os.sep, ".")
        subprocess.Popen([sys.executable, "-m", worker_pkg_path, "--port", str(port)])
        # wait until server listening
        count = 0
        while True:
            try:
                client = WorkerClient(port)
                break
            except socket.error:
                time.sleep(1)
                count += 1
                # 10: max number of retries
                if count == 10:
                    raise TimeoutError("サーバーに接続できませんでした")

    WORKER_CLIENT = client

    def terminate():
        global WORKER_CLIENT
        if not WORKER_CLIENT:
            return

        if WORKER_CLIENT.status().get("client-count") == 1:
            WORKER_CLIENT.quit_server()
        WORKER_CLIENT.close()
        WORKER_CLIENT = None

    atexit.register(terminate)
