"""
Run the pyopenjtalk worker in a separate process
to avoid user dictionary access error
"""

from typing import Optional, Any

from .worker_common import WORKER_PORT
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


def initialize(port: int = WORKER_PORT):
    import time
    import socket
    import sys
    import atexit
    import signal

    logger.debug("initialize")
    global WORKER_CLIENT
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
        args = [sys.executable, "-m", worker_pkg_path, "--port", str(port)]
        # new session, new process group
        if sys.platform.startswith("win"):
            cf = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
            si = subprocess.STARTUPINFO()  # type: ignore
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore
            si.wShowWindow = subprocess.SW_HIDE  # type: ignore
            subprocess.Popen(args, creationflags=cf, startupinfo=si)
        else:
            # align with Windows behavior
            # start_new_session is same as specifying setsid in preexec_fn
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)  # type: ignore

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
    atexit.register(terminate)

    # when the process is killed
    def signal_handler(signum, frame):
        terminate()

    signal.signal(signal.SIGTERM, signal_handler)


# top-level declaration
def terminate():
    logger.debug("terminate")
    global WORKER_CLIENT
    if not WORKER_CLIENT:
        return

    # repare for unexpected errors
    try:
        if WORKER_CLIENT.status() == 1:
            WORKER_CLIENT.quit_server()
    except Exception as e:
        logger.error(e)

    WORKER_CLIENT.close()
    WORKER_CLIENT = None
