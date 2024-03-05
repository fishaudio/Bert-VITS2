import pyopenjtalk
import socket
import select

from .worker_common import (
    ConnectionClosedException,
    RequestType,
    receive_data,
    send_data,
)

from common.log import logger

# To make it as fast as possible
# Probably faster than calling getattr every time
_PYOPENJTALK_FUNC_DICT = {
    "run_frontend": pyopenjtalk.run_frontend,
    "make_label": pyopenjtalk.make_label,
    "mecab_dict_index": pyopenjtalk.mecab_dict_index,
    "update_global_jtalk_with_user_dict": pyopenjtalk.update_global_jtalk_with_user_dict,
    "unset_user_dict": pyopenjtalk.unset_user_dict,
}


class WorkerServer:
    def __init__(self) -> None:
        self.client_count: int = 0
        self.quit: bool = False

    def handle_request(self, request):
        request_type = None
        try:
            request_type = RequestType(request.get("request-type"))
        except Exception:
            return {
                "success": False,
                "reason": "request-type is invalid",
            }

        if request_type:
            if request_type == RequestType.STATUS:
                response = {
                    "success": True,
                    "client-count": self.client_count,
                }
            elif request_type == RequestType.QUIT_SERVER:
                self.quit = True
                response = {"success": True}
            elif request_type == RequestType.PYOPENJTALK:
                func_name = request.get("func")
                assert isinstance(func_name, str)
                func = _PYOPENJTALK_FUNC_DICT[func_name]
                args = request.get("args")
                kwargs = request.get("kwargs")
                assert isinstance(args, list)
                assert isinstance(kwargs, dict)
                ret = func(*args, **kwargs)
                response = {"success": True, "return": ret}
            else:
                # NOT REACHED
                response = request

        return response

    def start_server(self, port: int):
        logger.info("start pyopenjtalk worker server")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((socket.gethostname(), port))
            server_socket.listen()
            sockets = [server_socket]
            while True:
                ready_sockets, _, _ = select.select(sockets, [], [], 0.1)
                for sock in ready_sockets:
                    if sock is server_socket:
                        logger.info("new client connected")
                        client_socket, _ = server_socket.accept()
                        sockets.append(client_socket)
                        self.client_count += 1
                    else:
                        # client
                        try:
                            request = receive_data(sock)
                        except ConnectionClosedException as e:
                            sock.close()
                            sockets.remove(sock)
                            self.client_count -= 1
                            logger.info("close connection")
                            continue

                        logger.trace(f"server received request: {request}")

                        response = self.handle_request(request)
                        logger.trace(f"server sends response: {response}")
                        try:
                            send_data(sock, response)
                            logger.trace("server sent response successfully")
                        except Exception:
                            logger.warning(
                                "an exception occurred during sending responce"
                            )
                        if self.quit:
                            logger.info("quit pyopenjtalk worker server")
                            return
