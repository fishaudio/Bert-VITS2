import argparse

from .worker_server import WorkerServer
from .worker_common import WOKER_PORT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=WOKER_PORT)
    args = parser.parse_args()
    server = WorkerServer()
    server.start_server(port=args.port)


if __name__ == "__main__":
    main()
