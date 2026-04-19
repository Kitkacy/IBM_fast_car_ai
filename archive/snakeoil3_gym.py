#!/usr/bin/python

from torcs_scr import (
    Client,
    DriverAction,
    PI,
    ServerState,
    bargraph,
    clip,
    data_size,
    destringify,
    drive_example,
    ophelp,
    usage,
    version,
)

__all__ = [
    "Client",
    "DriverAction",
    "PI",
    "ServerState",
    "bargraph",
    "clip",
    "data_size",
    "destringify",
    "drive_example",
    "ophelp",
    "usage",
    "version",
]


if __name__ == "__main__":
    client = Client(p=3001)
    for step in range(client.maxSteps, 0, -1):
        client.get_servers_input()
        drive_example(client)
        client.respond_to_server()
    client.shutdown()
