from typing import Any, Optional, Protocol


class TorcsClientProtocol(Protocol):
    S: Any
    R: Any
    MAX_STEPS: float
    torcs_pid: Optional[int]

    def get_servers_input(self) -> None:
        ...

    def respond_to_server(self) -> None:
        ...
