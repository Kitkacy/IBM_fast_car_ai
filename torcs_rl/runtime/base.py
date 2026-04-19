from abc import ABC, abstractmethod

from .types import TorcsClientProtocol


class TorcsRuntimeBackend(ABC):
    """Platform runtime contract used by RL code to interact with TORCS."""

    name = "unknown"

    @property
    def default_torcs_exe(self) -> str:
        return ""

    @property
    def default_race_config(self) -> str:
        return ""

    @property
    def default_autostart_script(self) -> str:
        return ""

    @abstractmethod
    def create_client(self, *, port: int, vision: bool, torcs_exe: str, launch_log: str, race_config: str) -> TorcsClientProtocol:
        ...

    def kill_process(self, pid, *, port=None, log_file=None) -> bool:
        return False

    def launch_process(
        self,
        *,
        torcs_exe: str,
        port: int,
        vision: bool,
        race_config: str,
        log_file: str,
        autostart_script: str,
        existing_pid=None,
    ) -> int:
        raise NotImplementedError(f"{self.name} runtime does not implement explicit process launch")
