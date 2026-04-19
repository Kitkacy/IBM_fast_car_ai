from pathlib import Path

from torcs_scr import Client
from torcs_scr.process_manager import kill_torcs_process, launch_torcs_process

from .base import TorcsRuntimeBackend


class NativeTorcsRuntimeBackend(TorcsRuntimeBackend):
    name = "native"

    @property
    def default_torcs_exe(self) -> str:
        return ""

    @property
    def default_race_config(self) -> str:
        return ""

    @property
    def default_autostart_script(self) -> str:
        return str(Path(__file__).resolve().parents[2] / "autostart_windows.ps1")

    def create_client(self, *, port: int, vision: bool, torcs_exe: str, launch_log: str, race_config: str):
        return Client(
            p=port,
            vision=vision,
            torcs_exe=torcs_exe,
            launch_log=launch_log,
            race_config=race_config,
            launch_on_start=True,
            parse_command_line=False,
        )

    def kill_process(self, pid, *, port=None, log_file=None) -> bool:
        return kill_torcs_process(pid, log_file=log_file, port=port)

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
        return launch_torcs_process(
            torcs_exe=torcs_exe,
            port=port,
            vision=vision,
            race_config=race_config,
            log_file=log_file,
            autostart_script=autostart_script,
            existing_pid=existing_pid,
        )
