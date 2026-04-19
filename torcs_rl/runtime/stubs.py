from .base import TorcsRuntimeBackend


class LinuxTorcsRuntimeBackend(TorcsRuntimeBackend):
    name = "linux"

    def create_client(self, *, port: int, vision: bool, torcs_exe: str, launch_log: str, race_config: str):
        raise NotImplementedError(
            "Linux runtime backend is not implemented yet. Implement TorcsRuntimeBackend.create_client for linux."
        )


class DockerTorcsRuntimeBackend(TorcsRuntimeBackend):
    name = "docker"

    def create_client(self, *, port: int, vision: bool, torcs_exe: str, launch_log: str, race_config: str):
        raise NotImplementedError(
            "Docker runtime backend is not implemented yet. Implement TorcsRuntimeBackend.create_client for docker."
        )
