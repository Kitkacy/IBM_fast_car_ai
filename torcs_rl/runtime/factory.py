from .stubs import DockerTorcsRuntimeBackend, LinuxTorcsRuntimeBackend
from .native import NativeTorcsRuntimeBackend


def create_runtime_backend(target: str):
    key = str(target).lower()
    if key == "native":
        return NativeTorcsRuntimeBackend()
    if key == "linux":
        return LinuxTorcsRuntimeBackend()
    if key == "docker":
        return DockerTorcsRuntimeBackend()
    raise ValueError(f"Unsupported runtime target: {target}")
