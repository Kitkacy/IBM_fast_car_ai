"""TORCS SCR client — manages a TORCS process and UDP connection."""

import socket
import subprocess
import sys
import time
from pathlib import Path

from .constants import data_size
from .state import DriverAction, ServerState


def _write_log(log_file, message):
    if not log_file:
        return
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _find_udp_port_pids(port):
    try:
        completed = subprocess.run(
            ["netstat", "-ano", "-p", "udp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return set()

    port_suffix = f":{int(port)}"
    pids = set()
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0].upper() == "UDP":
            if parts[1].endswith(port_suffix) and parts[3].isdigit():
                pids.add(int(parts[3]))
    return pids


def _kill_pid(pid, log_file=None, port=None):
    if pid is None:
        return
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return
    subprocess.run(
        ["taskkill", "/PID", str(pid_int), "/F", "/T"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if log_file:
        _write_log(log_file, f"killed pid={pid_int} port={port}")


class Client:
    """Manages a single TORCS process and its UDP SCR connection.

    Launches TORCS on construction, connects via UDP, and provides
    methods to interact with the SCR server across episodes.
    """

    def __init__(self, *, torcs_exe, port=3001, race_config="", gui=False, launch_log="torcs_launcher.log"):
        self.port = int(port)
        self.host = "localhost"
        self.gui = bool(gui)
        self.torcs_exe = str(Path(torcs_exe).expanduser().resolve())
        self.race_config = str(Path(race_config).expanduser().resolve()) if race_config else ""
        self.log_file = str(Path(launch_log).expanduser()) if launch_log else ""

        self.torcs_pid = None
        self.so = None
        self.did_crash = False
        self.S = ServerState()
        self.R = DriverAction()

        self._launch_torcs()
        try:
            self._connect()
        except Exception:
            self._kill_torcs()
            raise

    # -- process management ------------------------------------------------

    def _launch_torcs(self):
        torcs_path = Path(self.torcs_exe)
        if not torcs_path.exists():
            raise FileNotFoundError(f"TORCS executable not found: {torcs_path}")

        has_race_config = bool(self.race_config)
        if has_race_config:
            race_path = Path(self.race_config)
            if not race_path.exists():
                raise FileNotFoundError(f"Race config not found: {race_path}")

        _write_log(self.log_file, "begin launch")
        _write_log(self.log_file, f"exe={torcs_path}")
        _write_log(self.log_file, f"port={self.port}")
        _write_log(self.log_file, f"race_config={self.race_config or '(none)'}")
        _write_log(self.log_file, f"gui={self.gui}")

        # kill any stale process holding the port
        for stale_pid in _find_udp_port_pids(self.port):
            _write_log(self.log_file, f"stale UDP owner pid={stale_pid} on port={self.port}; killing")
            _kill_pid(stale_pid, log_file=self.log_file, port=self.port)

        if not has_race_config:
            time.sleep(0.7)

        args = [str(torcs_path), "-nofuel", "-nodamage", "-nolaptime"]
        if has_race_config:
            args.extend(["-r", str(self.race_config)])

        popen_kwargs = {
            "args": args,
            "cwd": str(torcs_path.parent),
        }
        if self.gui:
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NEW_CONSOLE
            )
        else:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0
            popen_kwargs["startupinfo"] = si
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        proc = subprocess.Popen(**popen_kwargs)
        _write_log(self.log_file, f"launched pid={proc.pid}")
        time.sleep(0.5)

        if proc.poll() is not None:
            # first attempt failed — maybe a stale process grabbed the port
            _write_log(self.log_file, f"early exit pid={proc.pid} code={proc.returncode}")
            for stale_pid in _find_udp_port_pids(self.port):
                if stale_pid != proc.pid:
                    _kill_pid(stale_pid, log_file=self.log_file, port=self.port)
                    break
            proc = subprocess.Popen(**popen_kwargs)
            _write_log(self.log_file, f"re-launched pid={proc.pid}")
            time.sleep(0.5)
            if proc.poll() is not None:
                raise RuntimeError(
                    f"TORCS exited immediately (pid={proc.pid}, code={proc.returncode}). "
                    f"Check {self.log_file}"
                )

        self.torcs_pid = proc.pid
        _write_log(self.log_file, "launch ok")

    def _kill_torcs(self):
        _kill_pid(self.torcs_pid, log_file=self.log_file, port=self.port)
        self.torcs_pid = None

    # -- UDP connection ----------------------------------------------------

    def _connect(self):
        """Create a UDP socket and perform the SCR init handshake."""
        self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.so.settimeout(1)

        max_failures = 6000 if self.gui else 60
        failures = 0
        initmsg = "SCR(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45)"

        while True:
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                self.so.close()
                self.so = None
                raise ConnectionError(f"Failed to send init to port {self.port}")

            try:
                sockdata = self.so.recvfrom(data_size)[0].decode("utf-8")
            except (socket.timeout, ConnectionResetError):
                failures += 1
                if failures >= max_failures:
                    self.so.close()
                    self.so = None
                    raise TimeoutError(
                        f"TORCS server did not respond on port {self.port}. "
                        "Another TORCS instance may own the port, or the SCR server is unreachable."
                    )
                continue

            if "***identified***" in sockdata:
                print(f"Client connected on {self.port}")
                break

    def _close_socket(self):
        if self.so:
            self.so.close()
            self.so = None

    def _recover(self):
        """Relaunch TORCS and reconnect after a crash or unexpected shutdown."""
        _write_log(self.log_file, "attempting auto-recovery")
        self._close_socket()
        try:
            self._kill_torcs()
            self._launch_torcs()
            self._connect()
            self.did_crash = False
            _write_log(self.log_file, "auto-recovery succeeded")
            return True
        except Exception as exc:
            _write_log(self.log_file, f"auto-recovery failed: {exc}")
            self.did_crash = True
            return False

    # -- episode lifecycle -------------------------------------------------

    def reset_episode(self):
        """End the current race and start a new one.

        TORCS is relaunched for each episode because the SCR server
        (launched with ``-r``) does not support the meta-restart protocol.
        """
        self._recover()

    # -- communication -----------------------------------------------------

    def get_servers_input(self):
        """Read one sensor frame from the SCR server.

        If the server sends ``***shutdown***`` or ``***restart***``, an
        automatic recovery is attempted: TORCS is relaunched, a fresh
        connection is established, and the next sensor frame is returned
        transparently.  If recovery fails, ``self.did_crash`` is set to
        ``True`` and the method returns without parsing data.
        """
        if not self.so:
            return

        while True:
            try:
                sockdata = self.so.recvfrom(data_size)[0].decode("utf-8")
            except socket.timeout:
                print(".", end=" ")
                continue
            except ConnectionResetError:
                # TORCS process likely died — attempt recovery
                print(f"Connection lost on {self.port} — attempting auto-restart")
                if self._recover():
                    continue
                return

            if "***identified***" in sockdata:
                continue

            if "***shutdown***" in sockdata or "***restart***" in sockdata:
                kind = "shutdown" if "shutdown" in sockdata else "restart"
                print(f"Server {kind} on {self.port} — attempting auto-restart")
                if self._recover():
                    continue  # re-read from the fresh connection
                return        # recovery failed, did_crash is set

            if not sockdata:
                continue

            self.S.parse_server_str(sockdata)
            break

    def respond_to_server(self):
        """Send the current action to the SCR server."""
        if not self.so:
            return
        try:
            self.so.sendto(repr(self.R).encode(), (self.host, self.port))
        except socket.error as exc:
            print(f"Error sending to server: {exc}")

    # -- cleanup -----------------------------------------------------------

    def close(self):
        """Kill the TORCS process and close the socket."""
        self._close_socket()
        self._kill_torcs()
