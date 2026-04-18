#!/usr/bin/python

import argparse
import subprocess
import time
from pathlib import Path


def _log_path(log_file):
    if log_file and str(log_file).strip():
        return Path(log_file).expanduser()
    return Path(__file__).resolve().parent / "torcs_launcher.log"


def _find_udp_port_owner_pids(port):
    try:
        completed = subprocess.run(
            ["netstat", "-ano", "-p", "udp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return None

    port_suffix = f":{int(port)}"
    pids = set()
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        proto, local_address, _foreign, pid = parts[0], parts[1], parts[2], parts[3]
        if proto.upper() != "UDP":
            continue
        if not local_address.endswith(port_suffix):
            continue
        if pid.isdigit():
            pids.add(int(pid))
    return pids


def write_launcher_log(log_file, message):
    log_path = _log_path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def kill_torcs_process(pid, log_file=None, port=None):
    if pid is None:
        return False

    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False

    subprocess.run(
        ["taskkill", "/PID", str(pid_int), "/F", "/T"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if log_file:
        suffix = f" port={port}" if port is not None else ""
        write_launcher_log(log_file, f"killed TORCS pid={pid_int}{suffix}")
    return True


def launch_torcs_process(
    torcs_exe,
    port=3001,
    vision=False,
    race_config="",
    log_file=None,
    autostart_script=None,
    existing_pid=None,
):
    torcs_path = Path(torcs_exe).expanduser().resolve()
    if not torcs_path.exists():
        raise FileNotFoundError(f"TORCS executable not found: {torcs_path}")

    if race_config and str(race_config).strip():
        race_config_path = Path(race_config).expanduser().resolve()
        if not race_config_path.exists():
            raise FileNotFoundError(f"Race config not found: {race_config_path}")

    has_race_config = bool(race_config and str(race_config).strip())

    write_launcher_log(log_file, "begin launcher")
    write_launcher_log(log_file, f"TORCS_EXE={torcs_path}")
    write_launcher_log(log_file, f"PORT={int(port)}")
    write_launcher_log(log_file, f"VISION_FLAG={'vision' if vision else 'novision'}")
    write_launcher_log(log_file, f"RACE_CONFIG={race_config}")

    if existing_pid is not None:
        kill_torcs_process(existing_pid, log_file=log_file, port=port)

    # Always clean up any process already bound to the target UDP port.
    # This prevents duplicate TORCS instances when caller PID tracking is stale.
    stale_pids = _find_udp_port_owner_pids(port)
    for stale_pid in stale_pids:
        if existing_pid is not None and int(stale_pid) == int(existing_pid):
            continue
        write_launcher_log(log_file, f"pre-launch cleanup: killing stale UDP owner pid={stale_pid} on port={port}")
        kill_torcs_process(stale_pid, log_file=log_file, port=port)

    if not has_race_config:
        time.sleep(0.7)

    args = [str(torcs_path), "-nofuel", "-nodamage", "-nolaptime"]
    if has_race_config:
        args.extend(["-r", str(race_config)])
    if vision:
        args.append("-vision")

    def _spawn_and_validate():
        proc_local = subprocess.Popen(
            args,
            cwd=str(torcs_path.parent),
            stdout=None,
            stderr=None,
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            ),
        )
        write_launcher_log(log_file, f"launch command dispatched pid={proc_local.pid}")

        # Ensure we do not report success for processes that die immediately.
        time.sleep(0.5)
        return proc_local, proc_local.poll()

    proc, exit_code = _spawn_and_validate()
    if exit_code is not None:
        write_launcher_log(log_file, f"launch failed early pid={proc.pid} exit_code={exit_code}")
        stale_pids = _find_udp_port_owner_pids(port)
        stale_pids.discard(proc.pid)
        if stale_pids:
            stale_pid = next(iter(stale_pids))
            write_launcher_log(log_file, f"detected stale UDP owner pid={stale_pid} on port={port}; retrying launch")
            kill_torcs_process(stale_pid, log_file=log_file, port=port)
            proc, exit_code = _spawn_and_validate()

    if exit_code is not None:
        write_launcher_log(log_file, f"launch failed after retry pid={proc.pid} exit_code={exit_code}")
        raise RuntimeError(
            f"TORCS exited immediately after launch (pid={proc.pid}, exit_code={exit_code}). "
            f"Check {_log_path(log_file)}"
        )

    if has_race_config:
        write_launcher_log(log_file, "race config specified: skipping autostart, focus, and waits")
    else:
        script_path = Path(autostart_script).expanduser() if autostart_script else None
        time.sleep(1.1)
        if script_path and script_path.exists():
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            write_launcher_log(log_file, f"autostart script finished: {script_path}")
        else:
            write_launcher_log(log_file, "autostart script missing or empty")

    write_launcher_log(log_file, "launcher completed")
    return int(proc.pid)


def _parse_args():
    parser = argparse.ArgumentParser(description="Launch/kill TORCS processes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch", help="Launch TORCS and print PID")
    launch_parser.add_argument("--torcs-exe", required=True)
    launch_parser.add_argument("--port", type=int, default=3001)
    launch_parser.add_argument("--vision-flag", choices=["vision", "novision"], default="novision")
    launch_parser.add_argument("--log-file", default="")
    launch_parser.add_argument("--race-config", default="")
    launch_parser.add_argument("--autostart-script", default="")
    launch_parser.add_argument("--existing-pid", type=int, default=None)

    kill_parser = subparsers.add_parser("kill", help="Kill TORCS by PID")
    kill_parser.add_argument("--pid", type=int, required=True)
    kill_parser.add_argument("--log-file", default="")
    kill_parser.add_argument("--port", type=int, default=None)

    return parser.parse_args()


def main():
    args = _parse_args()

    if args.command == "launch":
        pid = launch_torcs_process(
            torcs_exe=args.torcs_exe,
            port=args.port,
            vision=(args.vision_flag == "vision"),
            race_config=args.race_config,
            log_file=args.log_file,
            autostart_script=args.autostart_script,
            existing_pid=args.existing_pid,
        )
        print(f"TORCS_PID={pid}")
        return

    if args.command == "kill":
        kill_torcs_process(args.pid, log_file=args.log_file, port=args.port)


if __name__ == "__main__":
    main()
