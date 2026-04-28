import getopt
import os
import socket
import sys
import time
from pathlib import Path

from .process_manager import launch_torcs_process

from .constants import PI, data_size, usage, version
from .state import DriverAction, ServerState


class Client:
    def __init__(
        self,
        H=None,
        p=None,
        i=None,
        e=None,
        t=None,
        s=None,
        d=None,
        vision=False,
        torcs_exe=None,
        launch_log="torcs_launcher.log",
        race_config="",
        autostart_script="",
        launch_on_start=False,
        parse_command_line=False,
    ):
        self.vision = vision

        self.host = "localhost"
        self.port = 3001
        self.sid = "SCR"
        self.maxEpisodes = 1
        self.trackname = "unknown"
        self.stage = 3
        self.debug = False
        self.maxSteps = 100000
        self.launch_on_start = bool(launch_on_start)
        self.torcs_pid = None
        self.torcs_exe = str(Path(torcs_exe).expanduser()) if torcs_exe else ""
        self.launch_log = str(Path(launch_log).expanduser()) if launch_log else str(Path(__file__).resolve().parent.parent / "torcs_launcher.log")
        self.race_config = str(Path(race_config).expanduser()) if race_config else ""
        if autostart_script:
            self.autostart_script = str(Path(autostart_script).expanduser())
        else:
            self.autostart_script = str(Path(__file__).resolve().parent.parent / "autostart_windows.ps1")

        if parse_command_line:
            self.parse_the_command_line()

        if H:
            self.host = H
        if p:
            self.port = p
        if i:
            self.sid = i
        if e:
            self.maxEpisodes = e
        if t:
            self.trackname = t
        if s:
            self.stage = s
        if d:
            self.debug = d
        self.S = ServerState()
        self.R = DriverAction()

        if self.launch_on_start:
            self._launch_managed_torcs()

        self.setup_connection()

    def _launch_managed_torcs(self):
        if not self.torcs_exe:
            return

        self.torcs_pid = launch_torcs_process(
            torcs_exe=self.torcs_exe,
            port=self.port,
            vision=self.vision,
            race_config=self.race_config,
            log_file=self.launch_log,
            autostart_script=self.autostart_script,
            existing_pid=self.torcs_pid,
        )

    def _relaunch_torcs(self):
        if self.torcs_exe:
            self._launch_managed_torcs()
            return

        os.system("pkill torcs")
        time.sleep(1.0)
        cmd = "torcs -nofuel -nodamage -nolaptime"
        if self.vision:
            cmd += " -vision"
        os.system(cmd + " &")
        time.sleep(1.0)
        os.system("sh autostart.sh")

    def setup_connection(self):
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            print("Error: Could not create socket...")
            sys.exit(-1)
        self.so.settimeout(1)

        n_fail = 5
        while True:
            a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = "%s(init %s)" % (self.sid, a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                sys.exit(-1)
            sockdata = str()
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode("utf-8")
            except socket.error:
                print("Waiting for server on %d............" % self.port)
                # print("Count Down : " + str(n_fail))
                # if n_fail < 0:
                #     print("relaunch torcs")
                #     self._relaunch_torcs()
                #     n_fail = 5
                # n_fail -= 1

            if "***identified***" in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            opts, args = getopt.getopt(
                sys.argv[1:],
                "H:p:i:m:e:t:s:dhv",
                ["host=", "port=", "id=", "steps=", "episodes=", "track=", "stage=", "debug", "help", "version"],
            )
        except getopt.error as why:
            print("getopt error: %s\n%s" % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] in ["-h", "--help"]:
                    print(usage)
                    sys.exit(0)
                if opt[0] in ["-d", "--debug"]:
                    self.debug = True
                if opt[0] in ["-H", "--host"]:
                    self.host = opt[1]
                if opt[0] in ["-i", "--id"]:
                    self.sid = opt[1]
                if opt[0] in ["-t", "--track"]:
                    self.trackname = opt[1]
                if opt[0] in ["-s", "--stage"]:
                    self.stage = int(opt[1])
                if opt[0] in ["-p", "--port"]:
                    self.port = int(opt[1])
                if opt[0] in ["-e", "--episodes"]:
                    self.maxEpisodes = int(opt[1])
                if opt[0] in ["-m", "--steps"]:
                    self.maxSteps = int(opt[1])
                if opt[0] in ["-v", "--version"]:
                    print("%s %s" % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print("Bad parameter '%s' for option %s: %s\n%s" % (opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print("Superflous input? %s\n%s" % (", ".join(args), usage))
            sys.exit(-1)

    def get_servers_input(self):
        if not self.so:
            return
        sockdata = str()

        while True:
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode("utf-8")
            except socket.error:
                print(".", end=" ")
            if "***identified***" in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            if "***shutdown***" in sockdata:
                print((("Server has stopped the race on %d. " + "You were in %d place.") % (self.port, self.S.d["racePos"])))
                self.shutdown()
                return
            if "***restart***" in sockdata:
                print("Server has restarted the race on %d." % self.port)
                self.shutdown()
                return
            if not sockdata:
                continue

            self.S.parse_server_str(sockdata)
            if self.debug:
                sys.stderr.write("\x1b[2J\x1b[H")
                print(self.S)
            break

    def respond_to_server(self):
        if not self.so:
            return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1], str(emsg[0])))
            sys.exit(-1)
        if self.debug:
            print(self.R.d)

    def shutdown(self):
        if not self.so:
            return
        print(("Race terminated or %d steps elapsed. Shutting down %d." % (self.maxSteps, self.port)))
        self.so.close()
        self.so = None
