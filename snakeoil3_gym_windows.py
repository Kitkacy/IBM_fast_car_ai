#!/usr/bin/python
# Windows-friendly snakeoil client for TORCS SCR server.

import socket
import sys
from pathlib import Path

from torcs_process_manager import launch_torcs_process

PI = 3.14159265359
data_size = 2**17

def clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def bargraph(x, mn, mx, w, c="X"):
    if not w:
        return ""
    if x < mn:
        x = mn
    if x > mx:
        x = mx
    tx = mx - mn
    if tx <= 0:
        return "backwards"
    upw = tx / float(w)
    if upw <= 0:
        return "what?"

    negpu, pospu, negnonpu, posnonpu = 0, 0, 0, 0
    if mn < 0:
        if x < 0:
            negpu = -x + min(0, mx)
            negnonpu = -mn + x
        else:
            negnonpu = -mn + min(0, mx)
    if mx > 0:
        if x > 0:
            pospu = x - max(0, mn)
            posnonpu = mx - x
        else:
            posnonpu = mx - max(0, mn)

    nnc = int(negnonpu / upw) * "-"
    npc = int(negpu / upw) * c
    ppc = int(pospu / upw) * c
    pnc = int(posnonpu / upw) * "_"
    return "[%s]" % (nnc + npc + ppc + pnc)


class Client:
    def __init__(
        self,
        p=None,
        vision=False,
        torcs_exe=r"C:\Users\szymo\source\repos\torcs\torcs\wtorcs.exe",
        launch_log="torcs_launcher.log",
        race_config=r"C:\Users\szymo\source\repos\torcs\torcs\config\raceman\practice.xml",
        launch_on_start=True,
    ):
        self.vision = vision
        self.torcs_exe = torcs_exe
        self.launch_on_start = launch_on_start
        self.launch_log = Path(__file__).resolve().parent / launch_log
        self.autostart_script = Path(__file__).resolve().parent / "autostart_windows.ps1"
        self.race_config = str(Path(race_config).expanduser()) if race_config else ""

        self.host = "localhost"
        self.port = 3001
        self.sid = "SCR"
        self.debug = False
        self.maxSteps = 100000
        self.torcs_pid = None

        if p:
            self.port = p

        self.S = ServerState()
        self.R = DriverAction()
        if self.launch_on_start:
            self._launch_torcs_windows()
        self.setup_connection()

    def _launch_torcs_windows(self):
        torcs_path = Path(self.torcs_exe).expanduser()
        self.torcs_pid = launch_torcs_process(
            torcs_exe=str(torcs_path),
            port=self.port,
            vision=self.vision,
            race_config=self.race_config,
            log_file=str(self.launch_log),
            autostart_script=str(self.autostart_script),
            existing_pid=self.torcs_pid,
        )

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
                sockdata, _addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode("utf-8")
            except socket.error:
                if n_fail < 0:
                    n_fail = 5
                n_fail -= 1

            if "***identified***" in sockdata:
                break

    def get_servers_input(self):
        if not self.so:
            return

        sockdata = str()
        while True:
            try:
                sockdata, _addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode("utf-8")
            except socket.error:
                continue

            if "***identified***" in sockdata:
                continue
            if "***shutdown***" in sockdata:
                print(
                    (
                        (
                            "[Learning outcome] Episode stopped by simulator on port %d. "
                            + "Final position: %d."
                        )
                        % (self.port, self.S.d["racePos"])
                    )
                )
                self.shutdown()
                return
            if "***restart***" in sockdata:
                print("[Learning outcome] Simulator requested restart on port %d." % self.port)
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
            print(self.R.fancyout())

    def shutdown(self):
        if not self.so:
            return
        print("Race terminated or %d steps elapsed. Shutting down %d." % (self.maxSteps, self.port))
        self.so.close()
        self.so = None


class ServerState:
    def __init__(self):
        self.servstr = str()
        self.d = dict()

    def parse_server_str(self, server_string):
        self.servstr = server_string.strip()[:-1]
        sslisted = self.servstr.strip().lstrip("(").rstrip(")").split(")(")
        for item in sslisted:
            w = item.split(" ")
            self.d[w[0]] = destringify(w[1:])

    def __repr__(self):
        return self.fancyout()

    def fancyout(self):
        out = str()
        sensors = [
            "stucktimer",
            "fuel",
            "distRaced",
            "distFromStart",
            "opponents",
            "wheelSpinVel",
            "z",
            "speedZ",
            "speedY",
            "speedX",
            "targetSpeed",
            "rpm",
            "skid",
            "slip",
            "track",
            "trackPos",
            "angle",
        ]

        for k in sensors:
            if type(self.d.get(k)) is list:
                if k == "track":
                    raw_tsens = ["%.1f" % x for x in self.d["track"]]
                    strout = " ".join(raw_tsens[:9]) + "_" + raw_tsens[9] + "_" + " ".join(raw_tsens[10:])
                elif k == "opponents":
                    strout = str()
                    for osensor in self.d["opponents"]:
                        if osensor > 190:
                            oc = "_"
                        elif osensor > 90:
                            oc = "."
                        elif osensor > 39:
                            oc = chr(int(osensor / 2) + 97 - 19)
                        elif osensor > 13:
                            oc = chr(int(osensor) + 65 - 13)
                        elif osensor > 3:
                            oc = chr(int(osensor) + 48 - 3)
                        else:
                            oc = "?"
                        strout += oc
                    strout = " -> " + strout[:18] + " " + strout[18:] + " <-"
                else:
                    strout = ", ".join([str(i) for i in self.d[k]])
            else:
                if k == "fuel":
                    strout = "%6.0f %s" % (self.d[k], bargraph(self.d[k], 0, 100, 50, "f"))
                elif k == "speedX":
                    cx = "X"
                    if self.d[k] < 0:
                        cx = "R"
                    strout = "%6.1f %s" % (self.d[k], bargraph(self.d[k], -30, 300, 50, cx))
                elif k == "speedY":
                    strout = "%6.1f %s" % (self.d[k], bargraph(self.d[k] * -1, -25, 25, 50, "Y"))
                elif k == "speedZ":
                    strout = "%6.1f %s" % (self.d[k], bargraph(self.d[k], -13, 13, 50, "Z"))
                elif k == "z":
                    strout = "%6.3f %s" % (self.d[k], bargraph(self.d[k], 0.3, 0.5, 50, "z"))
                elif k == "trackPos":
                    cx = "<"
                    if self.d[k] < 0:
                        cx = ">"
                    strout = "%6.3f %s" % (self.d[k], bargraph(self.d[k] * -1, -1, 1, 50, cx))
                elif k == "stucktimer":
                    if self.d[k]:
                        strout = "%3d %s" % (self.d[k], bargraph(self.d[k], 0, 300, 50, "'"))
                    else:
                        strout = "Not stuck!"
                elif k == "rpm":
                    g = self.d["gear"]
                    if g < 0:
                        g = "R"
                    else:
                        g = "%1d" % g
                    strout = bargraph(self.d[k], 0, 10000, 50, g)
                elif k == "angle":
                    asyms = [
                        "  !  ",
                        ".|'  ",
                        "./'  ",
                        "_.-  ",
                        ".--  ",
                        "..-  ",
                        "---  ",
                        ".__  ",
                        "-._  ",
                        "'-.  ",
                        "'\\.  ",
                        "'|.  ",
                        "  |  ",
                        "  .|'",
                        "  ./'",
                        "  .-'",
                        "  _.-",
                        "  __.",
                        "  ---",
                        "  --.",
                        "  -._",
                        "  -..",
                        "  '\\.",
                        "  '|.",
                    ]
                    rad = self.d[k]
                    deg = int(rad * 180 / PI)
                    symno = int(0.5 + (rad + PI) / (PI / 12))
                    symno = symno % (len(asyms) - 1)
                    strout = "%5.2f %3d (%s)" % (rad, deg, asyms[symno])
                elif k == "skid":
                    frontwheelradpersec = self.d["wheelSpinVel"][0]
                    skid = 0
                    if frontwheelradpersec:
                        skid = 0.5555555555 * self.d["speedX"] / frontwheelradpersec - 0.66124
                    strout = bargraph(skid, -0.05, 0.4, 50, "*")
                elif k == "slip":
                    frontwheelradpersec = self.d["wheelSpinVel"][0]
                    slip = 0
                    if frontwheelradpersec:
                        slip = (self.d["wheelSpinVel"][2] + self.d["wheelSpinVel"][3]) - (
                            self.d["wheelSpinVel"][0] + self.d["wheelSpinVel"][1]
                        )
                    strout = bargraph(slip, -5, 150, 50, "@")
                else:
                    strout = str(self.d[k])

            out += "%s: %s\n" % (k, strout)

        return out


class DriverAction:
    def __init__(self):
        self.actionstr = str()
        self.d = {
            "accel": 0.2,
            "brake": 0,
            "clutch": 0,
            "gear": 1,
            "steer": 0,
            "focus": [-90, -45, 0, 45, 90],
            "meta": 0,
        }

    def clip_to_limits(self):
        self.d["steer"] = clip(self.d["steer"], -1, 1)
        self.d["brake"] = clip(self.d["brake"], 0, 1)
        self.d["accel"] = clip(self.d["accel"], 0, 1)
        self.d["clutch"] = clip(self.d["clutch"], 0, 1)
        if self.d["gear"] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d["gear"] = 0
        if self.d["meta"] not in [0, 1]:
            self.d["meta"] = 0
        if type(self.d["focus"]) is not list or min(self.d["focus"]) < -180 or max(self.d["focus"]) > 180:
            self.d["focus"] = 0

    def __repr__(self):
        self.clip_to_limits()
        out = str()
        for k in self.d:
            out += "(" + k + " "
            v = self.d[k]
            if not type(v) is list:
                out += "%.3f" % v
            else:
                out += " ".join([str(x) for x in v])
            out += ")"
        return out

    def fancyout(self):
        out = str()
        od = self.d.copy()
        od.pop("gear", "")
        od.pop("meta", "")
        od.pop("focus", "")
        for k in sorted(od):
            if k in ("clutch", "brake", "accel"):
                strout = "%6.3f %s" % (od[k], bargraph(od[k], 0, 1, 50, k[0].upper()))
            elif k == "steer":
                strout = "%6.3f %s" % (od[k], bargraph(od[k] * -1, -1, 1, 50, "S"))
            else:
                strout = str(od[k])
            out += "%s: %s\n" % (k, strout)
        return out


def destringify(s):
    if not s:
        return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    if type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        return [destringify(i) for i in s]


def drive_example(c):
    S, R = c.S.d, c.R.d
    target_speed = 50

    R["steer"] = S["angle"] * 15 / PI
    R["steer"] -= S["trackPos"] * 0.10

    if S["speedX"] < target_speed - (R["steer"] * 50):
        R["accel"] += 0.01
    else:
        R["accel"] -= 0.01

    if S["speedX"] < 10:
        R["accel"] += 1 / (S["speedX"] + 0.1)

    if ((S["wheelSpinVel"][2] + S["wheelSpinVel"][3]) - (S["wheelSpinVel"][0] + S["wheelSpinVel"][1]) > 5):
        R["accel"] -= 0.2

    R["gear"] = 1
    if S["speedX"] > 50:
        R["gear"] = 2
    if S["speedX"] > 80:
        R["gear"] = 3
    if S["speedX"] > 110:
        R["gear"] = 4
    if S["speedX"] > 140:
        R["gear"] = 5
    if S["speedX"] > 170:
        R["gear"] = 6


if __name__ == "__main__":
    C = Client(p=3001)
    for _step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive_example(C)
        C.respond_to_server()
    C.shutdown()
