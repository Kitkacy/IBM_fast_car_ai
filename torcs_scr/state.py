from .utils import clip, destringify


class ServerState:
    """Container for the latest server message and parsed sensor dictionary."""

    def __init__(self):
        self.servstr = ""
        self.d = {}

    def parse_server_str(self, server_string):
        self.servstr = server_string.strip()[:-1]
        segments = self.servstr.strip().lstrip("(").rstrip(")").split(")(")
        for segment in segments:
            parts = segment.split(" ")
            self.d[parts[0]] = destringify(parts[1:])

    def __repr__(self):
        return f"ServerState({self.d})"


class DriverAction:
    """Driver command model sent back to TORCS SCR server."""

    def __init__(self):
        self.actionstr = ""
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
        if not isinstance(self.d["focus"], list) or min(self.d["focus"]) < -180 or max(self.d["focus"]) > 180:
            self.d["focus"] = 0

    def __repr__(self):
        self.clip_to_limits()
        out = ""
        for key in self.d:
            out += f"({key} "
            value = self.d[key]
            if isinstance(value, list):
                out += " ".join([str(x) for x in value])
            else:
                out += f"{value:.3f}"
            out += ")"
        return out
