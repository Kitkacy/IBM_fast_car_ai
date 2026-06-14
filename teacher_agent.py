import numpy as np


class Agent(object):
    def __init__(self, dim_action):
        self.dim_action = dim_action

        self.prev_steer = 0.0
        self.prev_accel = 0.0
        self.prev_brake = 0.0
        self.prev_track = None

        self.STEER_SIGN = -1.0

    def act(self, ob, reward, done, vision_on):
        track = np.array(ob.track, dtype=float)
        speed_kmh = float(np.array(ob.speedX).item())

        try:
            angle = float(np.array(ob.angle).item())
        except Exception:
            angle = 0.0

        try:
            track_pos = float(np.array(ob.trackPos).item())
        except Exception:
            track_pos = 0.0

        if len(track) < 19 or np.max(track) <= 0:
            return np.array([0.0, 0.30, 0.0], dtype=np.float32)

        if np.max(track) <= 2.0:
            track_m = track * 200.0
        else:
            track_m = track

        if self.prev_track is None:
            smooth_track = track_m.copy()
        else:
            smooth_track = 0.82 * self.prev_track + 0.18 * track_m

        self.prev_track = smooth_track.copy()

        center_sensor = 9
        sensors = np.arange(19) - center_sensor

        front = float(smooth_track[9])
        best_sensor = int(np.argmax(smooth_track))

        weights = np.maximum(smooth_track, 0.0) ** 1.4

        if np.sum(weights) > 0:
            weighted_direction = np.sum(sensors * weights) / np.sum(weights)
        else:
            weighted_direction = 0.0

        steer_sensor = self.STEER_SIGN * (weighted_direction / center_sensor)
        steer_best = self.STEER_SIGN * ((best_sensor - center_sensor) / center_sensor)
        steer_stability = (-angle * 0.90) + (track_pos * 0.18)

        steer = (
            0.60 * steer_sensor
            + 0.15 * steer_best
            + 0.25 * steer_stability
        )

        if front > 150 and abs(best_sensor - center_sensor) <= 2:
            steer *= 0.35
        elif front > 120 and abs(best_sensor - center_sensor) <= 2:
            steer *= 0.55

        if front < 110:
            steer *= 1.08
        if front < 85:
            steer *= 1.20
        if front < 65:
            steer *= 1.38
        if front < 48:
            steer *= 1.60
        if front < 35:
            steer *= 1.85

        steer = float(np.clip(steer, -1.0, 1.0))

        if front > 140:
            max_change = 0.018
        elif front > 100:
            max_change = 0.035
        elif front > 70:
            max_change = 0.065
        elif front > 50:
            max_change = 0.095
        else:
            max_change = 0.130

        steer_change = steer - self.prev_steer
        steer_change = float(np.clip(steer_change, -max_change, max_change))

        steer = self.prev_steer + steer_change
        steer = float(np.clip(steer, -1.0, 1.0))

        abs_steer = abs(steer)
        target_speed = 85.0

        if front > 165 and abs_steer < 0.08:
            target_speed = 105.0
        elif front > 135 and abs_steer < 0.15:
            target_speed = 96.0

        if front < 115:
            target_speed = min(target_speed, 78.0)
        if front < 90:
            target_speed = min(target_speed, 66.0)
        if front < 70:
            target_speed = min(target_speed, 56.0)
        if front < 50:
            target_speed = min(target_speed, 44.0)
        if front < 35:
            target_speed = min(target_speed, 34.0)

        if abs_steer > 0.30:
            target_speed = min(target_speed, 70.0)
        if abs_steer > 0.50:
            target_speed = min(target_speed, 55.0)
        if abs_steer > 0.70:
            target_speed = min(target_speed, 42.0)

        if abs(angle) > 0.35:
            target_speed = min(target_speed, 40.0)

        if abs(track_pos) > 0.75:
            target_speed = min(target_speed, 58.0)

        if abs(track_pos) > 1.00:
            target_speed = min(target_speed, 38.0)

        if speed_kmh < target_speed - 10:
            accel = 0.65
            brake = 0.0
        elif speed_kmh < target_speed - 4:
            accel = 0.42
            brake = 0.0
        elif speed_kmh > target_speed + 14:
            accel = 0.0
            brake = 0.45
        elif speed_kmh > target_speed + 7:
            accel = 0.0
            brake = 0.25
        elif speed_kmh > target_speed + 3:
            accel = 0.05
            brake = 0.10
        else:
            accel = 0.20
            brake = 0.0

        if front < 42 and speed_kmh > 55:
            accel = 0.0
            brake = 0.60

        if front < 30 and speed_kmh > 42:
            accel = 0.0
            brake = 0.80

        if speed_kmh < 8:
            accel = 0.75
            brake = 0.0

        accel = 0.60 * self.prev_accel + 0.40 * accel
        brake = 0.55 * self.prev_brake + 0.45 * brake

        accel = float(np.clip(accel, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        self.prev_steer = steer
        self.prev_accel = accel
        self.prev_brake = brake

        return np.array([steer, accel, brake], dtype=np.float32)
