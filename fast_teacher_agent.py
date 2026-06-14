import math

import numpy as np


MIN_SPEED = 70.0
MAX_SPEED = 120.0
LOOKAHEAD_NEAR = 30.0
LOOKAHEAD_FAR = 100.0
STEER_GAIN = 35.0
CENTERING_GAIN = 0.40
BRAKE_CURVE_FACTOR = 0.8
EMERGENCY_BRAKE_DIST = 25.0
GEAR_UP = [0.0, 5500.0, 6000.0, 6500.0, 7000.0, 7500.0]
GEAR_DOWN = [0.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0]
ENABLE_TRACTION_CONTROL = True
ENABLE_RECOVERY = True


class FastTeacherAgent:
    def __init__(self):
        self.gear = 1

    def act(self, raw_obs):
        recovery = self._recovery_maneuver(raw_obs)
        if recovery is not None:
            accel, brake, steer = recovery
        else:
            steer = self._calculate_steering(raw_obs)
            accel, brake = self._calculate_throttle_and_brake(raw_obs)
            accel = self._traction_control(raw_obs, accel)

        self.gear = self._shift_gears(raw_obs, self.gear)
        return {
            "steer": float(np.clip(steer, -1.0, 1.0)),
            "accel": float(np.clip(accel, 0.0, 1.0)),
            "brake": float(np.clip(brake, 0.0, 1.0)),
            "gear": int(self.gear),
        }

    def _get_forward_distance(self, raw_obs):
        return max(0.0, min(200.0, float(raw_obs["track"][9])))

    def _target_speed_dynamic(self, raw_obs):
        forward = self._get_forward_distance(raw_obs)
        if forward >= LOOKAHEAD_FAR:
            return MAX_SPEED
        if forward <= LOOKAHEAD_NEAR:
            return MIN_SPEED
        ratio = (forward - LOOKAHEAD_NEAR) / (LOOKAHEAD_FAR - LOOKAHEAD_NEAR)
        return MIN_SPEED + ratio * (MAX_SPEED - MIN_SPEED)

    def _calculate_steering(self, raw_obs):
        angle = float(raw_obs["angle"])
        track_pos = float(raw_obs["trackPos"])
        steer = (angle * STEER_GAIN / math.pi) - (track_pos * CENTERING_GAIN)
        return max(-1.0, min(1.0, steer))

    def _calculate_throttle_and_brake(self, raw_obs):
        target = self._target_speed_dynamic(raw_obs)
        speed = float(raw_obs["speedX"])
        forward = self._get_forward_distance(raw_obs)

        if forward < EMERGENCY_BRAKE_DIST and speed > MIN_SPEED:
            return 0.0, 1.0

        diff = target - speed
        if diff > 5.0:
            accel, brake = 1.0, 0.0
        elif diff > 0.0:
            accel, brake = 0.6, 0.0
        elif diff > -5.0:
            accel, brake = 0.1, 0.0
        else:
            accel = 0.0
            brake = min(0.8, (-diff / 20.0) * BRAKE_CURVE_FACTOR + 0.2)

        if speed < 5.0:
            accel, brake = 1.0, 0.0

        return accel, brake

    def _shift_gears(self, raw_obs, current_gear):
        rpm = float(raw_obs["rpm"])
        gear = max(1, min(5, int(current_gear) if current_gear else 1))
        if gear < 6 and rpm > GEAR_UP[gear]:
            gear += 1
        elif gear > 1 and rpm < GEAR_DOWN[gear]:
            gear -= 1
        return max(1, min(6, gear))

    def _traction_control(self, raw_obs, accel):
        if not ENABLE_TRACTION_CONTROL:
            return accel
        wheel_spin = raw_obs.get("wheelSpinVel", [0.0, 0.0, 0.0, 0.0])
        rear = float(wheel_spin[2]) + float(wheel_spin[3])
        front = float(wheel_spin[0]) + float(wheel_spin[1])
        if (rear - front) > 2.0:
            accel -= 0.15
        return max(0.0, accel)

    def _recovery_maneuver(self, raw_obs):
        if not ENABLE_RECOVERY:
            return None
        track_pos = float(raw_obs["trackPos"])
        angle = float(raw_obs["angle"])
        if abs(track_pos) > 1.0:
            if float(raw_obs["speedX"]) < 5.0 and abs(angle) > math.pi / 2:
                return 0.3, 0.0, -angle * 0.5
            steer = -track_pos * 0.5 + angle * 0.5
            return 0.3, 0.0, max(-1.0, min(1.0, steer))
        return None
