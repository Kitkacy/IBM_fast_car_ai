import unittest
from torcs_scr.state import ServerState, DriverAction

class TestScrState(unittest.TestCase):
    def test_server_state_parsing(self):
        ss = ServerState()
        msg = "(angle 0.123)(curLapTime 10.5)(damage 0)(distFromStart 100)(distRaced 50)(fuel 10)(gear 1)(lastLapTime 0)(opponents 200 200 200)(racePos 3)(rpm 3000)(speedX 50.5)(speedY 0.1)(speedZ 0.0)(track 1 2 3 4 5)(trackPos 0.5)(wheelSpinVel 0 0 0 0)(z 0.5)"
        ss.parse_server_str(msg)

        self.assertEqual(ss.d["angle"], 0.123)
        self.assertEqual(ss.d["curLapTime"], 10.5)
        self.assertEqual(ss.d["damage"], 0.0)
        self.assertEqual(ss.d["opponents"], [200.0, 200.0, 200.0])
        self.assertEqual(ss.d["track"], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(ss.d["speedX"], 50.5)

    def test_driver_action_encoding(self):
        da = DriverAction()
        da.d["accel"] = 0.5
        da.d["steer"] = -0.2
        da.d["gear"] = 2

        encoded = repr(da)
        self.assertIn("(accel 0.500)", encoded)
        self.assertIn("(steer -0.200)", encoded)
        self.assertIn("(gear 2.000)", encoded)
        self.assertIn("(meta 0.000)", encoded)

    def test_driver_action_clipping(self):
        da = DriverAction()
        da.d["accel"] = 1.5
        da.d["steer"] = -2.0
        da.d["gear"] = 10

        repr(da)

        self.assertEqual(da.d["accel"], 1.0)
        self.assertEqual(da.d["steer"], -1.0)
        self.assertEqual(da.d["gear"], 0)

if __name__ == "__main__":
    unittest.main()
