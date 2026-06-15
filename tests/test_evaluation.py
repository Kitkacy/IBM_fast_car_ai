import unittest
from unittest.mock import MagicMock

from torcs_rl.evaluation import LapLoggerCallback, evaluate_with_lap_stats
from torcs_rl.utils import safe_float as _safe_float


class TestSafeFloat(unittest.TestCase):
    def test_normal_value(self):
        self.assertEqual(_safe_float(3.14), 3.14)

    def test_string_number(self):
        self.assertEqual(_safe_float("2.5"), 2.5)

    def test_none_returns_none(self):
        self.assertIsNone(_safe_float(None))

    def test_nan_returns_none(self):
        self.assertIsNone(_safe_float(float("nan")))

    def test_inf_returns_none(self):
        self.assertIsNone(_safe_float(float("inf")))


class TestLapLoggerCallback(unittest.TestCase):
    def _make_callback(self):
        cb = LapLoggerCallback(prefix="train", verbose=0)
        mock_model = MagicMock()
        cb.model = mock_model
        cb.locals = {}
        return cb

    def test_logs_lap_time_on_completion(self):
        cb = self._make_callback()
        cb.locals = {
            "infos": [
                {"lap_completed": True, "completed_lap_time": 30.5, "laps_completed": 2}
            ]
        }
        result = cb._on_step()
        self.assertTrue(result)
        calls = {c[0][0]: c[0][1] for c in cb.model.logger.record.call_args_list}
        self.assertEqual(calls["train/lap_time"], 30.5)
        self.assertEqual(calls["train/laps_completed"], 2.0)

    def test_logs_episode_info(self):
        cb = self._make_callback()
        cb.locals = {
            "infos": [
                {
                    "episode": {
                        "r": 150.0,
                        "l": 500,
                        "laps_completed": 3,
                        "last_lap_time": 32.0,
                        "best_lap_time": 30.0,
                        "distance_raced": 1000.0,
                    }
                }
            ]
        }
        cb._on_step()
        calls = {c[0][0]: c[0][1] for c in cb.model.logger.record.call_args_list}
        self.assertEqual(calls["train/episode_reward"], 150.0)
        self.assertEqual(calls["train/episode_length"], 500.0)
        self.assertEqual(calls["train/laps_completed"], 3.0)


class TestEvaluateWithLapStats(unittest.TestCase):
    def test_uses_timed_laps_for_median(self):
        model = MagicMock()
        env = MagicMock()

        def fake_evaluate_policy(_model, _env, **kwargs):
            callback = kwargs["callback"]
            callback(
                {"info": {"lap_completed": True, "completed_lap_time": 40.0, "laps_completed": 1}, "done": False},
                None,
            )
            callback(
                {"info": {"lap_completed": True, "completed_lap_time": 35.0, "laps_completed": 2}, "done": False},
                None,
            )
            callback(
                {
                    "info": {
                        "episode": {
                            "r": 100.0,
                            "l": 200,
                            "laps_completed": 2,
                            "best_lap_time": 35.0,
                            "last_lap_time": 35.0,
                            "distance_raced": 1000.0,
                            "off_track_count": 0,
                            "termination_reason": "max_steps",
                        }
                    },
                    "done": True,
                },
                None,
            )
            return [100.0], [200]

        from unittest.mock import patch

        with patch("torcs_rl.evaluation.evaluate_policy", side_effect=fake_evaluate_policy):
            rewards, lengths, lap_stats, episode_metrics = evaluate_with_lap_stats(
                model,
                env,
                n_eval_episodes=1,
                deterministic=True,
                render=False,
                warn=True,
            )

        self.assertEqual(rewards, [100.0])
        self.assertEqual(lengths, [200])
        self.assertEqual(lap_stats["eval_timed_laps"], 1)
        self.assertEqual(lap_stats["eval_median_lap_time"], 35.0)
        self.assertEqual(episode_metrics[0]["timed_laps"], 1)


if __name__ == "__main__":
    unittest.main()
