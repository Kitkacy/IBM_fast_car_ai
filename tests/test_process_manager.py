import unittest
from unittest.mock import patch, MagicMock
import os
from torcs_scr.process_manager import _find_udp_port_owner_pids, kill_torcs_process

class TestProcessManager(unittest.TestCase):
    @patch("subprocess.run")
    def test_find_udp_port_owner_pids_windows(self, mock_run):
        # Mock netstat output for Windows
        mock_run.return_value = MagicMock(
            stdout="  UDP    0.0.0.0:3001           *:*                                    1234\n",
            returncode=0
        )

        pids = _find_udp_port_owner_pids(3001)
        self.assertEqual(pids, {1234})

    @patch("subprocess.run")
    def test_kill_torcs_process_windows(self, mock_run):
        kill_torcs_process(1234)
        # Check if taskkill was called
        args, kwargs = mock_run.call_args
        self.assertIn("taskkill", args[0])
        self.assertIn("1234", args[0])

if __name__ == "__main__":
    unittest.main()
