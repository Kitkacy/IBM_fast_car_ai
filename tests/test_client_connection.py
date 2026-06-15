import socket
import threading
import unittest
from unittest.mock import patch, MagicMock

from torcs_scr.client import Client


class MockTorcsServer:
    def __init__(self, port=3002):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("localhost", self.port))
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while self.running:
            try:
                self.sock.settimeout(0.1)
                data, addr = self.sock.recvfrom(1024)
                if b"init" in data:
                    self.sock.sendto(b"***identified***", addr)
            except socket.timeout:
                continue
            except Exception:
                break

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.sock.close()


class TestClientConnection(unittest.TestCase):
    @patch("torcs_scr.client.Path")
    @patch("torcs_scr.client.subprocess.Popen")
    def test_successful_connection(self, mock_popen, mock_path_cls):
        server = MockTorcsServer(port=3002)
        try:
            mock_proc = MagicMock()
            mock_proc.pid = 99999
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.expanduser.return_value.resolve.return_value = mock_path
            mock_path.parent = "."
            mock_path_cls.return_value = mock_path

            client = Client(torcs_exe="fake_torcs.exe", port=3002)
            self.assertIsNotNone(client.so)
            self.assertEqual(client.torcs_pid, 99999)
            client.close()
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
