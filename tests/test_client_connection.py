import unittest
import socket
import threading
import time
from torcs_scr.client import Client

class MockTorcsServer:
    def __init__(self, port=3002):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("localhost", self.port))
        self.running = True
        self.thread = threading.Thread(target=self.run)
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
        self.thread.join()
        self.sock.close()

class TestClientConnection(unittest.TestCase):
    def test_successful_connection(self):
        server = MockTorcsServer(port=3002)
        try:
            # We don't want it to launch TORCS
            client = Client(p=3002)
            # Prevent auto-launch
            client.torcs_exe = ""
            client.setup_connection()
            self.assertIsNotNone(client.so)
        finally:
            server.stop()

if __name__ == "__main__":
    unittest.main()
