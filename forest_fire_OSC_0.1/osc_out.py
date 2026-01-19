from dataclasses import dataclass
from pythonosc.udp_client import SimpleUDPClient


@dataclass
class OSCConfig:
    host: str = "127.0.0.1"
    port: int = 9000
    enabled: bool = True


class OSCSender:
    def __init__(self, cfg: OSCConfig):
        self.cfg = cfg
        self.client = SimpleUDPClient(cfg.host, int(cfg.port))

    def set_target(self, host: str, port: int):
        self.cfg.host = host
        self.cfg.port = int(port)
        self.client = SimpleUDPClient(self.cfg.host, int(self.cfg.port))

    def send(self, address: str, *args):
        if not self.cfg.enabled:
            return
        self.client.send_message(address, list(args))
