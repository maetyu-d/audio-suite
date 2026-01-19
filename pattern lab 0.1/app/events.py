from dataclasses import dataclass


@dataclass
class NoteEvent:
    t0: float          # seconds
    dur: float         # seconds
    midi: float        # pitch in MIDI note number (can be fractional)
    vel: float = 1.0   # 0..1
    chan: int = 0      # synth channel
    engine: str = 'FM' # 'FM' or 'PSG'


@dataclass
class RenderConfig:
    sample_rate: int = 44100
    seconds: float = 20.0
    bpm: float = 120.0
    swing: float = 0.0            # 0..0.5
    time_stretch: float = 1.0     # multiplies event times
    micro_jitter: float = 0.0     # seconds
    master_gain: float = 0.9
    seed: int = 1
