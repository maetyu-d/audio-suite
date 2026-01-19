import numpy as np
from dataclasses import dataclass
from .music import midi_to_hz, quantize_to_bits

def _apply_micro_fade(x: np.ndarray, sr: int, fade_ms: float = 12.0) -> np.ndarray:
    """Short cosine fade-in/out to avoid clicks at note boundaries."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.shape[0])
    if n <= 16:
        return x
    fade_n = int(round(sr * (fade_ms / 1000.0)))
    fade_n = int(max(8, min(fade_n, n // 3)))
    if fade_n <= 1:
        return x
    # half-cosine ramp: smoother HF than linear
    t = np.linspace(0.0, np.pi, fade_n, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(t)
    x[:fade_n] *= ramp
    x[-fade_n:] *= ramp[::-1]
    x[0] = 0.0
    x[-1] = 0.0
    return x
    fade_n = int(round(sr * (fade_ms / 1000.0)))
    fade_n = int(max(4, min(fade_n, n // 4)))
    if fade_n <= 1:
        return x
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    x[:fade_n] *= ramp
    x[-fade_n:] *= ramp[::-1]
    x[0] = 0.0
    x[-1] = 0.0
    return x



@dataclass
class PSGParams:
    noise: bool = False
    duty: float = 0.5
    a: float = 0.001
    d: float = 0.1
    s: float = 0.6
    r: float = 0.1
    bits: int = 12  # vibe quantize


def _adsr(n: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    # De-click: enforce a tiny minimum attack/release so PSG edges don't pop.
    a = max(0.003, float(a))
    d = max(1e-4, float(d))
    r = max(0.006, float(r))
    # Clamp envelope stages so the total never exceeds the note duration.
    # This prevents numpy broadcast errors for very short notes.
    n_a = min(n, int(sr * a))
    rem = max(0, n - n_a)
    n_d = min(rem, int(sr * d))
    rem = max(0, rem - n_d)
    n_r = min(rem, int(sr * r))
    rem = max(0, rem - n_r)
    n_s = rem

    env = np.zeros(n, dtype=np.float32)
    i = 0
    if n_a > 0:
        ramp = np.linspace(0.0, 1.0, n_a, endpoint=False, dtype=np.float32)
        ramp = ramp * ramp
        env[i:i+n_a] = ramp
        i += n_a
    if n_d > 0:
        env[i:i+n_d] = np.linspace(1.0, s, n_d, endpoint=False, dtype=np.float32)
        i += n_d
    if n_s > 0:
        env[i:i+n_s] = s
        i += n_s
    if n_r > 0:
        ramp = np.linspace(1.0, 0.0, n_r, endpoint=True, dtype=np.float32)
        ramp = ramp * ramp
        startv = float(env[i-1] if i > 0 else s)
        env[i:i+n_r] = startv * ramp
    return env


def _square(sr: int, hz: float, n: int, duty: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / float(sr)
    phase = (t * hz) % 1.0
    return np.where(phase < duty, 1.0, -1.0).astype(np.float32)


def _noise_lfsr(n: int, seed: int = 1) -> np.ndarray:
    # 15-bit LFSR-ish noise
    lfsr = seed & 0x7FFF
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        bit = (lfsr ^ (lfsr >> 1)) & 1
        lfsr = (lfsr >> 1) | (bit << 14)
        out[i] = 1.0 if (lfsr & 1) else -1.0
    return out


def render_psg_note(sr: int, dur_s: float, midi: float, vel: float, p: PSGParams, seed: int = 1) -> np.ndarray:
    n = int(max(1, round(dur_s * sr)))
    env = _adsr(n, sr, p.a, p.d, p.s, p.r)
    if p.noise:
        sig = _noise_lfsr(n, seed=seed)
    else:
        hz = midi_to_hz(midi)
        sig = _square(sr, hz, n, duty=float(np.clip(p.duty, 0.05, 0.95)))
    y = sig * env * float(vel)
    y = quantize_to_bits(y.astype(np.float32), int(p.bits))
    y = _apply_micro_fade(y, sr)
    y = _one_pole_lp(y, sr, 12000.0)
    return y.astype(np.float32)

def _one_pole_lp(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Very cheap smoothing LPF (console-ish output stage vibe)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    cutoff_hz = float(max(50.0, cutoff_hz))
    a = np.exp(-2.0 * np.pi * cutoff_hz / float(sr))
    y = np.empty_like(x, dtype=np.float32)
    prev = 0.0
    for i in range(x.shape[0]):
        prev = a * prev + (1.0 - a) * float(x[i])
        y[i] = prev
    return y
