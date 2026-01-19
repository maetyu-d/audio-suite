import numpy as np
from typing import Optional
from dataclasses import dataclass
from .music import midi_to_hz, quantize_to_bits
from .constants import YM2612_DAC_BITS, POST_LP_HZ

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
class OpParams:
    ratio: float = 1.0          # freq ratio
    detune_cents: float = 0.0   # +/- cents
    level: float = 1.0          # linear gain
    index: float = 1.0          # modulation index (when used as modulator)
    a: float = 0.01
    d: float = 0.2
    s: float = 0.6
    r: float = 0.2


@dataclass
class FMVoiceParams:
    algorithm: int = 1  # 1=series 4->3->2->1, 2=2x stacks, 3=all carriers
    feedback: float = 0.0
    lfo_hz: float = 5.0
    lfo_depth: float = 0.0     # pitch vibrato in semitones
    ops: tuple[OpParams, OpParams, OpParams, OpParams] = (
        OpParams(ratio=1.0, level=0.9, index=0.0, a=0.01, d=0.3, s=0.7, r=0.15),
        OpParams(ratio=2.0, level=0.7, index=2.0, a=0.01, d=0.25, s=0.5, r=0.15),
        OpParams(ratio=3.0, level=0.6, index=2.0, a=0.01, d=0.2, s=0.4, r=0.15),
        OpParams(ratio=1.0, level=0.5, index=2.0, a=0.005, d=0.15, s=0.35, r=0.2),
    )


def _adsr_env(n: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    # De-click: enforce a tiny minimum attack/release so edges aren't razor sharp.
    a = max(0.004, float(a))
    d = max(1e-4, float(d))
    r = max(0.008, float(r))
    # The note duration (n) can be shorter than attack/decay/release.
    # Clamp each stage to the remaining budget so we never assign an array
    # longer than the slice (prevents numpy broadcast ValueError).
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
        # If release is the only stage (very short note), keep it well-defined.
        ramp = np.linspace(1.0, 0.0, n_r, endpoint=True, dtype=np.float32)
        ramp = ramp * ramp
        startv = float(env[i-1] if i > 0 else s)
        env[i:i+n_r] = startv * ramp
    return env


def _op_freq(base_hz: float, ratio: float, detune_cents: float) -> float:
    det = 2.0 ** (detune_cents / 1200.0)
    return float(base_hz * ratio * det)


def _sine_phase(sr: int, hz: float, n: int, phase0: float = 0.0, pm: Optional[np.ndarray] = None) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / float(sr)
    phase = (2.0 * np.pi * hz) * t + phase0
    if pm is not None:
        phase = phase + pm
    return np.sin(phase, dtype=np.float32)


def _one_pole_lp(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    # simple post-filter to reduce harshness and mimic console output stage
    cutoff_hz = float(max(20.0, cutoff_hz))
    a = np.exp(-2.0 * np.pi * cutoff_hz / float(sr))
    y = np.empty_like(x, dtype=np.float32)
    prev = 0.0
    for i in range(x.shape[0]):
        prev = a * prev + (1.0 - a) * float(x[i])
        y[i] = prev
    return y


def render_fm_note(sr: int, dur_s: float, midi: float, vel: float, p: FMVoiceParams) -> np.ndarray:
    n = int(max(1, round(dur_s * sr)))
    base = midi_to_hz(midi)

    # LFO vibrato
    if p.lfo_depth > 0.0:
        t = np.arange(n, dtype=np.float32) / float(sr)
        vib = np.sin(2.0 * np.pi * p.lfo_hz * t, dtype=np.float32)
        # semitone -> ratio
        vib_ratio = 2.0 ** ((p.lfo_depth * vib) / 12.0)
    else:
        vib_ratio = None

    ops = p.ops

    def op_sig(op: OpParams, pm: Optional[np.ndarray]) -> np.ndarray:
        hz = _op_freq(base, op.ratio, op.detune_cents)
        if vib_ratio is not None:
            # cheap vibrato: scale phase by time-varying ratio
            t = np.arange(n, dtype=np.float32) / float(sr)
            phase = (2.0 * np.pi * hz) * t * vib_ratio
            if pm is not None:
                phase = phase + pm
            sig = np.sin(phase, dtype=np.float32)
        else:
            sig = _sine_phase(sr, hz, n, pm=pm)
        env = _adsr_env(n, sr, op.a, op.d, op.s, op.r)
        return (sig * env * op.level).astype(np.float32)

    # feedback on last modulator in stack (Glass-y grit)
    fb = float(max(0.0, p.feedback))

    if p.algorithm == 1:
        # 4 -> 3 -> 2 -> 1 carrier
        o4 = op_sig(ops[3], pm=None)
        if fb > 0:
            o4 = o4 + fb * np.concatenate([[0.0], o4[:-1]]).astype(np.float32)
        o3 = op_sig(ops[2], pm=ops[2].index * o4)
        o2 = op_sig(ops[1], pm=ops[1].index * o3)
        o1 = op_sig(ops[0], pm=ops[0].index * o2)
        y = o1
    elif p.algorithm == 2:
        # (4->3) + (2->1) both carriers summed
        o4 = op_sig(ops[3], None)
        if fb > 0:
            o4 = o4 + fb * np.concatenate([[0.0], o4[:-1]]).astype(np.float32)
        o3 = op_sig(ops[2], pm=ops[2].index * o4)
        o2 = op_sig(ops[1], None)
        o1 = op_sig(ops[0], pm=ops[0].index * o2)
        y = (o3 + o1) * 0.6
    else:
        # all carriers, no modulation (organ-ish)
        o1 = op_sig(ops[0], None)
        o2 = op_sig(ops[1], None)
        o3 = op_sig(ops[2], None)
        o4 = op_sig(ops[3], None)
        y = (o1 + o2 + o3 + o4) * 0.25

    # velocity and MegaDrive-ish DAC quantization + mild lowpass
    y = (y * float(vel)).astype(np.float32)
    y = quantize_to_bits(y, YM2612_DAC_BITS)
    y = _apply_micro_fade(y, sr)
    y = _one_pole_lp(y, sr, POST_LP_HZ)
    y = _one_pole_lp(y, sr, 14000.0)
    return y.astype(np.float32)