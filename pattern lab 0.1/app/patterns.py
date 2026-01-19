import numpy as np
from typing import Optional
from pathlib import Path
from .events import NoteEvent, RenderConfig
from .music import fibonacci, primes_upto, euclidean_rhythm, pythagorean_ratio
from .script_host import load_script_generator


SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'glass': [0, 2, 5, 7, 9],  # a modal-ish pentatonic scaffold
}


def _rng(seed: int):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def _beat_to_sec(bpm: float, beats: float) -> float:
    return float(beats) * 60.0 / float(bpm)


def pattern_glass_cells(cfg: RenderConfig, root_midi: int = 60, scale: str = 'glass',
                        cell_len: int = 8, voices: int = 2, drift: float = 0.0) -> list[NoteEvent]:
    """Glass-ish: repeated cell + additive/subtractive process + optional slow drift."""
    rng = _rng(cfg.seed)
    sc = SCALES.get(scale, SCALES['glass'])

    # build an arpeggiated cell of degrees
    degrees = [0, 1, 2, 3, 2, 1, 4, 3]
    degrees = (degrees * ((cell_len + len(degrees) - 1) // len(degrees)))[:cell_len]

    events: list[NoteEvent] = []
    beat = 0.0
    bar_beats = 4.0
    step_beats = bar_beats / cell_len

    # process length grows then shrinks (additive form)
    total_bars = int(max(1, cfg.seconds / _beat_to_sec(cfg.bpm, bar_beats)))
    grow = list(range(2, cell_len + 1)) + list(range(cell_len - 1, 1, -1))

    for b in range(total_bars):
        k = grow[b % len(grow)]
        for v in range(voices):
            chan = v % 6
            for i in range(k):
                deg = degrees[i]
                semis = sc[deg % len(sc)] + 12 * (deg // len(sc))
                midi = root_midi + semis + (v * 12)
                # subtle drift: very slow random walk in cents
                midi += drift * float(rng.normal(0, 0.02))
                t0 = _beat_to_sec(cfg.bpm, beat + i * step_beats)
                events.append(NoteEvent(t0=t0, dur=_beat_to_sec(cfg.bpm, step_beats*0.95), midi=midi,
                                        vel=0.9 if (i % 4 == 0) else 0.65,
                                        chan=chan, engine='FM'))
        beat += bar_beats

    return events


def pattern_fibonacci(cfg: RenderConfig, root_midi: int = 57, scale: str = 'minor',
                      steps: int = 64, pulses: int = 13) -> list[NoteEvent]:
    """Fibonacci drives both pitch-walk and duration ratios; Euclidean rhythm provides the gate."""
    rng = _rng(cfg.seed)
    sc = SCALES.get(scale, SCALES['minor'])
    fib = fibonacci(max(16, steps//2))
    gate = euclidean_rhythm(steps, pulses, rotate=int(rng.integers(0, steps)))

    events: list[NoteEvent] = []
    beat = 0.0
    base_step = 0.25  # 16ths

    for i in range(steps):
        dur_mul = 1.0 + (fib[i % len(fib)] % 5) * 0.25  # 1..2x-ish
        if gate[i] == 1:
            deg = fib[i % len(fib)] % len(sc)
            octv = (fib[(i+3) % len(fib)] % 3)
            midi = root_midi + sc[deg] + 12 * octv
            chan = int(i % 6)
            vel = 0.7 + 0.25 * float((i % 8) == 0)
            events.append(NoteEvent(
                t0=_beat_to_sec(cfg.bpm, beat),
                dur=_beat_to_sec(cfg.bpm, base_step * dur_mul * 0.92),
                midi=midi,
                vel=vel,
                chan=chan,
                engine='FM'
            ))
        beat += base_step
        if _beat_to_sec(cfg.bpm, beat) > cfg.seconds:
            break

    # add PSG clicky noise accents on prime indices
    primes = set(primes_upto(steps*2))
    beat = 0.0
    for i in range(steps):
        if i in primes and (i % 2 == 1):
            events.append(NoteEvent(
                t0=_beat_to_sec(cfg.bpm, beat),
                dur=_beat_to_sec(cfg.bpm, base_step*0.35),
                midi=48,
                vel=0.5,
                chan=0,
                engine='PSG'
            ))
        beat += base_step
        if _beat_to_sec(cfg.bpm, beat) > cfg.seconds:
            break

    return events


def pattern_prime_phase(cfg: RenderConfig, root_midi: int = 60, scale: str = 'dorian') -> list[NoteEvent]:
    """Two Glass-ish arps with prime-length cycles to create phase shift / beating."""
    sc = SCALES.get(scale, SCALES['dorian'])
    primes = primes_upto(50)
    p1, p2 = primes[8], primes[10]  # e.g. 23, 31

    events: list[NoteEvent] = []
    base_step = 0.25
    beat = 0.0

    for i in range(int(cfg.seconds / _beat_to_sec(cfg.bpm, base_step)) + 1):
        # voice A
        deg_a = (i % p1) % len(sc)
        midi_a = root_midi + sc[deg_a] + 12 * ((i % p1) // len(sc))
        events.append(NoteEvent(_beat_to_sec(cfg.bpm, beat), _beat_to_sec(cfg.bpm, base_step*0.9), midi_a,
                                vel=0.75, chan=0, engine='FM'))
        # voice B (different prime loop)
        deg_b = (i % p2) % len(sc)
        midi_b = root_midi + 12 + sc[deg_b] + 12 * ((i % p2) // len(sc))
        events.append(NoteEvent(_beat_to_sec(cfg.bpm, beat + base_step*0.5), _beat_to_sec(cfg.bpm, base_step*0.9), midi_b,
                                vel=0.65, chan=1, engine='FM'))

        # PSG hi-hat tick every 3
        if i % 3 == 0:
            events.append(NoteEvent(_beat_to_sec(cfg.bpm, beat), _beat_to_sec(cfg.bpm, base_step*0.2), 60,
                                    vel=0.35, chan=0, engine='PSG'))

        beat += base_step
        if _beat_to_sec(cfg.bpm, beat) > cfg.seconds:
            break

    return events


def pattern_pythagorean(cfg: RenderConfig, base_midi: int = 52, fifth_steps: Optional[list[int]] = None) -> list[NoteEvent]:
    """A canon where each voice follows Pythagorean fifth-steps; produces slow harmonic drift."""
    if fifth_steps is None:
        fifth_steps = [0, 1, 2, 3, 2, 1, 4, 5, 4, 3, 2, 1]

    events: list[NoteEvent] = []
    base_step = 0.5
    beat = 0.0

    # convert pythag ratios to fractional midi offsets (log2)
    for i in range(int(cfg.seconds / _beat_to_sec(cfg.bpm, base_step)) + 1):
        st = fifth_steps[i % len(fifth_steps)]
        ratio = pythagorean_ratio(st)
        midi_off = 12.0 * np.log2(ratio)

        # three voices staggered
        for v in range(3):
            t0 = _beat_to_sec(cfg.bpm, beat + v * base_step * 2.0)
            midi = base_midi + midi_off + 12 * v
            events.append(NoteEvent(t0, _beat_to_sec(cfg.bpm, base_step*1.8), float(midi),
                                    vel=0.55, chan=v, engine='FM'))

        # PSG bass pulse
        if i % 4 == 0:
            events.append(NoteEvent(_beat_to_sec(cfg.bpm, beat), _beat_to_sec(cfg.bpm, base_step*0.95), base_midi-12,
                                    vel=0.5, chan=1, engine='PSG'))

        beat += base_step
        if _beat_to_sec(cfg.bpm, beat) > cfg.seconds:
            break

    return events


def list_generators() -> list[str]:
    return ['Glass Cells', 'Fibonacci Gate', 'Prime Phase', 'Pythagorean Canon', 'Python Script']


def generate(name: str, cfg: RenderConfig, **kwargs) -> list[NoteEvent]:
    name = (name or '').strip().lower()
    if 'python' in name:
        script_path = kwargs.pop('script_path', '')
        entry = kwargs.pop('entry', 'generate')
        if not script_path:
            raise ValueError("Python Script generator requires gen.script_path")

        # Allow relative paths (relative to project root) as well as absolute.
        p = Path(script_path)
        if not p.is_absolute():
            # project root is .../glass_math_megadrive_app
            project_root = Path(__file__).resolve().parent.parent
            p = (project_root / p).resolve()
        fn = load_script_generator(p, entry)
        ev = fn(cfg=cfg, **kwargs)
        return ev
    if 'glass' in name:
        return pattern_glass_cells(cfg, **kwargs)
    if 'fibonacci' in name:
        return pattern_fibonacci(cfg, **kwargs)
    if 'prime' in name:
        return pattern_prime_phase(cfg, **kwargs)
    if 'pythag' in name:
        return pattern_pythagorean(cfg, **kwargs)
    # default
    return pattern_glass_cells(cfg)
