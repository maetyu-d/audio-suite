from __future__ import annotations

"""Fibonacci Clock Desync

Three concurrent step clocks with Fibonacci lengths.
"""

from examples._common import NoteEvent, RenderConfig, SCALES, beat_to_sec


def generate(cfg: RenderConfig, root_midi: int = 57, scale: str = 'minor',
             a: int = 21, b: int = 34, c: int = 55, step_beats: float = 0.25) -> list[NoteEvent]:
    sc = SCALES.get(scale, SCALES['minor'])
    events: list[NoteEvent] = []

    t_beats = 0.0
    n_steps = int(cfg.seconds / beat_to_sec(cfg.bpm, step_beats)) + 1
    for i in range(n_steps):
        t0 = beat_to_sec(cfg.bpm, t_beats)

        # A: main pulse
        deg_a = (i % a) % len(sc)
        events.append(NoteEvent(t0=t0, dur=beat_to_sec(cfg.bpm, step_beats*0.9), midi=float(root_midi + sc[deg_a]), vel=0.7, chan=0, engine='FM'))

        # B: counter pulse
        if (i % b) < (b//3):
            deg_b = (i*2 % b) % len(sc)
            events.append(NoteEvent(t0=t0 + beat_to_sec(cfg.bpm, step_beats*0.5), dur=beat_to_sec(cfg.bpm, step_beats*0.85), midi=float(root_midi + 12 + sc[deg_b]), vel=0.55, chan=1, engine='FM'))

        # C: PSG accent
        if (i % c) == 0:
            events.append(NoteEvent(t0=t0, dur=beat_to_sec(cfg.bpm, step_beats*0.2), midi=float(root_midi-12), vel=0.35, chan=0, engine='PSG'))

        t_beats += step_beats
        if beat_to_sec(cfg.bpm, t_beats) >= cfg.seconds:
            break

    return events
