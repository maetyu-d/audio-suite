from __future__ import annotations

"""Pythagorean Spiral Organ

Fifth-stacking without strict octave correction: slow drift / tilting consonance.
"""

import numpy as np
from examples._common import NoteEvent, RenderConfig, beat_to_sec, pythagorean_ratio


def generate(cfg: RenderConfig, base_midi: int = 52, step_beats: float = 0.5,
             voices: int = 3, fifth_step: int = 1) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    t_beats = 0.0
    n_steps = int(cfg.seconds / beat_to_sec(cfg.bpm, step_beats)) + 1

    for i in range(n_steps):
        st = i * int(fifth_step)
        ratio = pythagorean_ratio(st)
        midi_off = 12.0 * float(np.log2(ratio))
        t0 = beat_to_sec(cfg.bpm, t_beats)

        for v in range(int(voices)):
            midi = float(base_midi + midi_off + 12*v)
            events.append(NoteEvent(t0=t0 + v*beat_to_sec(cfg.bpm, step_beats*2.0), dur=beat_to_sec(cfg.bpm, step_beats*1.85), midi=midi, vel=0.52, chan=v % 6, engine='FM'))

        if i % 4 == 0:
            events.append(NoteEvent(t0=t0, dur=beat_to_sec(cfg.bpm, step_beats*0.9), midi=float(base_midi-12), vel=0.38, chan=1, engine='PSG'))

        t_beats += step_beats
        if beat_to_sec(cfg.bpm, t_beats) >= cfg.seconds:
            break

    return events
