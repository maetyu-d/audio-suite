from __future__ import annotations

"""Prime Delay Illusion

Not a true audio delay: instead, schedules echo-notes at prime step offsets.
(You can turn it into a true delay by editing the renderer, but this is easy to hack.)
"""

from examples._common import NoteEvent, RenderConfig, SCALES, beat_to_sec, primes_upto


def generate(cfg: RenderConfig, root_midi: int = 60, scale: str = 'glass',
             step_beats: float = 0.25, primes: str = '17,19,23') -> list[NoteEvent]:
    sc = SCALES.get(scale, SCALES['glass'])
    prime_offsets = [int(x.strip()) for x in primes.split(',') if x.strip()]
    if not prime_offsets:
        prime_offsets = [17, 19, 23]

    cell = [0,1,2,1,3,2,4,3]
    events: list[NoteEvent] = []

    t_beats = 0.0
    n_steps = int(cfg.seconds / beat_to_sec(cfg.bpm, step_beats)) + 1
    for i in range(n_steps):
        t0 = beat_to_sec(cfg.bpm, t_beats)
        deg = cell[i % len(cell)]
        midi = root_midi + sc[deg % len(sc)]
        events.append(NoteEvent(t0=t0, dur=beat_to_sec(cfg.bpm, step_beats*0.9), midi=float(midi), vel=0.75, chan=0, engine='FM'))

        # "echo" notes
        for k, off in enumerate(prime_offsets):
            te = beat_to_sec(cfg.bpm, t_beats + off*step_beats)
            if te < cfg.seconds:
                events.append(NoteEvent(t0=te, dur=beat_to_sec(cfg.bpm, step_beats*0.85), midi=float(midi+12), vel=0.25/(1+k), chan=1, engine='FM'))

        t_beats += step_beats
        if beat_to_sec(cfg.bpm, t_beats) >= cfg.seconds:
            break

    return events
