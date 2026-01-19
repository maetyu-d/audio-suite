import numpy as np
from dataclasses import asdict
from .events import NoteEvent, RenderConfig
from .synth_fm import FMVoiceParams, OpParams, render_fm_note
from .synth_psg import PSGParams, render_psg_note


def apply_time_ops(events: list[NoteEvent], cfg: RenderConfig) -> list[NoteEvent]:
    rng = np.random.default_rng(int(cfg.seed) & 0xFFFFFFFF)
    out: list[NoteEvent] = []
    swing = float(np.clip(cfg.swing, 0.0, 0.5))
    for e in events:
        t0 = float(e.t0) * float(cfg.time_stretch)
        dur = float(e.dur) * float(cfg.time_stretch)

        # Swing: delay every 2nd 16th (approx) using BPM grid
        # (simple: infer 16th length and shift odd steps)
        if swing > 0.0 and cfg.bpm > 0:
            sec_16th = 60.0 / float(cfg.bpm) / 4.0
            if sec_16th > 1e-6:
                idx = int(round(t0 / sec_16th))
                if idx % 2 == 1:
                    t0 += swing * sec_16th

        if cfg.micro_jitter > 0.0:
            t0 += float(rng.normal(0.0, cfg.micro_jitter))
            t0 = max(0.0, t0)

        out.append(NoteEvent(t0=t0, dur=max(1e-4, dur), midi=float(e.midi), vel=float(e.vel), chan=int(e.chan), engine=e.engine))

    return out


class MegaDriveInspiredSynth:
    """A small, creative model of YM2612 FM + SN76489 PSG.

    It's meant for *pattern laboratory* use: stable, fast, tweakable.
    """

    def __init__(self, sr: int, seed: int = 1):
        self.sr = int(sr)
        self.seed = int(seed)

        # channel presets (FM)
        self.fm_channels: list[FMVoiceParams] = [
            FMVoiceParams(algorithm=1, feedback=0.12, lfo_hz=5.0, lfo_depth=0.0),  # glassy
            FMVoiceParams(algorithm=2, feedback=0.05, lfo_hz=6.0, lfo_depth=0.1),  # chorused
            FMVoiceParams(algorithm=1, feedback=0.18, lfo_hz=4.5, lfo_depth=0.0),
            FMVoiceParams(algorithm=3, feedback=0.0, lfo_hz=5.0, lfo_depth=0.0,
                          ops=(
                              OpParams(ratio=1.0, level=0.8, index=0.0, a=0.01, d=0.2, s=0.8, r=0.2),
                              OpParams(ratio=2.0, level=0.45, index=0.0, a=0.01, d=0.2, s=0.8, r=0.2),
                              OpParams(ratio=4.0, level=0.25, index=0.0, a=0.01, d=0.2, s=0.8, r=0.2),
                              OpParams(ratio=8.0, level=0.15, index=0.0, a=0.01, d=0.2, s=0.8, r=0.2),
                          )),
            FMVoiceParams(algorithm=2, feedback=0.2, lfo_hz=7.0, lfo_depth=0.0),
            FMVoiceParams(algorithm=1, feedback=0.0, lfo_hz=5.0, lfo_depth=0.0,
                          ops=(
                              OpParams(ratio=0.5, level=1.0, index=0.0, a=0.01, d=0.35, s=0.65, r=0.2),
                              OpParams(ratio=1.0, level=0.8, index=2.5, a=0.01, d=0.2, s=0.45, r=0.18),
                              OpParams(ratio=2.0, level=0.7, index=2.2, a=0.005, d=0.15, s=0.35, r=0.18),
                              OpParams(ratio=3.0, level=0.6, index=1.7, a=0.003, d=0.12, s=0.25, r=0.22),
                          )),
        ]

        self.psg_channels: list[PSGParams] = [
            PSGParams(noise=False, duty=0.5, a=0.001, d=0.08, s=0.5, r=0.08, bits=10),
            PSGParams(noise=False, duty=0.25, a=0.001, d=0.12, s=0.45, r=0.12, bits=10),
            PSGParams(noise=False, duty=0.75, a=0.001, d=0.1, s=0.35, r=0.1, bits=10),
            PSGParams(noise=True, duty=0.5, a=0.001, d=0.05, s=0.0, r=0.05, bits=8),
        ]

    def set_fm_channel(self, i: int, params: FMVoiceParams):
        self.fm_channels[int(i) % 6] = params

    def set_psg_channel(self, i: int, params: PSGParams):
        self.psg_channels[int(i) % 4] = params

    def render(self, events: list[NoteEvent], seconds: float, master_gain: float = 0.9) -> np.ndarray:
        n_total = int(max(1, round(float(seconds) * self.sr)))
        y = np.zeros(n_total, dtype=np.float32)

        for k, e in enumerate(events):
            start = int(round(float(e.t0) * self.sr))
            # Defensive: if upstream timing ops ever yield a negative start (e.g.
            # due to extreme jitter + rounding), clamp it. Negative slice indices
            # would otherwise wrap from the end of the buffer and can cause
            # confusing broadcast errors.
            if start < 0:
                start = 0
            if start >= n_total:
                continue
            # Hard cap to remaining buffer time *before* synthesis.
            # This avoids edge-case overruns when a voice renders slightly longer than requested.
            remain_s = max(0.0, (n_total - start) / float(self.sr))
            dur = min(float(e.dur), remain_s)
            if dur <= 1e-4:
                continue

            if e.engine.upper() == 'PSG':
                p = self.psg_channels[int(e.chan) % 4]
                note = render_psg_note(self.sr, dur, e.midi, e.vel, p, seed=self.seed + k)
            else:
                p = self.fm_channels[int(e.chan) % 6]
                note = render_fm_note(self.sr, dur, e.midi, e.vel, p)

            # Ultra-robust mix-in: clamp note to remaining buffer samples.
            # Also force 1-D float32 to avoid shape surprises.
            note = np.asarray(note, dtype=np.float32).reshape(-1)
            tail = int(n_total - start)
            if tail <= 0:
                continue
            if int(note.shape[0]) > tail:
                note = note[:tail]
            seg = int(min(int(note.shape[0]), tail))
            if seg <= 0:
                continue
            # Final safety: even if something upstream hands us a longer note buffer,
            # never allow a numpy broadcast error to escape.
            try:
                y[start:start + seg] += note[:seg]
            except ValueError:
                sl = y[start:].shape[0]
                if sl <= 0:
                    continue
                nn = note[:sl]
                y[start:start + nn.shape[0]] += nn

        # soft clip
        y = np.tanh(y).astype(np.float32)
        y *= float(master_gain)
        return y


def render(events: list[NoteEvent], cfg: RenderConfig) -> tuple[np.ndarray, list[NoteEvent]]:
    ev = apply_time_ops(events, cfg)
    synth = MegaDriveInspiredSynth(cfg.sample_rate, seed=cfg.seed)
    y = synth.render(ev, seconds=cfg.seconds, master_gain=cfg.master_gain)
    return y, ev
