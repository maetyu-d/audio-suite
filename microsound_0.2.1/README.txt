Deep Microsound V2 — packaged

Requirements:
  pip install PyQt6 pyqtgraph numpy soundfile

Run:
  python main_v2.py

What’s inside:
  - main_v2.py            The app
  - presets/              A few starting-point presets
  - irs/                  Tiny example impulse responses (very short, microsound-friendly)

Using presets:
  In the app: Actions / Workflow → Load preset…

Using IRs:
  1) Generator tab → Load IR / space impulse…
  2) You can use an IR as a generator (IR fragment) and/or as a space convolution (Space as Microsound).

Notes:
  - The app is offline-rendering focused (no realtime audio output).
  - Design SR can get very high: Output SR * Unfold factor. If you go extreme + long durations + dense event fields,
    renders will take longer.
