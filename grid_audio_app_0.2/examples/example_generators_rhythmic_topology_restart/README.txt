Topology + Restart-aware rhythm generators

These generators use the optional 3rd argument `context` when available:
    generate(sr, duration, context)

They are designed for builds where context includes:
  - cell_index, cells_total
  - cell_start, cell_duration
  - track_pattern_duration, track_offset
  - track_sync_points_master  (list of master seconds)

TOPOLOGY-AWARE (cell index -> density):
  - topo_euclid_density.py
  - topo_burst_weave.py

RESTART-AWARE (pattern changes after every restart):
  These treat track sync points as 'restart markers' (phase resets).
  Add sync points on a track (e.g. 2,4,6,8...) and the rhythm will mutate:
  - restart_mutating_euclid.py
  - restart_aware_groove_switch.py
