Context-aware generators

Your app now supports generator scripts with either:
  generate(sr, duration) -> np.ndarray
or
  generate(sr, duration, context) -> np.ndarray

The 'context' dict includes:
  track_index, track_name
  cell_index, cells_total
  cell_start (seconds within track pattern), cell_duration
  track_pattern_duration
  track_offset, track_loop_to_master
  track_sync_points_master (list of master seconds)

Note: sync points currently reset track phase at those master times.
Generators don't run per-master-sample; they render per-cell, so
'track_sync_points_master' is provided for *marking* / *designing*
content, not for true dynamic re-render at sync instants.
