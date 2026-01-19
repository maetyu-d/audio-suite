Track restart events

This build adds EVENT cells: any .py assigned to a cell may define:

    def event(context) -> dict

At render time, event() is called at the cell onset (for every occurrence if the track loops),
and can request restarts of other tracks by returning:

    {"restart_tracks": [1,2]}         # indices
    {"restart_tracks": ["Track 2"]}   # names
    {"restart_tracks": "all"}         # all tracks
    {"restart_tracks": "all_except_self"}  # all but the triggering track

Optional:
    {"delay": 0.25}  # seconds after the cell onset

If the .py has generate() too, it still renders audio normally.

Examples (in examples/):
- restart_track2_and_3.py (event-only)
- click_restart_all_except_self.py (audio + event)

Assign one of these scripts to a cell on a track, render, and you'll hear
other tracks snap back to their pattern start at that instant.
