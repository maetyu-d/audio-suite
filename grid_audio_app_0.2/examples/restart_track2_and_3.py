"""
Drop this into a cell on Track 1 (or any track). It emits a restart event that
resets other track(s) phase to the start at the cell's master_time.

You can choose targets by:
- track indices: [1,2]
- track names: ["Track 2"]
- special strings: "all", "all_except_self"

Optional:
- delay (seconds): delay the restart relative to the cell onset.
"""

def event(context: dict) -> dict:
    # Example: restart Track 2 and Track 3 (by index) whenever this cell is hit.
    return {
        "restart_tracks": [1, 2],
        "delay": 0.0
    }
