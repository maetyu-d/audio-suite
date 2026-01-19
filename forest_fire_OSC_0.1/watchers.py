from dataclasses import dataclass
import time


@dataclass
class ThresholdRule:
    enabled: bool = True

    metric_key: str = "burning"     # stats key, e.g. "burning"
    op: str = ">"                   # ">", "<", "band"
    threshold: float = 100.0
    threshold_hi: float = 200.0     # for band op

    hysteresis: float = 0.0         # prevents chatter around boundary
    cooldown_s: float = 0.25        # minimum time between OSC sends

    edge: str = "rising"            # "rising", "falling", "both", "level"
    osc_address: str = "/fire/trigger"

    send_value: bool = True         # include value
    send_state: bool = True         # include state (0/1)


class RuleState:
    def __init__(self):
        self.active = False
        self.last_send_t = 0.0


class WatchEngine:
    def __init__(self):
        self.rules: list[ThresholdRule] = []
        self._states: list[RuleState] = []

    def set_rules(self, rules: list[ThresholdRule]):
        self.rules = list(rules)
        self._states = [RuleState() for _ in self.rules]

    def _eval_active(self, rule: ThresholdRule, x: float, prev_active: bool) -> bool:
        h = float(rule.hysteresis)

        if rule.op == ">":
            return x > ((rule.threshold - h) if prev_active else (rule.threshold + h))

        if rule.op == "<":
            return x < ((rule.threshold + h) if prev_active else (rule.threshold - h))

        if rule.op == "band":
            lo = min(rule.threshold, rule.threshold_hi)
            hi = max(rule.threshold, rule.threshold_hi)
            if prev_active:
                return (x > (lo - h)) and (x < (hi + h))
            else:
                return (x > (lo + h)) and (x < (hi - h))

        return False

    def update(self, stats: dict, osc_send_fn):
        """Return list of (enabled, active) for LEDs."""
        now = time.perf_counter()
        led_states: list[tuple[bool, bool]] = []

        for i, rule in enumerate(self.rules):
            st = self._states[i]

            if not rule.enabled:
                st.active = False
                led_states.append((False, False))
                continue

            if rule.metric_key not in stats:
                st.active = False
                led_states.append((True, False))
                continue

            x = float(stats[rule.metric_key])
            prev = st.active
            st.active = self._eval_active(rule, x, prev)
            changed = (st.active != prev)

            should_send = False
            if rule.edge == "level":
                should_send = st.active
            elif rule.edge == "both":
                should_send = changed
            elif rule.edge == "rising":
                should_send = (not prev) and st.active
            elif rule.edge == "falling":
                should_send = prev and (not st.active)

            if should_send and (now - st.last_send_t) >= float(rule.cooldown_s):
                st.last_send_t = now
                payload = []
                if rule.send_state:
                    payload.append(1 if st.active else 0)
                if rule.send_value:
                    payload.append(x)
                osc_send_fn(rule.osc_address, *payload)

            led_states.append((True, bool(st.active)))

        return led_states
