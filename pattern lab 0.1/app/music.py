import numpy as np

A4 = 440.0


def midi_to_hz(m: float, a4: float = A4) -> float:
    return float(a4 * (2.0 ** ((m - 69.0) / 12.0)))


def pythagorean_ratio(steps: int) -> float:
    """Return a Pythagorean tuning ratio for a number of perfect-fifth steps.

    We build from 3/2 and fold into [1, 2) by octaves.
    """
    ratio = (3.0 / 2.0) ** steps
    # fold into octave
    while ratio >= 2.0:
        ratio *= 0.5
    while ratio < 1.0:
        ratio *= 2.0
    return float(ratio)


def primes_upto(n: int) -> list[int]:
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:n+1:p] = False
    return [int(i) for i in np.nonzero(sieve)[0].tolist()]


def fibonacci(n: int) -> list[int]:
    if n <= 0:
        return []
    a, b = 1, 1
    out = [a]
    for _ in range(n - 1):
        a, b = b, a + b
        out.append(a)
    return out


def euclidean_rhythm(steps: int, pulses: int, rotate: int = 0) -> np.ndarray:
    """Bjorklund algorithm; returns 0/1 array of length 'steps'."""
    steps = int(max(1, steps))
    pulses = int(np.clip(pulses, 0, steps))
    if pulses == 0:
        pat = np.zeros(steps, dtype=np.int32)
    elif pulses == steps:
        pat = np.ones(steps, dtype=np.int32)
    else:
        pattern = []
        counts = []
        remainders = []
        divisor = steps - pulses
        remainders.append(pulses)
        level = 0
        while True:
            counts.append(divisor // remainders[level])
            remainders.append(divisor % remainders[level])
            divisor = remainders[level]
            level += 1
            if remainders[level] <= 1:
                break
        counts.append(divisor)

        def build(level_: int):
            if level_ == -1:
                pattern.append(0)
            elif level_ == -2:
                pattern.append(1)
            else:
                for _ in range(counts[level_]):
                    build(level_ - 1)
                if remainders[level_] != 0:
                    build(level_ - 2)

        build(level)
        pat = np.array(pattern[:steps], dtype=np.int32)

    if rotate != 0:
        rotate = int(rotate) % steps
        pat = np.roll(pat, rotate)
    return pat


def quantize_to_bits(x: np.ndarray, bits: int) -> np.ndarray:
    # symmetric quantization to +/-1 range
    levels = 2 ** (bits - 1)
    y = np.clip(x, -1.0, 1.0)
    yq = np.round(y * (levels - 1)) / (levels - 1)
    return yq.astype(np.float32)
