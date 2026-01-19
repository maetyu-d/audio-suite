# main_v2.py
# Deep Microsound V2: impulses at extreme design SR, unfolded into delicate audio
# PyQt6 + pyqtgraph + numpy + soundfile only.

import sys, os, json, math
import numpy as np
import soundfile as sf

from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg


# ----------------------------
# Helpers / DSP
# ----------------------------

def hann(n):
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    a = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2*np.pi*a/(n-1))

def db(x, eps=1e-12):
    return 20*np.log10(np.maximum(np.abs(x), eps))

def normalize(x, peak=0.98):
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m <= 0: return x
    return x * (peak / m)

def soft_clip(x, drive=1.0):
    drive = float(drive)
    if drive <= 0: return x
    return np.tanh(x*drive) / np.tanh(drive)

def rfft_freqs(n, sr):
    return np.fft.rfftfreq(n, d=1.0/sr)

def lowpass_fft(x, sr, cutoff, roll=0.0):
    # Offline FFT lowpass with optional cosine rolloff
    n = len(x)
    if n < 8: return x
    nyq = 0.5*sr
    cutoff = float(np.clip(cutoff, 1.0, nyq))
    roll = float(max(0.0, roll))
    X = np.fft.rfft(x)
    f = rfft_freqs(n, sr)
    if roll <= 0:
        X[f > cutoff] = 0.0
    else:
        f0, f1 = cutoff, min(nyq, cutoff + roll)
        hard = f > f1
        X[hard] = 0.0
        band = (f >= f0) & (f <= f1)
        if np.any(band):
            t = (f[band] - f0) / max(1e-12, (f1 - f0))
            w = 0.5*(1.0 + np.cos(np.pi*t))  # 1->0
            X[band] *= w
    return np.fft.irfft(X, n=n).astype(np.float64, copy=False)

def bandpass_fft(x, sr, lo, hi, roll=0.0):
    n = len(x)
    if n < 8: return x
    lo = max(0.0, float(lo))
    hi = max(lo, float(hi))
    X = np.fft.rfft(x)
    f = rfft_freqs(n, sr)
    nyq = 0.5*sr
    hi = min(hi, nyq)
    if hi <= 0: return np.zeros_like(x)
    roll = float(max(0.0, roll))
    Y = X.copy()
    # below lo
    if lo > 0:
        if roll <= 0:
            Y[f < lo] = 0.0
        else:
            f0 = max(0.0, lo-roll)
            f1 = lo
            hard = f < f0
            Y[hard] = 0.0
            band = (f >= f0) & (f <= f1)
            if np.any(band):
                t = (f[band] - f0) / max(1e-12, (f1-f0))
                w = 0.5*(1.0 - np.cos(np.pi*t))  # 0->1
                Y[band] *= w
    # above hi
    if hi < nyq:
        if roll <= 0:
            Y[f > hi] = 0.0
        else:
            f0 = hi
            f1 = min(nyq, hi + roll)
            hard = f > f1
            Y[hard] = 0.0
            band = (f >= f0) & (f <= f1)
            if np.any(band):
                t = (f[band] - f0) / max(1e-12, (f1-f0))
                w = 0.5*(1.0 + np.cos(np.pi*t))  # 1->0
                Y[band] *= w
    return np.fft.irfft(Y, n=n).astype(np.float64, copy=False)

def fft_warp_power(x, power):
    n = len(x)
    if n < 16: return x
    X = np.fft.rfft(x)
    k = np.arange(X.size, dtype=np.float64)
    kmax = max(1.0, k[-1])
    u = k / kmax
    u_in = np.power(u, 1.0/max(1e-6, float(power)))
    k_in = u_in * kmax
    re = np.interp(k_in, k, X.real, left=0.0, right=0.0)
    im = np.interp(k_in, k, X.imag, left=0.0, right=0.0)
    Y = re + 1j*im
    return np.fft.irfft(Y, n=n).astype(np.float64, copy=False)

def fft_partial_stretch(x, factor):
    n = len(x)
    if n < 16: return x
    factor = float(factor)
    if abs(factor-1.0) < 1e-9: return x
    X = np.fft.rfft(x)
    k = np.arange(X.size, dtype=np.float64)
    k_in = k / max(1e-12, factor)
    re = np.interp(k_in, k, X.real, left=0.0, right=0.0)
    im = np.interp(k_in, k, X.imag, left=0.0, right=0.0)
    Y = re + 1j*im
    return np.fft.irfft(Y, n=n).astype(np.float64, copy=False)

def partial_lock_stretch(x, factor, top_n=24, neighborhood=4):
    n = len(x)
    if n < 64: return x
    factor = float(factor)
    if abs(factor-1.0) < 1e-9: return x
    X = np.fft.rfft(x)
    mag = np.abs(X)
    idx = np.argsort(mag[1:])[-top_n:] + 1
    Y = np.zeros_like(X)
    for k in idx:
        k2 = int(round(k * factor))
        if 1 <= k2 < Y.size:
            for d in range(-neighborhood, neighborhood+1):
                kk = k2 + d
                if 1 <= kk < Y.size:
                    w = 1.0 - (abs(d)/(neighborhood+1))
                    Y[kk] += X[k] * w
    Y += 0.12 * X
    return np.fft.irfft(Y, n=n).astype(np.float64)

def cepstral_warp(x, factor):
    n = len(x)
    if n < 64: return x
    X = np.fft.rfft(x)
    mag = np.abs(X) + 1e-12
    logmag = np.log(mag)
    cep = np.fft.irfft(logmag, n=n)
    t = np.arange(n, dtype=np.float64)
    t_in = t / max(1e-12, float(factor))
    cep2 = np.interp(t_in, t, cep, left=0.0, right=0.0)
    logmag2 = np.fft.rfft(cep2).real
    mag2 = np.exp(logmag2)
    Y = mag2 * np.exp(1j*np.angle(X))
    return np.fft.irfft(Y, n=n).astype(np.float64)

def morlet_atom(gen_sr, dur_ms, f0, sigma_ms, phase=0.0):
    n = int(max(16, round(gen_sr*dur_ms/1000.0)))
    t = (np.arange(n, dtype=np.float64) - (n/2)) / gen_sr
    sigma = max(1e-9, (sigma_ms/1000.0))
    w = np.exp(-0.5*(t/sigma)**2) * np.cos(2*np.pi*f0*t + phase)
    return w.astype(np.float64)

def make_adsr(n, sr, A_ms, D_ms, S, R_ms, curve=1.8):
    A = int(round(sr*A_ms/1000.0)); D = int(round(sr*D_ms/1000.0))
    R = int(round(sr*R_ms/1000.0))
    A = max(0,A); D = max(0,D); R=max(0,R)
    S = float(np.clip(S,0,1))
    curve = float(max(1e-6, curve))
    env = np.ones(n, dtype=np.float64)
    i = 0
    if A > 0:
        a = np.linspace(0,1,A,endpoint=False)
        env[:A] = a**curve
        i = A
    j = min(n, i + D)
    if D > 0 and j > i:
        d = np.linspace(0,1,j-i,endpoint=False)
        env[i:j] = 1.0 - (1.0-S)*(d**curve)
    sus_start = j
    sus_end = max(sus_start, n-R)
    if sus_end > sus_start:
        env[sus_start:sus_end] = S
    if R > 0 and n > sus_end:
        r = np.linspace(0,1,n-sus_end,endpoint=True)
        env[sus_end:] = S*(1.0 - (r**curve))
    return env

def stft_mag_db(x, sr, win=2048, hop=256, max_frames=3000):
    n = len(x)
    if n < win:
        X = np.fft.rfft(x*hann(n), n=win)
        S = db(X)[:, None]
        return S
    frames = 1 + (n - win)//hop
    frames = min(frames, max_frames)
    w = hann(win)
    S = np.empty((win//2+1, frames), dtype=np.float64)
    for i in range(frames):
        a = i*hop
        seg = x[a:a+win]*w
        X = np.fft.rfft(seg)
        S[:, i] = db(X)
    return S


# ----------------------------
# Generators
# ----------------------------

def gen_basic(gen_sr, micro_ms, seed, mode, dust_density, noise_tilt_db_oct, ring_hz, ring_decay_ms):
    rng = np.random.default_rng(int(seed))
    n = int(max(16, round(gen_sr * micro_ms / 1000.0)))
    t = np.arange(n, dtype=np.float64) / gen_sr

    def tilted_noise(n, tilt_db_per_oct):
        w = rng.standard_normal(n).astype(np.float64)
        W = np.fft.rfft(w)
        f = rfft_freqs(n, gen_sr)
        if f.size > 1: f[0] = f[1]
        gain_per_oct = 10.0 ** (tilt_db_per_oct / 20.0)
        alpha = math.log(gain_per_oct, 2.0)
        shape = (f / max(1e-12, f[1])) ** alpha
        W *= shape
        return np.fft.irfft(W, n=n).astype(np.float64)

    if mode == "Gaussian click":
        sigma = max(1, int(0.0025*n))
        g = np.exp(-0.5*((np.arange(n)/sigma)**2))
        x = g * (rng.standard_normal(n)*0.12 + 1.0)
    elif mode == "Dust impulses":
        x = np.zeros(n, dtype=np.float64)
        k = int(max(1, round(dust_density*n)))
        idx = rng.integers(0, n, size=k)
        x[idx] = rng.uniform(-1,1,size=k)
        ker = np.exp(-np.linspace(0, 6, max(8, int(0.01*n))))
        x = np.convolve(x, ker, mode="same")
    elif mode == "Noise burst":
        w = tilted_noise(n, noise_tilt_db_oct)
        env = np.exp(-t / max(1e-6, (micro_ms/1000.0)*0.25))
        x = w * env
    elif mode == "Skewed transient":
        w = tilted_noise(n, noise_tilt_db_oct)
        w = np.maximum(0.0, w)
        d = np.diff(w, prepend=w[0])
        env = np.exp(-t / max(1e-6, (micro_ms/1000.0)*0.2))
        x = d * env
    elif mode == "Resonant strike":
        f = max(10.0, float(ring_hz))
        tau = max(1e-6, float(ring_decay_ms)/1000.0)
        s = np.sin(2*np.pi*f*t) * np.exp(-t/tau)
        exc = rng.standard_normal(n) * np.exp(-t / max(1e-6, (micro_ms/1000.0)*0.15))
        x = 0.9*s + 0.25*exc
    else:
        x = rng.standard_normal(n).astype(np.float64) * 0.1

    fade = max(8, int(0.01*n))
    w = np.ones(n, dtype=np.float64)
    w[:fade] *= np.linspace(0,1,fade,endpoint=False)
    w[-fade:] *= np.linspace(1,0,fade,endpoint=False)
    return (x*w).astype(np.float64)

def gen_crackle(gen_sr, micro_ms, seed, alpha=1.4, density=180, kernel=64):
    rng = np.random.default_rng(int(seed))
    n = int(max(16, round(gen_sr*micro_ms/1000.0)))
    x = np.zeros(n, dtype=np.float64)
    steps = rng.pareto(alpha, int(max(8, density)))
    times = np.cumsum(steps)
    times = times[times < n]
    for ti in times.astype(int):
        x[ti] += rng.uniform(-1, 1)
    ker = np.exp(-np.linspace(0, 6, max(8, int(kernel))))
    return np.convolve(x, ker, mode="same").astype(np.float64)

def gen_stick_slip(gen_sr, micro_ms, seed, threshold=0.9, build=0.06, decay=0.75, noise=0.08):
    rng = np.random.default_rng(int(seed))
    n = int(max(64, round(gen_sr*micro_ms/1000.0)))
    x = np.zeros(n, dtype=np.float64)
    sticking = True
    force = 0.0
    for i in range(n):
        if sticking:
            force += build*(rng.standard_normal()*noise + 0.2)
            if abs(force) > threshold:
                sticking = False
        else:
            x[i] = force + 0.25*rng.standard_normal()
            force *= decay
            if abs(force) < 0.02:
                sticking = True
                force = 0.0
    x *= hann(n)
    return x.astype(np.float64)

def gen_micro_chaos(gen_sr, micro_ms, seed, r=3.92, gate=0.35):
    rng = np.random.default_rng(int(seed))
    n = int(max(64, round(gen_sr*micro_ms/1000.0)))
    x = np.zeros(n, dtype=np.float64)
    y = (int(seed) % 10000) / 10000.0
    for i in range(n):
        y = r*y*(1.0 - y)
        v = y - 0.5
        if rng.random() < gate:
            x[i] = v
    x = np.convolve(x, np.exp(-np.linspace(0, 5, 48)), mode="same")
    x *= hann(n)
    return x.astype(np.float64)

def gen_wavelet_atoms(gen_sr, micro_ms, seed, base_hz=2400, count=8, spread=0.6):
    rng = np.random.default_rng(int(seed))
    n = int(max(128, round(gen_sr*micro_ms/1000.0)))
    x = np.zeros(n, dtype=np.float64)
    dur = micro_ms
    for k in range(int(max(1,count))):
        f0 = base_hz * (2.0 ** rng.uniform(-spread, spread))
        sigma_ms = max(0.03, dur * rng.uniform(0.04, 0.18))
        phase = rng.uniform(0, 2*np.pi)
        atom = morlet_atom(gen_sr, dur_ms=dur, f0=f0, sigma_ms=sigma_ms, phase=phase)
        shift = rng.integers(-n//8, n//8)
        atom = np.roll(atom, int(shift))
        x += (1.0/(1+k*0.6))*atom[:n]
    x *= hann(n)
    return x.astype(np.float64)

def gen_ir_fragment(ir_audio, gen_sr, micro_ms, seed):
    rng = np.random.default_rng(int(seed))
    if ir_audio is None or ir_audio.size < 32:
        return np.zeros(int(max(16, round(gen_sr*micro_ms/1000.0)))), "No IR loaded"
    n = int(max(64, round(gen_sr*micro_ms/1000.0)))
    src = ir_audio.astype(np.float64)
    if src.ndim > 1:
        src = src.mean(axis=1)
    start = rng.integers(0, max(1, src.size - 256))
    sl = src[start:start+256]
    t_src = np.linspace(0, 1, sl.size)
    t_dst = np.linspace(0, 1, n)
    x = np.interp(t_dst, t_src, sl)
    x *= hann(n)
    x = normalize(x, 0.9)
    return x.astype(np.float64), "IR fragment"

def gen_image_scanline(img_gray, gen_sr, micro_ms, seed):
    rng = np.random.default_rng(int(seed))
    n = int(max(64, round(gen_sr*micro_ms/1000.0)))
    if img_gray is None:
        return np.zeros(n, dtype=np.float64), "No image loaded"
    h, w = img_gray.shape
    y = int(rng.integers(0, h))
    line = img_gray[y, :].astype(np.float64) / 255.0
    line = (line - line.mean()) * 2.0
    x = np.interp(np.linspace(0,1,n), np.linspace(0,1,w), line)
    x *= hann(n)
    x = np.convolve(x, np.exp(-np.linspace(0,5,48)), mode="same")
    return x.astype(np.float64), f"Image line y={y}"


# ----------------------------
# Physical-ish models
# ----------------------------

def resonator_bank(x, sr, modes=24, f_min=120, f_max=12000, decay_ms=80, seed=0):
    rng = np.random.default_rng(int(seed)+321)
    n = len(x)
    if n < 32: return x
    out = np.zeros_like(x)
    t = np.arange(n, dtype=np.float64) / sr
    tau = max(1e-6, decay_ms/1000.0)
    env = np.exp(-t/tau)
    for k in range(int(max(1,modes))):
        f = float(f_min) * ((float(f_max)/max(1.0,float(f_min))) ** (k/max(1,modes-1)))
        f *= 2.0**rng.uniform(-0.02, 0.02)
        ph = rng.uniform(0,2*np.pi)
        carrier = np.sin(2*np.pi*f*t + ph)
        out += (1.0/(1+k*0.35)) * carrier * env
    out = out / max(1e-12, np.max(np.abs(out)))
    return (0.55*x + 0.45*(x*0.0 + out)*np.sign(x)).astype(np.float64)

def waveguide_splinters(x, sr, lines=8, max_ms=8.0, feedback=0.7, seed=0):
    rng = np.random.default_rng(int(seed)+777)
    n = len(x)
    if n < 64: return x
    y = x.copy()
    for _ in range(int(max(1,lines))):
        d = int(max(1, round((rng.uniform(0.4, max_ms)/1000.0)*sr)))
        buf = np.zeros(d, dtype=np.float64)
        wp = 0
        g = feedback * rng.uniform(0.6, 0.98)
        mix = rng.uniform(0.15, 0.45)
        for t in range(n):
            v = y[t] + g*buf[wp]
            buf[wp] = v
            wp = (wp+1) % d
            y[t] = (1.0-mix)*y[t] + mix*v
    return y.astype(np.float64)


# ----------------------------
# Space as microsound
# ----------------------------

def early_reflection_cloud(x, sr, taps=320, max_ms=45, seed=0):
    rng = np.random.default_rng(int(seed)+202)
    n = len(x)
    y = x.copy()
    delays = rng.uniform(0.3, max_ms, size=int(max(1,taps))) / 1000.0
    gains  = rng.uniform(-1.0, 1.0, size=delays.size)
    gains *= np.exp(-delays * 42.0)
    for d, g in zip(delays, gains):
        off = int(round(d * sr))
        if off <= 0: continue
        if off < n:
            y[off:] += g * x[:-off]
    return y.astype(np.float64)

def spectral_diffusion_stereo(x, sr, width=0.6):
    width = float(np.clip(width, 0.0, 1.0))
    n = len(x)
    if n < 64:
        return np.column_stack([x, x])
    dl = int(round((1 + 7*width) * 0.0005 * sr))
    dr = int(round((1 + 9*width) * 0.0007 * sr))
    L = np.roll(x, dl)
    R = np.roll(x, -dr)
    X = np.fft.rfft(R)
    k = np.arange(X.size, dtype=np.float64)
    rot = np.exp(1j * (width * 0.9) * np.sin(2*np.pi*k/max(1.0,k[-1])) )
    R2 = np.fft.irfft(X*rot, n=n)
    return np.column_stack([L, R2]).astype(np.float64)

def convolve_ir_short(x, ir):
    if ir is None or ir.size < 8: return x
    ir = ir.astype(np.float64)
    if ir.ndim > 1:
        ir = ir.mean(axis=1)
    ir = ir[:min(ir.size, 8192)]
    y = np.convolve(x, ir, mode="full")[:len(x)]
    return y.astype(np.float64)


# ----------------------------
# Macro modulation
# ----------------------------

def parse_breakpoints(s):
    pts = []
    s = (s or "").strip()
    if not s:
        return pts
    for part in s.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        t, v = part.split(":")
        try:
            pts.append((float(t.strip()), float(v.strip())))
        except:
            pass
    pts.sort(key=lambda p: p[0])
    return pts

def eval_breakpoints(pts, t, default):
    if not pts:
        return default
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts)-1):
        t0,v0 = pts[i]
        t1,v1 = pts[i+1]
        if t0 <= t <= t1:
            a = (t - t0) / max(1e-12, (t1 - t0))
            return (1-a)*v0 + a*v1
    return default


# ----------------------------
# Unfold variants
# ----------------------------

def unfold_reinterpret(x_gen, base_sr):
    return x_gen.astype(np.float64, copy=False)

def unfold_multiband(x_gen, gen_sr, base_sr, bands_out_hz, unfolds, roll_hz=0.0):
    out = None
    for (lo_out, hi_out), u in zip(bands_out_hz, unfolds):
        lo_g = lo_out * u
        hi_g = hi_out * u
        band = bandpass_fft(x_gen, gen_sr, lo_g, hi_g, roll=roll_hz)
        y = unfold_reinterpret(band, base_sr)
        out = y if out is None else (out + y)
    return out if out is not None else x_gen


# ----------------------------
# Event fields
# ----------------------------

def generate_event_times(process, dur_s, rate, seed, cluster_size=6, cluster_spread_ms=25,
                         hawkes_gain=0.6, hawkes_decay_s=0.25):
    rng = np.random.default_rng(int(seed)+9999)
    times = []

    if process == "Single":
        return [0.0]

    if rate <= 0:
        return [0.0]

    if process == "Poisson":
        t = 0.0
        while t < dur_s:
            t += rng.exponential(1.0/rate)
            if t < dur_s:
                times.append(t)
        return times

    if process == "Clustered":
        parents = []
        t = 0.0
        parent_rate = max(0.1, rate/max(1, cluster_size))
        while t < dur_s:
            t += rng.exponential(1.0/parent_rate)
            if t < dur_s:
                parents.append(t)
        spread = cluster_spread_ms / 1000.0
        for p in parents:
            k = int(max(1, round(rng.uniform(0.6, 1.4)*cluster_size)))
            for _ in range(k):
                tt = p + rng.normal(0.0, spread)
                if 0.0 <= tt < dur_s:
                    times.append(tt)
        times.sort()
        return times

    if process == "Hawkes":
        dt = 0.002
        n = int(math.ceil(dur_s/dt))
        activity = 0.0
        for i in range(n):
            t = i*dt
            activity *= math.exp(-dt/max(1e-6, hawkes_decay_s))
            lam = rate + hawkes_gain*activity*rate
            p = min(0.95, lam*dt)
            if rng.random() < p:
                times.append(t + rng.uniform(0, dt))
                activity += 1.0
        return times

    return times


# ----------------------------
# Feedback
# ----------------------------

class SpectralImprint:
    def __init__(self):
        self.mem = None

    def apply(self, x, amount=0.35, smooth=0.92):
        n = len(x)
        if n < 64 or amount <= 0:
            return x
        X = np.fft.rfft(x)
        mag = np.abs(X)
        if self.mem is None or self.mem.size != mag.size:
            self.mem = mag.copy()
        else:
            self.mem = smooth*self.mem + (1.0-smooth)*mag
        mag2 = (1.0-amount)*mag + amount*self.mem
        Y = mag2 * np.exp(1j*np.angle(X))
        return np.fft.irfft(Y, n=n).astype(np.float64)


# ----------------------------
# Render engine
# ----------------------------

def render(params, progress=None):
    base_sr = int(params["base_sr"])
    out_dur = float(params["out_dur_s"])
    out_n = int(max(1, round(out_dur * base_sr)))

    base_unfold = float(params["time_unfold"])
    base_unfold = max(1.0, base_unfold)

    gen_sr = int(round(base_sr * base_unfold))
    gen_sr = int(np.clip(gen_sr, base_sr, 30_000_000))

    if progress:
        progress(0, f"Output SR {base_sr} Hz | Design SR {gen_sr} Hz")

    bp_density = parse_breakpoints(params["bp_density"])
    bp_unfold  = parse_breakpoints(params["bp_unfold"])
    bp_cutoff  = parse_breakpoints(params["bp_cutoff"])
    bp_stretch = parse_breakpoints(params["bp_stretch"])

    process = params["event_process"]
    rate = float(params["grains_per_sec"])
    times = generate_event_times(
        process, out_dur, rate,
        seed=int(params["seed"]),
        cluster_size=int(params["cluster_size"]),
        cluster_spread_ms=float(params["cluster_spread_ms"]),
        hawkes_gain=float(params["hawkes_gain"]),
        hawkes_decay_s=float(params["hawkes_decay_s"]),
    )
    max_events = int(params["max_grains"])
    times = times[:max_events]

    rng = np.random.default_rng(int(params["seed"]) + 123456)

    out = np.zeros(out_n, dtype=np.float64)

    prev_grain = None
    imprint = SpectralImprint() if params["spectral_imprint_on"] else None

    ir_audio = params.get("_ir_audio")
    img_gray = params.get("_img_gray")

    micro_last = None
    grain_last = None

    for i, t0 in enumerate(times):
        dens = eval_breakpoints(bp_density, t0, default=rate)
        ufac = eval_breakpoints(bp_unfold,  t0, default=base_unfold)
        cutoff_out = eval_breakpoints(bp_cutoff, t0, default=float(params["bandlimit_out_hz"]))
        stretch = eval_breakpoints(bp_stretch, t0, default=float(params["partial_stretch"]))

        amp = 1.0
        if rate > 0:
            amp *= np.clip(dens / max(1e-6, rate), 0.15, 4.0)
        amp *= rng.uniform(1.0 - float(params["grain_amp_rand"]), 1.0 + float(params["grain_amp_rand"]))

        ufac = max(1.0, float(ufac))
        gen_sr_evt = int(round(base_sr * ufac))
        gen_sr_evt = int(np.clip(gen_sr_evt, base_sr, 30_000_000))

        micro_ms = float(params["micro_ms"])

        gmode = params["gen_mode"]
        note = ""
        if gmode in ("Gaussian click","Dust impulses","Noise burst","Skewed transient","Resonant strike"):
            xg = gen_basic(
                gen_sr_evt, micro_ms, int(params["seed"])+i,
                mode=gmode,
                dust_density=float(params["dust_density"]),
                noise_tilt_db_oct=float(params["noise_tilt"]),
                ring_hz=float(params["ring_hz"]),
                ring_decay_ms=float(params["ring_decay_ms"]),
            )
        elif gmode == "Crackle / corona":
            xg = gen_crackle(gen_sr_evt, micro_ms, int(params["seed"])+i,
                             alpha=float(params["crackle_alpha"]),
                             density=float(params["crackle_density"]),
                             kernel=int(params["crackle_kernel"]))
        elif gmode == "Stick–slip friction":
            xg = gen_stick_slip(gen_sr_evt, micro_ms, int(params["seed"])+i,
                                threshold=float(params["ss_threshold"]),
                                build=float(params["ss_build"]),
                                decay=float(params["ss_decay"]),
                                noise=float(params["ss_noise"]))
        elif gmode == "Micro-chaos":
            xg = gen_micro_chaos(gen_sr_evt, micro_ms, int(params["seed"])+i,
                                 r=float(params["chaos_r"]),
                                 gate=float(params["chaos_gate"]))
        elif gmode == "Wavelet atoms":
            xg = gen_wavelet_atoms(gen_sr_evt, micro_ms, int(params["seed"])+i,
                                   base_hz=float(params["wav_base_hz"]),
                                   count=int(params["wav_count"]),
                                   spread=float(params["wav_spread"]))
        elif gmode == "IR fragment":
            xg, note = gen_ir_fragment(ir_audio, gen_sr_evt, micro_ms, int(params["seed"])+i)
        elif gmode == "Image scanline":
            xg, note = gen_image_scanline(img_gray, gen_sr_evt, micro_ms, int(params["seed"])+i)
        else:
            xg = gen_basic(gen_sr_evt, micro_ms, int(params["seed"])+i, "Noise burst", 0.01, -3.0, 4000, 12)

        micro_last = xg.copy()

        if params["bandlimit_on"]:
            cutoff_gen = cutoff_out * ufac
            xg = lowpass_fft(xg, gen_sr_evt, cutoff_gen, roll=float(params["bandlimit_roll_hz"]))

        if params["nl_warp_on"]:
            xg = fft_warp_power(xg, float(params["nl_warp_power"]))
        if params["cep_warp_on"]:
            xg = cepstral_warp(xg, float(params["cep_factor"]))

        if params["partial_lock_on"]:
            xg = partial_lock_stretch(xg, stretch, top_n=int(params["pl_top_n"]), neighborhood=int(params["pl_neigh"]))
        else:
            xg = fft_partial_stretch(xg, stretch)

        if params["res_bank_on"]:
            xg = resonator_bank(xg, gen_sr_evt,
                                modes=int(params["res_modes"]),
                                f_min=float(params["res_fmin"]),
                                f_max=float(params["res_fmax"]),
                                decay_ms=float(params["res_decay_ms"]),
                                seed=int(params["seed"])+i)

        if params["wg_on"]:
            xg = waveguide_splinters(xg, gen_sr_evt,
                                     lines=int(params["wg_lines"]),
                                     max_ms=float(params["wg_max_ms"]),
                                     feedback=float(params["wg_fb"]),
                                     seed=int(params["seed"])+i)

        unfold_mode = params["unfold_mode"]
        if unfold_mode == "Classic reinterpret":
            grain = unfold_reinterpret(xg, base_sr)
        else:
            bands = [(0, float(params["mb_b1"])),
                     (float(params["mb_b1"]), float(params["mb_b2"])),
                     (float(params["mb_b2"]), float(params["mb_b3"]))]
            unfolds = [float(params["mb_u1"]), float(params["mb_u2"]), float(params["mb_u3"])]
            grain = unfold_multiband(xg, gen_sr_evt, base_sr, bands, unfolds, roll_hz=float(params["mb_roll"]))

        grain_last = grain.copy()

        if params["event_feedback_on"] and prev_grain is not None:
            fb = float(params["event_feedback_amt"])
            L = min(len(grain), len(prev_grain))
            grain[:L] = (1.0-fb)*grain[:L] + fb*prev_grain[:L]

        if imprint is not None:
            grain = imprint.apply(grain, amount=float(params["spectral_imprint_amt"]),
                                  smooth=float(params["spectral_imprint_smooth"]))

        prev_grain = grain.copy()

        start = int(round(t0 * base_sr))
        if start >= out_n:
            continue

        offset = 0
        if params["grain_offset_on"]:
            max_off = int(round((float(params["grain_offset_max_ms"])/1000.0)*base_sr))
            if max_off > 0:
                offset = int(rng.integers(0, max(1, min(max_off, grain.size))))
        g = grain[offset:]

        L = min(out_n - start, g.size)
        if L > 0:
            out[start:start+L] += amp * g[:L]

        if progress and (i % 50 == 0):
            progress(int(5 + 70*(i/max(1,len(times)))), f"Events {i}/{len(times)}  {note}".strip())

    env = make_adsr(out_n, base_sr,
                    float(params["env_a"]), float(params["env_d"]),
                    float(params["env_s"]), float(params["env_r"]),
                    float(params["env_curve"]))
    out *= env

    if params["er_cloud_on"]:
        out = early_reflection_cloud(out, base_sr,
                                     taps=int(params["er_taps"]),
                                     max_ms=float(params["er_max_ms"]),
                                     seed=int(params["seed"]))

    if params["space_ir_on"] and params.get("_ir_audio") is not None:
        out = convolve_ir_short(out, params["_ir_audio"][:int(params["space_ir_max_samps"])])

    if params["stereo_on"]:
        stereo = spectral_diffusion_stereo(out, base_sr, width=float(params["stereo_width"]))
    else:
        stereo = np.column_stack([out, out])

    stereo = soft_clip(stereo, drive=float(params["sat_drive"]))
    stereo = normalize(stereo, peak=float(params["peak"]))

    if progress:
        progress(100, "Done.")

    meta = {
        "out_sr": base_sr,
        "design_sr_base": gen_sr,
        "micro_last": micro_last,
        "grain_last": grain_last,
    }
    return stereo.astype(np.float64), meta


# ----------------------------
# UI
# ----------------------------

class RenderWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    @QtCore.pyqtSlot()
    def run(self):
        try:
            audio, meta = render(self.params, progress=self.progress.emit)
            self.finished.emit(audio, meta)
        except Exception as e:
            self.failed.emit(str(e))


class MicrosoundV2(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Microsound V2 — Transient Unfolder")
        self.resize(1320, 820)

        pg.setConfigOptions(antialias=True)
        self.audio = None
        self.meta = None

        self.ir_audio = None
        self.img_gray = None

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        main = QtWidgets.QHBoxLayout(root)

        self.ctrl = QtWidgets.QScrollArea()
        self.ctrl.setWidgetResizable(True)
        self.ctrl_inner = QtWidgets.QWidget()
        self.ctrl.setWidget(self.ctrl_inner)
        self.ctrl_layout = QtWidgets.QVBoxLayout(self.ctrl_inner)

        self.tabs = QtWidgets.QTabWidget()

        main.addWidget(self.ctrl, 0)
        main.addWidget(self.tabs, 1)

        wf = QtWidgets.QWidget(); wfl = QtWidgets.QVBoxLayout(wf)
        self.wave = pg.PlotWidget();
        self.wave.setLabel("bottom", "Time", units="s")
        self.wave.setLabel("left", "Amplitude")
        self.wave_curveL = self.wave.plot([], [])
        self.wave_curveR = self.wave.plot([], [])
        wfl.addWidget(self.wave)

        sp = QtWidgets.QWidget(); spl = QtWidgets.QVBoxLayout(sp)
        self.spec = pg.PlotWidget()
        self.spec.setLabel("bottom", "Frequency", units="Hz")
        self.spec.setLabel("left", "Magnitude", units="dB")
        self.spec_curve = self.spec.plot([], [])
        spl.addWidget(self.spec)

        sg = QtWidgets.QWidget(); sgl = QtWidgets.QVBoxLayout(sg)
        self.simg = pg.ImageView(); sgl.addWidget(self.simg)

        ms = QtWidgets.QWidget(); msl = QtWidgets.QVBoxLayout(ms)
        self.micro_plot = pg.PlotWidget()
        self.micro_plot.setLabel("bottom", "Samples (design SR view)")
        self.micro_plot.setLabel("left", "Amplitude")
        self.micro_curve = self.micro_plot.plot([], [])
        self.grain_plot = pg.PlotWidget()
        self.grain_plot.setLabel("bottom", "Samples (unfolded grain at output SR)")
        self.grain_plot.setLabel("left", "Amplitude")
        self.grain_curve = self.grain_plot.plot([], [])
        msl.addWidget(self.micro_plot)
        msl.addWidget(self.grain_plot)

        self.tabs.addTab(wf, "Waveform")
        self.tabs.addTab(sp, "Spectrum")
        self.tabs.addTab(sg, "Spectrogram")
        self.tabs.addTab(ms, "Microscope")

        def group(title):
            gb = QtWidgets.QGroupBox(title)
            v = QtWidgets.QVBoxLayout(gb)
            return gb, v

        def row(vbox, label, widget, minw=175):
            w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(0,0,0,0)
            lab = QtWidgets.QLabel(label); lab.setMinimumWidth(minw)
            h.addWidget(lab); h.addWidget(widget, 1)
            vbox.addWidget(w)
            return widget

        gb, v = group("Output")
        self.base_sr = QtWidgets.QComboBox()
        self.base_sr.addItems(["44100","48000","88200","96000","176400","192000"])
        self.base_sr.setCurrentText("48000")
        row(v, "Output SR", self.base_sr)
        self.out_dur = QtWidgets.QDoubleSpinBox(); self.out_dur.setRange(0.05, 240.0)
        self.out_dur.setValue(8.0); self.out_dur.setSingleStep(0.25)
        row(v, "Duration (s)", self.out_dur)
        self.time_unfold = QtWidgets.QDoubleSpinBox(); self.time_unfold.setRange(1.0, 200.0)
        self.time_unfold.setValue(25.0); self.time_unfold.setSingleStep(1.0)
        row(v, "Base unfold", self.time_unfold)
        self.peak = QtWidgets.QDoubleSpinBox(); self.peak.setRange(0.05, 0.999)
        self.peak.setValue(0.98); self.peak.setSingleStep(0.01)
        row(v, "Normalize peak", self.peak)
        self.sat_drive = QtWidgets.QDoubleSpinBox(); self.sat_drive.setRange(0.0, 6.0)
        self.sat_drive.setValue(1.0); self.sat_drive.setSingleStep(0.1)
        row(v, "Saturation", self.sat_drive)
        self.stereo_on = QtWidgets.QCheckBox("Stereo diffusion"); self.stereo_on.setChecked(True)
        v.addWidget(self.stereo_on)
        self.stereo_width = QtWidgets.QDoubleSpinBox(); self.stereo_width.setRange(0.0, 1.0)
        self.stereo_width.setValue(0.65); self.stereo_width.setSingleStep(0.05)
        row(v, "Stereo width", self.stereo_width)
        self.ctrl_layout.addWidget(gb)

        gb, v = group("Generator (micro event)")
        self.gen_mode = QtWidgets.QComboBox()
        self.gen_mode.addItems([
            "Gaussian click","Dust impulses","Noise burst","Skewed transient","Resonant strike",
            "Crackle / corona","Stick–slip friction","Micro-chaos","Wavelet atoms",
            "IR fragment","Image scanline"
        ])
        row(v, "Mode", self.gen_mode)
        self.micro_ms = QtWidgets.QDoubleSpinBox(); self.micro_ms.setRange(0.05, 80.0)
        self.micro_ms.setValue(1.25); self.micro_ms.setSingleStep(0.1)
        row(v, "Micro window (ms)", self.micro_ms)
        self.seed = QtWidgets.QSpinBox(); self.seed.setRange(0, 2_000_000_000); self.seed.setValue(12345)
        row(v, "Seed", self.seed)
        self.dust_density = QtWidgets.QDoubleSpinBox(); self.dust_density.setRange(0.0, 0.2)
        self.dust_density.setValue(0.02); self.dust_density.setSingleStep(0.005)
        row(v, "Dust density", self.dust_density)
        self.noise_tilt = QtWidgets.QDoubleSpinBox(); self.noise_tilt.setRange(-12.0, 12.0)
        self.noise_tilt.setValue(-3.0); self.noise_tilt.setSingleStep(0.5)
        row(v, "Noise tilt (dB/oct)", self.noise_tilt)
        self.ring_hz = QtWidgets.QDoubleSpinBox(); self.ring_hz.setRange(10.0, 200000.0)
        self.ring_hz.setValue(4200.0); self.ring_hz.setSingleStep(10.0)
        row(v, "Ring Hz", self.ring_hz)
        self.ring_decay = QtWidgets.QDoubleSpinBox(); self.ring_decay.setRange(0.1, 300.0)
        self.ring_decay.setValue(12.0); self.ring_decay.setSingleStep(1.0)
        row(v, "Ring decay (ms)", self.ring_decay)

        self.crackle_alpha = QtWidgets.QDoubleSpinBox(); self.crackle_alpha.setRange(1.05, 3.0)
        self.crackle_alpha.setValue(1.4); self.crackle_alpha.setSingleStep(0.05)
        row(v, "Crackle alpha", self.crackle_alpha)
        self.crackle_density = QtWidgets.QDoubleSpinBox(); self.crackle_density.setRange(20.0, 2000.0)
        self.crackle_density.setValue(180.0); self.crackle_density.setSingleStep(10.0)
        row(v, "Crackle density", self.crackle_density)
        self.crackle_kernel = QtWidgets.QSpinBox(); self.crackle_kernel.setRange(8, 256)
        self.crackle_kernel.setValue(64)
        row(v, "Crackle kernel", self.crackle_kernel)

        self.ss_threshold = QtWidgets.QDoubleSpinBox(); self.ss_threshold.setRange(0.1, 3.0)
        self.ss_threshold.setValue(0.9); self.ss_threshold.setSingleStep(0.05)
        row(v, "SS threshold", self.ss_threshold)
        self.ss_build = QtWidgets.QDoubleSpinBox(); self.ss_build.setRange(0.001, 0.3)
        self.ss_build.setValue(0.06); self.ss_build.setSingleStep(0.01)
        row(v, "SS build", self.ss_build)
        self.ss_decay = QtWidgets.QDoubleSpinBox(); self.ss_decay.setRange(0.1, 0.99)
        self.ss_decay.setValue(0.75); self.ss_decay.setSingleStep(0.02)
        row(v, "SS decay", self.ss_decay)
        self.ss_noise = QtWidgets.QDoubleSpinBox(); self.ss_noise.setRange(0.0, 1.0)
        self.ss_noise.setValue(0.08); self.ss_noise.setSingleStep(0.02)
        row(v, "SS noise", self.ss_noise)

        self.chaos_r = QtWidgets.QDoubleSpinBox(); self.chaos_r.setRange(3.2, 3.99)
        self.chaos_r.setValue(3.92); self.chaos_r.setSingleStep(0.01)
        row(v, "Chaos r", self.chaos_r)
        self.chaos_gate = QtWidgets.QDoubleSpinBox(); self.chaos_gate.setRange(0.01, 1.0)
        self.chaos_gate.setValue(0.35); self.chaos_gate.setSingleStep(0.02)
        row(v, "Chaos gate", self.chaos_gate)

        self.wav_base_hz = QtWidgets.QDoubleSpinBox(); self.wav_base_hz.setRange(20.0, 200000.0)
        self.wav_base_hz.setValue(2400.0); self.wav_base_hz.setSingleStep(50.0)
        row(v, "Wavelet base Hz", self.wav_base_hz)
        self.wav_count = QtWidgets.QSpinBox(); self.wav_count.setRange(1, 64); self.wav_count.setValue(8)
        row(v, "Wavelet count", self.wav_count)
        self.wav_spread = QtWidgets.QDoubleSpinBox(); self.wav_spread.setRange(0.0, 3.0)
        self.wav_spread.setValue(0.6); self.wav_spread.setSingleStep(0.1)
        row(v, "Wavelet spread", self.wav_spread)

        self.btn_load_ir = QtWidgets.QPushButton("Load IR / space impulse…")
        self.btn_load_img = QtWidgets.QPushButton("Load image…")
        v.addWidget(self.btn_load_ir); v.addWidget(self.btn_load_img)
        self.ctrl_layout.addWidget(gb)

        gb, v = group("Unfold + Spectral Warps")
        self.unfold_mode = QtWidgets.QComboBox(); self.unfold_mode.addItems(["Classic reinterpret", "Multi-band unfold"])
        row(v, "Unfold mode", self.unfold_mode)
        self.partial_stretch = QtWidgets.QDoubleSpinBox(); self.partial_stretch.setRange(0.25, 4.0)
        self.partial_stretch.setValue(1.0); self.partial_stretch.setSingleStep(0.05)
        row(v, "Stretch factor", self.partial_stretch)
        self.partial_lock_on = QtWidgets.QCheckBox("Partial locking (peak-only)")
        self.partial_lock_on.setChecked(False); v.addWidget(self.partial_lock_on)
        self.pl_top_n = QtWidgets.QSpinBox(); self.pl_top_n.setRange(4, 200); self.pl_top_n.setValue(24)
        row(v, "Lock peaks (N)", self.pl_top_n)
        self.pl_neigh = QtWidgets.QSpinBox(); self.pl_neigh.setRange(0, 16); self.pl_neigh.setValue(4)
        row(v, "Peak spread", self.pl_neigh)
        self.nl_warp_on = QtWidgets.QCheckBox("Nonlinear freq warp"); self.nl_warp_on.setChecked(False)
        v.addWidget(self.nl_warp_on)
        self.nl_warp_power = QtWidgets.QDoubleSpinBox(); self.nl_warp_power.setRange(0.25, 4.0)
        self.nl_warp_power.setValue(1.25); self.nl_warp_power.setSingleStep(0.05)
        row(v, "Warp power", self.nl_warp_power)
        self.cep_warp_on = QtWidgets.QCheckBox("Cepstral warp"); self.cep_warp_on.setChecked(False)
        v.addWidget(self.cep_warp_on)
        self.cep_factor = QtWidgets.QDoubleSpinBox(); self.cep_factor.setRange(0.25, 4.0)
        self.cep_factor.setValue(1.2); self.cep_factor.setSingleStep(0.05)
        row(v, "Cep factor", self.cep_factor)

        self.mb_b1 = QtWidgets.QDoubleSpinBox(); self.mb_b2 = QtWidgets.QDoubleSpinBox(); self.mb_b3 = QtWidgets.QDoubleSpinBox()
        for w, val in [(self.mb_b1, 2000.0), (self.mb_b2, 8000.0), (self.mb_b3, 20000.0)]:
            w.setRange(200.0, 22050.0); w.setValue(val); w.setSingleStep(100.0)
        row(v, "Band1 hi Hz", self.mb_b1)
        row(v, "Band2 hi Hz", self.mb_b2)
        row(v, "Band3 hi Hz", self.mb_b3)
        self.mb_u1 = QtWidgets.QDoubleSpinBox(); self.mb_u2 = QtWidgets.QDoubleSpinBox(); self.mb_u3 = QtWidgets.QDoubleSpinBox()
        for w, val in [(self.mb_u1, 35.0), (self.mb_u2, 20.0), (self.mb_u3, 12.0)]:
            w.setRange(1.0, 200.0); w.setValue(val); w.setSingleStep(1.0)
        row(v, "Band1 unfold", self.mb_u1)
        row(v, "Band2 unfold", self.mb_u2)
        row(v, "Band3 unfold", self.mb_u3)
        self.mb_roll = QtWidgets.QDoubleSpinBox(); self.mb_roll.setRange(0.0, 20000.0)
        self.mb_roll.setValue(2000.0); self.mb_roll.setSingleStep(100.0)
        row(v, "MB rolloff Hz", self.mb_roll)
        self.ctrl_layout.addWidget(gb)

        gb, v = group("Bandlimit Before Unfold")
        self.band_on = QtWidgets.QCheckBox("Enable bandlimit"); self.band_on.setChecked(True)
        v.addWidget(self.band_on)
        self.band_out = QtWidgets.QDoubleSpinBox(); self.band_out.setRange(200.0, 22050.0)
        self.band_out.setValue(18000.0); self.band_out.setSingleStep(100.0)
        row(v, "Max output Hz", self.band_out)
        self.band_roll = QtWidgets.QDoubleSpinBox(); self.band_roll.setRange(0.0, 20000.0)
        self.band_roll.setValue(2500.0); self.band_roll.setSingleStep(100.0)
        row(v, "Rolloff Hz", self.band_roll)
        self.ctrl_layout.addWidget(gb)

        gb, v = group("Event Field")
        self.event_process = QtWidgets.QComboBox(); self.event_process.addItems(["Single","Poisson","Clustered","Hawkes"])
        row(v, "Process", self.event_process)
        self.grains_per_sec = QtWidgets.QDoubleSpinBox(); self.grains_per_sec.setRange(0.0, 500.0)
        self.grains_per_sec.setValue(18.0); self.grains_per_sec.setSingleStep(1.0)
        row(v, "Rate (events/s)", self.grains_per_sec)
        self.max_grains = QtWidgets.QSpinBox(); self.max_grains.setRange(1, 50000); self.max_grains.setValue(4000)
        row(v, "Max events", self.max_grains)
        self.grain_amp_rand = QtWidgets.QDoubleSpinBox(); self.grain_amp_rand.setRange(0.0, 1.0)
        self.grain_amp_rand.setValue(0.35); self.grain_amp_rand.setSingleStep(0.05)
        row(v, "Amp randomness", self.grain_amp_rand)
        self.grain_offset_on = QtWidgets.QCheckBox("Random offset into grain"); self.grain_offset_on.setChecked(True)
        v.addWidget(self.grain_offset_on)
        self.grain_offset_max = QtWidgets.QDoubleSpinBox(); self.grain_offset_max.setRange(0.0, 500.0)
        self.grain_offset_max.setValue(60.0); self.grain_offset_max.setSingleStep(5.0)
        row(v, "Max offset (ms)", self.grain_offset_max)
        self.cluster_size = QtWidgets.QSpinBox(); self.cluster_size.setRange(2, 64); self.cluster_size.setValue(6)
        row(v, "Cluster size", self.cluster_size)
        self.cluster_spread = QtWidgets.QDoubleSpinBox(); self.cluster_spread.setRange(1.0, 200.0)
        self.cluster_spread.setValue(25.0); self.cluster_spread.setSingleStep(1.0)
        row(v, "Cluster spread (ms)", self.cluster_spread)
        self.hawkes_gain = QtWidgets.QDoubleSpinBox(); self.hawkes_gain.setRange(0.0, 5.0)
        self.hawkes_gain.setValue(0.6); self.hawkes_gain.setSingleStep(0.05)
        row(v, "Hawkes gain", self.hawkes_gain)
        self.hawkes_decay = QtWidgets.QDoubleSpinBox(); self.hawkes_decay.setRange(0.01, 2.0)
        self.hawkes_decay.setValue(0.25); self.hawkes_decay.setSingleStep(0.01)
        row(v, "Hawkes decay (s)", self.hawkes_decay)
        self.ctrl_layout.addWidget(gb)

        gb, v = group("Macro Modulation (breakpoints)")
        hint = QtWidgets.QLabel('Format: "time:value, time:value" in seconds. Example: 0:18, 3:45, 8:12')
        hint.setWordWrap(True); v.addWidget(hint)
        self.bp_density = QtWidgets.QLineEdit("0:18, 4:40, 8:14"); row(v, "Density lane", self.bp_density)
        self.bp_unfold = QtWidgets.QLineEdit(""); row(v, "Unfold lane", self.bp_unfold)
        self.bp_cutoff = QtWidgets.QLineEdit(""); row(v, "Cutoff lane", self.bp_cutoff)
        self.bp_stretch = QtWidgets.QLineEdit(""); row(v, "Stretch lane", self.bp_stretch)
        self.ctrl_layout.addWidget(gb)

        gb, v = group("Physical-ish Micro Models")
        self.res_bank_on = QtWidgets.QCheckBox("Resonator bank"); self.res_bank_on.setChecked(False)
        v.addWidget(self.res_bank_on)
        self.res_modes = QtWidgets.QSpinBox(); self.res_modes.setRange(4, 128); self.res_modes.setValue(24)
        row(v, "Modes", self.res_modes)
        self.res_fmin = QtWidgets.QDoubleSpinBox(); self.res_fmin.setRange(10.0, 5000.0); self.res_fmin.setValue(120.0)
        row(v, "F min", self.res_fmin)
        self.res_fmax = QtWidgets.QDoubleSpinBox(); self.res_fmax.setRange(200.0, 100000.0); self.res_fmax.setValue(12000.0)
        row(v, "F max", self.res_fmax)
        self.res_decay = QtWidgets.QDoubleSpinBox(); self.res_decay.setRange(1.0, 2000.0); self.res_decay.setValue(80.0)
        row(v, "Decay (ms)", self.res_decay)
        self.wg_on = QtWidgets.QCheckBox("Waveguide splinters"); self.wg_on.setChecked(False)
        v.addWidget(self.wg_on)
        self.wg_lines = QtWidgets.QSpinBox(); self.wg_lines.setRange(1, 32); self.wg_lines.setValue(8)
        row(v, "Lines", self.wg_lines)
        self.wg_max = QtWidgets.QDoubleSpinBox(); self.wg_max.setRange(0.5, 50.0); self.wg_max.setValue(8.0)
        row(v, "Max delay (ms)", self.wg_max)
        self.wg_fb = QtWidgets.QDoubleSpinBox(); self.wg_fb.setRange(0.0, 0.98); self.wg_fb.setValue(0.7)
        row(v, "Feedback", self.wg_fb)
        self.ctrl_layout.addWidget(gb)

        gb, v = group("Feedback / Recursion")
        self.ev_fb_on = QtWidgets.QCheckBox("Event-to-event feedback"); self.ev_fb_on.setChecked(False)
        v.addWidget(self.ev_fb_on)
        self.ev_fb_amt = QtWidgets.QDoubleSpinBox(); self.ev_fb_amt.setRange(0.0, 0.99)
        self.ev_fb_amt.setValue(0.35); self.ev_fb_amt.setSingleStep(0.05)
        row(v, "Event feedback", self.ev_fb_amt)
        self.imprint_on = QtWidgets.QCheckBox("Spectral imprint feedback"); self.imprint_on.setChecked(False)
        v.addWidget(self.imprint_on)
        self.imprint_amt = QtWidgets.QDoubleSpinBox(); self.imprint_amt.setRange(0.0, 1.0)
        self.imprint_amt.setValue(0.35); self.imprint_amt.setSingleStep(0.05)
        row(v, "Imprint amount", self.imprint_amt)
        self.imprint_smooth = QtWidgets.QDoubleSpinBox(); self.imprint_smooth.setRange(0.0, 0.999)
        self.imprint_smooth.setValue(0.92); self.imprint_smooth.setSingleStep(0.01)
        row(v, "Imprint smooth", self.imprint_smooth)
        self.ctrl_layout.addWidget(gb)

        gb, v = group("Space as Microsound")
        self.er_on = QtWidgets.QCheckBox("Early reflection microcloud"); self.er_on.setChecked(True)
        v.addWidget(self.er_on)
        self.er_taps = QtWidgets.QSpinBox(); self.er_taps.setRange(16, 2000); self.er_taps.setValue(320)
        row(v, "Taps", self.er_taps)
        self.er_max = QtWidgets.QDoubleSpinBox(); self.er_max.setRange(5.0, 150.0); self.er_max.setValue(45.0)
        row(v, "Max delay (ms)", self.er_max)
        self.space_ir_on = QtWidgets.QCheckBox("Convolve with loaded IR (short)"); self.space_ir_on.setChecked(False)
        v.addWidget(self.space_ir_on)
        self.space_ir_max = QtWidgets.QSpinBox(); self.space_ir_max.setRange(256, 200000); self.space_ir_max.setValue(12000)
        row(v, "IR max samples", self.space_ir_max)
        self.ctrl_layout.addWidget(gb)

        gb, v = group("Overall Envelope")
        self.env_a = QtWidgets.QDoubleSpinBox(); self.env_a.setRange(0, 5000); self.env_a.setValue(20); self.env_a.setSingleStep(10)
        row(v, "Attack (ms)", self.env_a)
        self.env_d = QtWidgets.QDoubleSpinBox(); self.env_d.setRange(0, 5000); self.env_d.setValue(250); self.env_d.setSingleStep(10)
        row(v, "Decay (ms)", self.env_d)
        self.env_s = QtWidgets.QDoubleSpinBox(); self.env_s.setRange(0, 1); self.env_s.setValue(0.65); self.env_s.setSingleStep(0.05)
        row(v, "Sustain", self.env_s)
        self.env_r = QtWidgets.QDoubleSpinBox(); self.env_r.setRange(0, 20000); self.env_r.setValue(1800); self.env_r.setSingleStep(50)
        row(v, "Release (ms)", self.env_r)
        self.env_curve = QtWidgets.QDoubleSpinBox(); self.env_curve.setRange(0.2, 6.0); self.env_curve.setValue(1.8); self.env_curve.setSingleStep(0.1)
        row(v, "Curve", self.env_curve)
        self.ctrl_layout.addWidget(gb)

        gb, v = group("Actions / Workflow")
        self.btn_render = QtWidgets.QPushButton("Render")
        self.btn_export = QtWidgets.QPushButton("Export WAV…"); self.btn_export.setEnabled(False)
        self.btn_save_preset = QtWidgets.QPushButton("Save preset…")
        self.btn_load_preset = QtWidgets.QPushButton("Load preset…")
        self.btn_batch = QtWidgets.QPushButton("Batch render…")
        v.addWidget(self.btn_render); v.addWidget(self.btn_export)
        v.addSpacing(6)
        v.addWidget(self.btn_save_preset); v.addWidget(self.btn_load_preset); v.addWidget(self.btn_batch)
        self.status = QtWidgets.QLabel("Ready."); self.status.setWordWrap(True)
        v.addWidget(self.status)
        self.pbar = QtWidgets.QProgressBar(); self.pbar.setRange(0,100); self.pbar.setValue(0)
        v.addWidget(self.pbar)
        self.ctrl_layout.addWidget(gb)
        self.ctrl_layout.addStretch(1)

        self.btn_render.clicked.connect(self.on_render)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_load_ir.clicked.connect(self.on_load_ir)
        self.btn_load_img.clicked.connect(self.on_load_img)
        self.btn_save_preset.clicked.connect(self.on_save_preset)
        self.btn_load_preset.clicked.connect(self.on_load_preset)
        self.btn_batch.clicked.connect(self.on_batch)

        # Snapshot factory defaults so loading a partial preset still updates *all* UI fields
        self._factory_defaults = self.get_params()

    def get_params(self):
        return {
            "base_sr": int(self.base_sr.currentText()),
            "out_dur_s": float(self.out_dur.value()),
            "time_unfold": float(self.time_unfold.value()),
            "peak": float(self.peak.value()),
            "sat_drive": float(self.sat_drive.value()),
            "stereo_on": bool(self.stereo_on.isChecked()),
            "stereo_width": float(self.stereo_width.value()),

            "gen_mode": self.gen_mode.currentText(),
            "micro_ms": float(self.micro_ms.value()),
            "seed": int(self.seed.value()),
            "dust_density": float(self.dust_density.value()),
            "noise_tilt": float(self.noise_tilt.value()),
            "ring_hz": float(self.ring_hz.value()),
            "ring_decay_ms": float(self.ring_decay.value()),

            "crackle_alpha": float(self.crackle_alpha.value()),
            "crackle_density": float(self.crackle_density.value()),
            "crackle_kernel": int(self.crackle_kernel.value()),

            "ss_threshold": float(self.ss_threshold.value()),
            "ss_build": float(self.ss_build.value()),
            "ss_decay": float(self.ss_decay.value()),
            "ss_noise": float(self.ss_noise.value()),

            "chaos_r": float(self.chaos_r.value()),
            "chaos_gate": float(self.chaos_gate.value()),

            "wav_base_hz": float(self.wav_base_hz.value()),
            "wav_count": int(self.wav_count.value()),
            "wav_spread": float(self.wav_spread.value()),

            "unfold_mode": self.unfold_mode.currentText(),
            "partial_stretch": float(self.partial_stretch.value()),
            "partial_lock_on": bool(self.partial_lock_on.isChecked()),
            "pl_top_n": int(self.pl_top_n.value()),
            "pl_neigh": int(self.pl_neigh.value()),
            "nl_warp_on": bool(self.nl_warp_on.isChecked()),
            "nl_warp_power": float(self.nl_warp_power.value()),
            "cep_warp_on": bool(self.cep_warp_on.isChecked()),
            "cep_factor": float(self.cep_factor.value()),

            "mb_b1": float(self.mb_b1.value()),
            "mb_b2": float(self.mb_b2.value()),
            "mb_b3": float(self.mb_b3.value()),
            "mb_u1": float(self.mb_u1.value()),
            "mb_u2": float(self.mb_u2.value()),
            "mb_u3": float(self.mb_u3.value()),
            "mb_roll": float(self.mb_roll.value()),

            "bandlimit_on": bool(self.band_on.isChecked()),
            "bandlimit_out_hz": float(self.band_out.value()),
            "bandlimit_roll_hz": float(self.band_roll.value()),

            "event_process": self.event_process.currentText(),
            "grains_per_sec": float(self.grains_per_sec.value()),
            "max_grains": int(self.max_grains.value()),
            "grain_amp_rand": float(self.grain_amp_rand.value()),
            "grain_offset_on": bool(self.grain_offset_on.isChecked()),
            "grain_offset_max_ms": float(self.grain_offset_max.value()),
            "cluster_size": int(self.cluster_size.value()),
            "cluster_spread_ms": float(self.cluster_spread.value()),
            "hawkes_gain": float(self.hawkes_gain.value()),
            "hawkes_decay_s": float(self.hawkes_decay.value()),

            "bp_density": self.bp_density.text(),
            "bp_unfold": self.bp_unfold.text(),
            "bp_cutoff": self.bp_cutoff.text(),
            "bp_stretch": self.bp_stretch.text(),

            "res_bank_on": bool(self.res_bank_on.isChecked()),
            "res_modes": int(self.res_modes.value()),
            "res_fmin": float(self.res_fmin.value()),
            "res_fmax": float(self.res_fmax.value()),
            "res_decay_ms": float(self.res_decay.value()),

            "wg_on": bool(self.wg_on.isChecked()),
            "wg_lines": int(self.wg_lines.value()),
            "wg_max_ms": float(self.wg_max.value()),
            "wg_fb": float(self.wg_fb.value()),

            "event_feedback_on": bool(self.ev_fb_on.isChecked()),
            "event_feedback_amt": float(self.ev_fb_amt.value()),
            "spectral_imprint_on": bool(self.imprint_on.isChecked()),
            "spectral_imprint_amt": float(self.imprint_amt.value()),
            "spectral_imprint_smooth": float(self.imprint_smooth.value()),

            "er_cloud_on": bool(self.er_on.isChecked()),
            "er_taps": int(self.er_taps.value()),
            "er_max_ms": float(self.er_max.value()),
            "space_ir_on": bool(self.space_ir_on.isChecked()),
            "space_ir_max_samps": int(self.space_ir_max.value()),

            "env_a": float(self.env_a.value()),
            "env_d": float(self.env_d.value()),
            "env_s": float(self.env_s.value()),
            "env_r": float(self.env_r.value()),
            "env_curve": float(self.env_curve.value()),
        }

    def on_save_preset(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save preset", "microsound_preset.json", "JSON (*.json)")
        if not fn: return
        p = self.get_params()
        try:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(p, f, indent=2)
            self.status.setText(f"Preset saved: {fn}")
        except Exception as e:
            self.status.setText(f"Preset save failed: {e}")

    def on_load_preset(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load preset", "", "JSON (*.json)")
        if not fn: return
        try:
            with open(fn, "r", encoding="utf-8") as f:
                p = json.load(f)

            # Ensure *all* UI settings change when a preset is loaded.
            # Many presets are intentionally partial; we merge them over factory defaults.
            base = dict(getattr(self, "_factory_defaults", self.get_params()))
            if isinstance(p, dict):
                base.update(p)
            self.apply_params(base)
            self.status.setText(f"Preset loaded: {fn}")
        except Exception as e:
            self.status.setText(f"Preset load failed: {e}")

    def apply_params(self, p):
        def set_combo(c, val):
            i = c.findText(str(val))
            if i >= 0: c.setCurrentIndex(i)

        set_combo(self.base_sr, p.get("base_sr", 48000))
        self.out_dur.setValue(float(p.get("out_dur_s", 8.0)))
        self.time_unfold.setValue(float(p.get("time_unfold", 25.0)))
        self.peak.setValue(float(p.get("peak", 0.98)))
        self.sat_drive.setValue(float(p.get("sat_drive", 1.0)))
        self.stereo_on.setChecked(bool(p.get("stereo_on", True)))
        self.stereo_width.setValue(float(p.get("stereo_width", 0.65)))

        set_combo(self.gen_mode, p.get("gen_mode", "Gaussian click"))
        self.micro_ms.setValue(float(p.get("micro_ms", 1.25)))
        self.seed.setValue(int(p.get("seed", 12345)))

        self.dust_density.setValue(float(p.get("dust_density", 0.02)))
        self.noise_tilt.setValue(float(p.get("noise_tilt", -3.0)))
        self.ring_hz.setValue(float(p.get("ring_hz", 4200.0)))
        self.ring_decay.setValue(float(p.get("ring_decay_ms", 12.0)))

        self.crackle_alpha.setValue(float(p.get("crackle_alpha", 1.4)))
        self.crackle_density.setValue(float(p.get("crackle_density", 180.0)))
        self.crackle_kernel.setValue(int(p.get("crackle_kernel", 64)))

        self.ss_threshold.setValue(float(p.get("ss_threshold", 0.9)))
        self.ss_build.setValue(float(p.get("ss_build", 0.06)))
        self.ss_decay.setValue(float(p.get("ss_decay", 0.75)))
        self.ss_noise.setValue(float(p.get("ss_noise", 0.08)))

        self.chaos_r.setValue(float(p.get("chaos_r", 3.92)))
        self.chaos_gate.setValue(float(p.get("chaos_gate", 0.35)))

        self.wav_base_hz.setValue(float(p.get("wav_base_hz", 2400.0)))
        self.wav_count.setValue(int(p.get("wav_count", 8)))
        self.wav_spread.setValue(float(p.get("wav_spread", 0.6)))

        set_combo(self.unfold_mode, p.get("unfold_mode", "Classic reinterpret"))
        self.partial_stretch.setValue(float(p.get("partial_stretch", 1.0)))
        self.partial_lock_on.setChecked(bool(p.get("partial_lock_on", False)))
        self.pl_top_n.setValue(int(p.get("pl_top_n", 24)))
        self.pl_neigh.setValue(int(p.get("pl_neigh", 4)))
        self.nl_warp_on.setChecked(bool(p.get("nl_warp_on", False)))
        self.nl_warp_power.setValue(float(p.get("nl_warp_power", 1.25)))
        self.cep_warp_on.setChecked(bool(p.get("cep_warp_on", False)))
        self.cep_factor.setValue(float(p.get("cep_factor", 1.2)))

        self.mb_b1.setValue(float(p.get("mb_b1", 2000.0)))
        self.mb_b2.setValue(float(p.get("mb_b2", 8000.0)))
        self.mb_b3.setValue(float(p.get("mb_b3", 20000.0)))
        self.mb_u1.setValue(float(p.get("mb_u1", 35.0)))
        self.mb_u2.setValue(float(p.get("mb_u2", 20.0)))
        self.mb_u3.setValue(float(p.get("mb_u3", 12.0)))
        self.mb_roll.setValue(float(p.get("mb_roll", 2000.0)))

        self.band_on.setChecked(bool(p.get("bandlimit_on", True)))
        self.band_out.setValue(float(p.get("bandlimit_out_hz", 18000.0)))
        self.band_roll.setValue(float(p.get("bandlimit_roll_hz", 2500.0)))

        set_combo(self.event_process, p.get("event_process", "Poisson"))
        self.grains_per_sec.setValue(float(p.get("grains_per_sec", 18.0)))
        self.max_grains.setValue(int(p.get("max_grains", 4000)))
        self.grain_amp_rand.setValue(float(p.get("grain_amp_rand", 0.35)))
        self.grain_offset_on.setChecked(bool(p.get("grain_offset_on", True)))
        self.grain_offset_max.setValue(float(p.get("grain_offset_max_ms", 60.0)))
        self.cluster_size.setValue(int(p.get("cluster_size", 6)))
        self.cluster_spread.setValue(float(p.get("cluster_spread_ms", 25.0)))
        self.hawkes_gain.setValue(float(p.get("hawkes_gain", 0.6)))
        self.hawkes_decay.setValue(float(p.get("hawkes_decay_s", 0.25)))

        self.bp_density.setText(str(p.get("bp_density", "0:18, 4:40, 8:14")))
        self.bp_unfold.setText(str(p.get("bp_unfold", "")))
        self.bp_cutoff.setText(str(p.get("bp_cutoff", "")))
        self.bp_stretch.setText(str(p.get("bp_stretch", "")))

        self.res_bank_on.setChecked(bool(p.get("res_bank_on", False)))
        self.res_modes.setValue(int(p.get("res_modes", 24)))
        self.res_fmin.setValue(float(p.get("res_fmin", 120.0)))
        self.res_fmax.setValue(float(p.get("res_fmax", 12000.0)))
        self.res_decay.setValue(float(p.get("res_decay_ms", 80.0)))

        self.wg_on.setChecked(bool(p.get("wg_on", False)))
        self.wg_lines.setValue(int(p.get("wg_lines", 8)))
        self.wg_max.setValue(float(p.get("wg_max_ms", 8.0)))
        self.wg_fb.setValue(float(p.get("wg_fb", 0.7)))

        self.ev_fb_on.setChecked(bool(p.get("event_feedback_on", False)))
        self.ev_fb_amt.setValue(float(p.get("event_feedback_amt", 0.35)))
        self.imprint_on.setChecked(bool(p.get("spectral_imprint_on", False)))
        self.imprint_amt.setValue(float(p.get("spectral_imprint_amt", 0.35)))
        self.imprint_smooth.setValue(float(p.get("spectral_imprint_smooth", 0.92)))

        self.er_on.setChecked(bool(p.get("er_cloud_on", True)))
        self.er_taps.setValue(int(p.get("er_taps", 320)))
        self.er_max.setValue(float(p.get("er_max_ms", 45.0)))
        self.space_ir_on.setChecked(bool(p.get("space_ir_on", False)))
        self.space_ir_max.setValue(int(p.get("space_ir_max_samps", 12000)))

        self.env_a.setValue(float(p.get("env_a", 20.0)))
        self.env_d.setValue(float(p.get("env_d", 250.0)))
        self.env_s.setValue(float(p.get("env_s", 0.65)))
        self.env_r.setValue(float(p.get("env_r", 1800.0)))
        self.env_curve.setValue(float(p.get("env_curve", 1.8)))

    def on_load_ir(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load IR / impulse (wav/flac/ogg)", "", "Audio (*.wav *.flac *.ogg)")
        if not fn: return
        try:
            a, sr = sf.read(fn, always_2d=False)
            a = a.astype(np.float64)
            if a.ndim > 1:
                a = a.mean(axis=1)
            a = normalize(a, 0.9)
            self.ir_audio = a
            self.status.setText(f"Loaded IR: {os.path.basename(fn)} (sr={sr}, samples={a.size})")
        except Exception as e:
            self.status.setText(f"Load IR failed: {e}")

    def on_load_img(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not fn: return
        try:
            qimg = QtWidgets.QImage(fn)
            if qimg.isNull():
                raise RuntimeError("Could not load image")
            qimg = qimg.convertToFormat(QtWidgets.QImage.Format.Format_Grayscale8)
            w = qimg.width(); h = qimg.height()
            ptr = qimg.bits(); ptr.setsize(h*w)
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h,w))
            self.img_gray = arr
            self.status.setText(f"Loaded image: {os.path.basename(fn)} ({w}x{h})")
        except Exception as e:
            self.status.setText(f"Load image failed: {e}")

    def on_render(self):
        self.btn_render.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.pbar.setValue(0)
        self.status.setText("Rendering…")

        params = self.get_params()
        params["_ir_audio"] = self.ir_audio
        params["_img_gray"] = self.img_gray

        self.thread = QtCore.QThread()
        self.worker = RenderWorker(params)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_progress(self, pct, msg):
        self.pbar.setValue(int(pct))
        self.status.setText(msg)

    def on_failed(self, err):
        self.btn_render.setEnabled(True)
        self.status.setText(f"Error: {err}")
        self.pbar.setValue(0)

    def on_finished(self, audio, meta):
        self.audio = audio
        self.meta = meta
        self.btn_render.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.pbar.setValue(100)
        self.status.setText(f"Done. Output SR={meta['out_sr']} | channels={audio.shape[1]} | samples={audio.shape[0]}")
        self.update_views()

    def update_views(self):
        if self.audio is None:
            return
        y = self.audio
        sr = int(self.meta["out_sr"])
        n = y.shape[0]

        max_points = 120_000
        step = max(1, n//max_points)
        t = (np.arange(0, n, step) / sr).astype(np.float64)
        L = y[::step, 0]
        R = y[::step, 1]
        self.wave_curveL.setData(t, L)
        self.wave_curveR.setData(t, R)

        take = min(n, int(sr*4.0))
        seg = y[:take, :].mean(axis=1)
        seg = seg * hann(take)
        X = np.fft.rfft(seg)
        f = rfft_freqs(take, sr)
        self.spec_curve.setData(f, db(X))

        win = 2048 if sr < 96000 else 4096
        hop = 256 if sr < 96000 else 512
        S = stft_mag_db(y[:, :].mean(axis=1), sr, win=win, hop=hop)
        self.simg.setImage(S, autoLevels=True)

        micro = self.meta.get("micro_last")
        grain = self.meta.get("grain_last")
        if micro is not None and micro.size:
            mm = micro[:min(micro.size, 24000)]
            self.micro_curve.setData(np.arange(mm.size), mm)
        if grain is not None and grain.size:
            gg = grain[:min(grain.size, 24000)]
            self.grain_curve.setData(np.arange(gg.size), gg)

    def on_export(self):
        if self.audio is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export WAV", "microsound_v2.wav", "WAV (*.wav)")
        if not fn:
            return
        try:
            sf.write(fn, self.audio.astype(np.float32), int(self.meta["out_sr"]))
            self.status.setText(f"Exported: {fn}")
        except Exception as e:
            self.status.setText(f"Export failed: {e}")

    def on_batch(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose batch output folder")
        if not folder:
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Batch render")
        lay = QtWidgets.QVBoxLayout(dlg)

        info = QtWidgets.QLabel("Renders variations over seeds/unfold/stretch.\nComma-separated lists.")
        info.setWordWrap(True)
        lay.addWidget(info)

        seeds = QtWidgets.QLineEdit("1001,1002,1003,1004,1005")
        unfolds = QtWidgets.QLineEdit("15,25,40")
        stretches = QtWidgets.QLineEdit("0.9,1.0,1.2")
        lay.addWidget(QtWidgets.QLabel("Seeds")); lay.addWidget(seeds)
        lay.addWidget(QtWidgets.QLabel("Base unfold")); lay.addWidget(unfolds)
        lay.addWidget(QtWidgets.QLabel("Stretch factor")); lay.addWidget(stretches)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                          QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        lay.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        def parse_list(s, cast=float):
            out = []
            for p in s.split(","):
                p = p.strip()
                if not p: continue
                try:
                    out.append(cast(p))
                except:
                    pass
            return out

        seed_list = parse_list(seeds.text(), int)
        unfold_list = parse_list(unfolds.text(), float)
        stretch_list = parse_list(stretches.text(), float)

        base_params = self.get_params()
        base_params["_ir_audio"] = self.ir_audio
        base_params["_img_gray"] = self.img_gray

        self.status.setText("Batch rendering…")
        QtWidgets.QApplication.processEvents()

        total = max(1, len(seed_list)*len(unfold_list)*len(stretch_list))
        done = 0

        for sd in seed_list:
            for u in unfold_list:
                for st in stretch_list:
                    p = dict(base_params)
                    p["seed"] = int(sd)
                    p["time_unfold"] = float(u)
                    p["partial_stretch"] = float(st)
                    audio, meta = render(p, progress=None)
                    out_sr = int(meta["out_sr"])
                    name = f"ms_seed{sd}_unf{u:g}_st{st:g}_{out_sr}Hz.wav".replace(".", "p")
                    path = os.path.join(folder, name)
                    sf.write(path, audio.astype(np.float32), out_sr)
                    done += 1
                    self.pbar.setValue(int(100*done/total))
                    self.status.setText(f"Batch: {done}/{total} → {name}")
                    QtWidgets.QApplication.processEvents()

        self.status.setText(f"Batch complete: {folder}")
        self.pbar.setValue(100)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MicrosoundV2()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
