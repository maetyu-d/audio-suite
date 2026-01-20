import sys
import os
import numpy as np
import sounddevice as sd
import soundfile as sf

from PySide6 import QtCore, QtGui, QtWidgets

"""
Virtual Tape Deck – Qt (Gentle, 1–3 Heads, User Offsets, Base Speed, No Recording)

New in this version:
- Player can set a BASE TAPE SPEED from -100% to +100% (reverse to forward).
- Manual tape scrubbing is layered on top of the base speed.
- Up to 3 tape heads, with **user-adjustable offsets** in samples:
    - Keys Z / X : move Head 1 (left) earlier / later
    - Keys C / V : move Head 2 (center) earlier / later
    - Keys B / N : move Head 3 (right) earlier / later
    - Key R     : reset all head offsets to default [-2000, 0, +2000] samples

Other controls:
- LMB drag: scrub (adds/subtracts on top of base speed).
- RMB drag: jump to position.
- 1 / 2 / 3: number of active tape heads.
- Up arrow / '+': increase base speed by 10%.
- Down arrow / '-': decrease base speed by 10%.
- '0': reset base speed to 0% (stopped).
- Esc / Q: quit.
"""

# -----------------------------
#  CONFIG TWEAKS (GENTLER)
# -----------------------------

DEFAULT_HEAD_OFFSETS = [-2000.0, 0.0, 2000.0]  # samples (left, center, right)
HEAD_GAIN = 0.8

TAPE_FRICTION_PER_FRAME = 0.93
MOUSE_SPEED_TO_TAPE_SPEED = 70.0   # gesture -> samples/sec scaling

# Wow/flutter (lower depths)
WOW_FREQ = 0.5
WOW_DEPTH = 0.006                  # was 0.015
FLUTTER_FREQ = 6.0
FLUTTER_DEPTH = 0.002              # was 0.004

# Tape stretch jitter (reduced)
STRETCH_DEPTH = 0.007              # was 0.02
STRETCH_SPEED_SCALE = 25000.0

# Dropouts (rarer and milder)
DROPOUT_DEPTH = 0.35               # was 0.6
DROPOUT_PROB = 0.008               # was 0.03
DROPOUT_MIN_BLOCKS = 3
DROPOUT_MAX_BLOCKS = 10

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 380
FPS = 60.0

REEL_ANGLE_SPEED_SCALE = 0.00035

BACKGROUND_COLOR = QtGui.QColor(10, 10, 10)
TAPE_COLOR = QtGui.QColor(130, 130, 130)
HEAD_COLOR = QtGui.QColor(220, 100, 80)
REEL_COLOR = QtGui.QColor(180, 180, 180)
GUIDE_COLOR = QtGui.QColor(90, 90, 90)
TEXT_COLOR = QtGui.QColor(220, 220, 220)
MARKER_COLOR = QtGui.QColor(250, 250, 100)

# Limit maximum absolute total tape speed (samples/sec)
MAX_TAPE_SPEED = 40_000  # approx 1x–<2x at 44.1k

# Offset tweak step (samples) when nudging heads
HEAD_OFFSET_STEP = 200.0


# -----------------------------
#  GLOBAL AUDIO STATE
# -----------------------------

audio_data = None       # numpy array, mono, float32
sample_rate = None
tape_pos = 0.0          # in samples

# Base + gesture speed
base_speed_factor = 0.0   # -1.0 .. +1.0, represents -100% .. +100%
gesture_speed = 0.0       # additional samples/sec from scrubbing

# For info only; updated in audio thread & GUI
current_total_speed = 0.0  # samples/sec

playback_time = 0.0     # seconds for LFOs

dropout_active = False
dropout_blocks_left = 0

# Tape head config (1–3 heads)
current_head_count = 3  # default: triple head
head_offsets_samples = DEFAULT_HEAD_OFFSETS.copy()  # [left, center, right]


# -----------------------------
#  AUDIO HELPERS
# -----------------------------

def get_head_offsets():
    """
    Return a list of sample offsets based on current_head_count and the
    user-set head_offsets_samples [left, center, right].
    """
    left, center, right = head_offsets_samples
    if current_head_count <= 1:
        return [center]
    elif current_head_count == 2:
        return [left, right]
    else:
        return [left, center, right]


def load_wav(path):
    global audio_data, sample_rate

    data, sr = sf.read(path, always_2d=True)
    sample_rate = sr

    # mono
    if data.shape[1] > 1:
        data = np.mean(data, axis=1)
    else:
        data = data[:, 0]

    max_val = np.max(np.abs(data)) if data.size > 0 else 1.0
    if max_val > 1.0:
        data = data / max_val

    audio_data = data.astype(np.float32)
    print(f"Loaded {path}: {len(audio_data)} samples @ {sample_rate} Hz")


def sample_from_tape_array(positions):
    if audio_data is None or len(audio_data) == 0:
        return np.zeros_like(positions, dtype=np.float32)

    n = len(audio_data)
    pw = np.mod(positions, n)

    i0 = np.floor(pw).astype(np.int64)
    i1 = (i0 + 1) % n
    frac = pw - i0

    s0 = audio_data[i0]
    s1 = audio_data[i1]
    return ((1.0 - frac) * s0 + frac * s1).astype(np.float32)


def audio_callback(outdata, frames, time_info, status):
    global tape_pos, playback_time
    global dropout_active, dropout_blocks_left
    global current_total_speed, base_speed_factor, gesture_speed

    if status:
        # print(status, file=sys.stderr)
        pass

    if audio_data is None or sample_rate is None:
        outdata[:] = 0.0
        return

    # Compute total speed (samples/sec) from base + gesture
    base_samples_per_sec = base_speed_factor * sample_rate
    total_speed = base_samples_per_sec + gesture_speed

    # Clamp total speed
    if abs(total_speed) > MAX_TAPE_SPEED:
        total_speed = np.sign(total_speed) * MAX_TAPE_SPEED

    # Share for GUI
    current_total_speed = float(total_speed)

    # Time vector for LFOs
    t = playback_time + np.arange(frames, dtype=np.float32) / sample_rate
    base_inc = total_speed / sample_rate  # samples per audio sample

    # Wow & flutter on speed (gentler)
    wow = WOW_DEPTH * np.sin(2.0 * np.pi * WOW_FREQ * t)
    flutter = FLUTTER_DEPTH * np.sin(2.0 * np.pi * FLUTTER_FREQ * t)
    speed_factor = 1.0 + wow + flutter

    # Stretch jitter when moving fast (weaker)
    if abs(total_speed) > 1.0 and STRETCH_DEPTH > 0.0:
        jitter_strength = STRETCH_DEPTH * np.tanh(abs(total_speed) / STRETCH_SPEED_SCALE)
        stretch_noise = jitter_strength * np.random.randn(frames).astype(np.float32)
        speed_factor *= (1.0 + stretch_noise)

    increments = base_inc * speed_factor
    if frames > 0:
        offsets = np.concatenate(([0.0], np.cumsum(increments[:-1])))
    else:
        offsets = np.array([], dtype=np.float32)

    positions = tape_pos + offsets

    buf = np.zeros(frames, dtype=np.float32)
    head_offsets = get_head_offsets()
    for off in head_offsets:
        buf += sample_from_tape_array(positions + off)

    if len(head_offsets) > 0:
        buf *= (HEAD_GAIN / float(len(head_offsets)))

    # Dropouts (block-based, rarer & milder)
    if dropout_active:
        env = 1.0 - DROPOUT_DEPTH
        dropout_blocks_left -= 1
        if dropout_blocks_left <= 0:
            dropout_active = False
    else:
        env = 1.0
        if np.random.rand() < DROPOUT_PROB:
            dropout_active = True
            dropout_blocks_left = np.random.randint(DROPOUT_MIN_BLOCKS,
                                                    DROPOUT_MAX_BLOCKS + 1)
            env = 1.0 - DROPOUT_DEPTH
    buf *= env

    # Update shared state
    if frames > 0:
        tape_pos = positions[-1] + increments[-1]
        playback_time = t[-1] + 1.0 / sample_rate

    # Fill stereo output
    outdata[:, 0] = buf
    if outdata.shape[1] > 1:
        outdata[:, 1] = buf


# -----------------------------
#  QT WIDGET
# -----------------------------

class TapeDeckWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Virtual Tape Deck – Qt (Gentle, User Offsets, Base Speed, No Recording)")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.reel_angle = 0.0
        self.dragging_left = False
        self.dragging_right = False
        self.last_mouse_x = 0
        self.last_time = QtCore.QElapsedTimer()
        self.last_time.start()

        # Timer for animation / GUI updates
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(int(1000.0 / FPS))

    def on_timer(self):
        global gesture_speed, base_speed_factor, sample_rate, current_total_speed

        # dt used for reel rotation & friction
        dt_ms = self.last_time.restart()
        dt = max(dt_ms / 1000.0, 1e-6)

        # Apply friction to gesture component if not dragging
        if not self.dragging_left:
            gesture_speed *= TAPE_FRICTION_PER_FRAME

        # Recompute approximate total speed for GUI reel rotation
        if sample_rate is not None:
            base_samples_per_sec = base_speed_factor * sample_rate
        else:
            base_samples_per_sec = 0.0
        total_speed = base_samples_per_sec + gesture_speed
        if abs(total_speed) > MAX_TAPE_SPEED:
            total_speed = np.sign(total_speed) * MAX_TAPE_SPEED
        current_total_speed = float(total_speed)

        # Update reel angle based on total speed
        self.reel_angle += total_speed * REEL_ANGLE_SPEED_SCALE * dt

        self.update()

    # --- Mouse events ---

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self.dragging_left = True
            self.last_mouse_x = event.position().x()
        elif event.button() == QtCore.Qt.RightButton:
            self.dragging_right = True
            self.last_mouse_x = event.position().x()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self.dragging_left = False
        elif event.button() == QtCore.Qt.RightButton:
            self.dragging_right = False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        global gesture_speed, tape_pos, audio_data

        x = event.position().x()
        if self.dragging_left:
            dx = x - self.last_mouse_x
            self.last_mouse_x = x
            # Convert mouse drag to gesture speed (samples/sec)
            gesture_speed = dx * MOUSE_SPEED_TO_TAPE_SPEED * FPS

        if self.dragging_right and audio_data is not None:
            w = float(self.width())
            norm = max(0.0, min(1.0, x / w))
            tape_pos = norm * (len(audio_data) - 1)

    # --- Keyboard ---

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        global current_head_count, base_speed_factor, head_offsets_samples

        key = event.key()
        if key in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            QtWidgets.QApplication.quit()

        # Head count selection
        elif key == QtCore.Qt.Key_1:
            current_head_count = 1
        elif key == QtCore.Qt.Key_2:
            current_head_count = 2
        elif key == QtCore.Qt.Key_3:
            current_head_count = 3

        # Base speed controls
        elif key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            base_speed_factor = min(1.0, base_speed_factor + 0.1)
        elif key in (QtCore.Qt.Key_Down, QtCore.Qt.Key_Minus):
            base_speed_factor = max(-1.0, base_speed_factor - 0.1)
        elif key == QtCore.Qt.Key_0:
            base_speed_factor = 0.0

        # Head offset nudging (Z/X: head 1, C/V: head 2, B/N: head 3)
        elif key == QtCore.Qt.Key_Z:
            head_offsets_samples[0] -= HEAD_OFFSET_STEP
        elif key == QtCore.Qt.Key_X:
            head_offsets_samples[0] += HEAD_OFFSET_STEP
        elif key == QtCore.Qt.Key_C:
            head_offsets_samples[1] -= HEAD_OFFSET_STEP
        elif key == QtCore.Qt.Key_V:
            head_offsets_samples[1] += HEAD_OFFSET_STEP
        elif key == QtCore.Qt.Key_B:
            head_offsets_samples[2] -= HEAD_OFFSET_STEP
        elif key == QtCore.Qt.Key_N:
            head_offsets_samples[2] += HEAD_OFFSET_STEP
        elif key == QtCore.Qt.Key_R:
            head_offsets_samples = DEFAULT_HEAD_OFFSETS.copy()

        else:
            super().keyPressEvent(event)

    # --- Painting ---

    def paintEvent(self, event: QtGui.QPaintEvent):
        global tape_pos, audio_data, current_head_count, current_total_speed
        global base_speed_factor, sample_rate, head_offsets_samples

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # Background
        painter.fillRect(self.rect(), BACKGROUND_COLOR)

        tape_y = self.height() // 2 + 20
        left_center = QtCore.QPoint(self.width() // 4, tape_y - 40)
        right_center = QtCore.QPoint(3 * self.width() // 4, tape_y - 40)
        reel_radius = 45
        head_x = self.width() // 2

        # Reels
        self.draw_reel(painter, left_center, reel_radius, self.reel_angle)
        self.draw_reel(painter, right_center, reel_radius, -self.reel_angle * 1.1)

        # Tape path
        pen = QtGui.QPen(TAPE_COLOR, 6)
        painter.setPen(pen)
        painter.drawLine(left_center.x() + reel_radius, tape_y,
                         head_x - 20, tape_y)
        painter.drawLine(head_x + 20, tape_y,
                         right_center.x() - reel_radius, tape_y)

        # Guides
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(GUIDE_COLOR)
        painter.drawEllipse(QtCore.QPoint(head_x - 26, tape_y - 12), 6, 6)
        painter.drawEllipse(QtCore.QPoint(head_x + 26, tape_y - 12), 6, 6)

        # Main head block
        painter.setBrush(HEAD_COLOR)
        head_rect = QtCore.QRect(head_x - 12, tape_y - 23, 24, 46)
        painter.drawRoundedRect(head_rect, 6, 6)

        # Additional head indicators (small blocks above the tape)
        painter.setBrush(HEAD_COLOR.lighter(130))
        offsets = get_head_offsets()
        if len(offsets) > 1:
            # Use relative ordering to place side-heads left/right visually
            side_spacing_px = 18
            # Map actual offsets to left/center/right labels for drawing
            # (purely indicative; not spatially accurate)
            sorted_offsets = sorted(offsets)
            for off in sorted_offsets:
                if abs(off) < 1e-3:
                    # already represented by main head block
                    continue
                sign = -1 if off < 0 else 1
                rect = QtCore.QRect(
                    head_x + sign * side_spacing_px - 6,
                    tape_y - 37,
                    12,
                    18,
                )
                painter.drawRoundedRect(rect, 3, 3)

        # Position marker
        if audio_data is not None and len(audio_data) > 0:
            norm = (tape_pos % len(audio_data)) / float(len(audio_data))
        else:
            norm = 0.0
        marker_x = int(norm * self.width())
        pen = QtGui.QPen(MARKER_COLOR, 2)
        painter.setPen(pen)
        painter.drawLine(marker_x, 10, marker_x, 80)

        # Text overlay
        painter.setPen(TEXT_COLOR)
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        if sample_rate is not None and sample_rate > 0:
            base_pct = base_speed_factor * 100.0
            # Convert offsets to ms for display
            left_ms  = head_offsets_samples[0] / sample_rate * 1000.0
            center_ms= head_offsets_samples[1] / sample_rate * 1000.0
            right_ms = head_offsets_samples[2] / sample_rate * 1000.0
        else:
            base_pct = 0.0
            left_ms = center_ms = right_ms = 0.0

        lines = [
            "LMB drag: scrub tape (adds on top of base speed).",
            "RMB drag: jump to position.",
            "Keys 1/2/3: number of tape heads.",
            "Arrow Up/Down or +/-: base speed -100%..+100%, 0 = stop.",
            "Z/X: left head offset -, +   C/V: center head -, +   B/N: right head -, +",
            f"Heads: {current_head_count}",
            f"Base speed:      {base_pct:6.1f} %",
            f"Total tape speed: {current_total_speed:9.1f} samples/s",
            f"Tape pos:        {tape_pos:10.1f}" + (f" / {len(audio_data)}" if audio_data is not None else ""),
            f"Head offsets (ms): L={left_ms:7.2f}  C={center_ms:7.2f}  R={right_ms:7.2f}",
        ]
        x = 10
        y = 18
        for line in lines:
            painter.drawText(x, y, line)
            y += 16

        painter.end()

    def draw_reel(self, painter: QtGui.QPainter, center: QtCore.QPoint, radius: int, angle: float):
        # Outer ring
        pen = QtGui.QPen(REEL_COLOR, 3)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)

        # Hub
        painter.setBrush(REEL_COLOR)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(center, radius // 6, radius // 6)

        # Spokes
        painter.setPen(QtGui.QPen(REEL_COLOR, 2))
        num_spokes = 4
        for i in range(num_spokes):
            a = angle + (2.0 * np.pi * i / num_spokes)
            x = center.x() + int(np.cos(a) * radius * 0.75)
            y = center.y() + int(np.sin(a) * radius * 0.75)
            painter.drawLine(center, QtCore.QPoint(x, y))


# -----------------------------
#  MAIN
# -----------------------------

def choose_wav_qt():
    """Use a Qt file dialog to choose a WAV file."""
    dialog = QtWidgets.QFileDialog(
        None,
        "Select a WAV file",
        os.getcwd(),
        "WAV files (*.wav);;All files (*)"
    )
    dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None
    files = dialog.selectedFiles()
    if not files:
        return None
    return files[0]


def main():
    global audio_data, sample_rate

    app = QtWidgets.QApplication(sys.argv)

    wav_path = None
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        wav_path = sys.argv[1]
    else:
        wav_path = choose_wav_qt()

    if not wav_path:
        print("No WAV file selected. Exiting.")
        return

    load_wav(wav_path)
    if audio_data is None or len(audio_data) == 0:
        print("Failed to load audio. Exiting.")
        return

    widget = TapeDeckWidget()
    widget.show()

    # Slightly larger blocksize for more stable playback, still interactive.
    with sd.OutputStream(
        samplerate=sample_rate,
        channels=2,
        callback=audio_callback,
        blocksize=1024,
        dtype="float32",
    ):
        app.exec()

    print("Bye.")


if __name__ == "__main__":
    main()
