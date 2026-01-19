import os
import sys
import traceback
import importlib.util
import inspect
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple

import numpy as np
import soundfile as sf

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QMessageBox, QFileDialog, QComboBox,
    QSplitter, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QSpinBox, QDoubleSpinBox, QPlainTextEdit, QCheckBox, QGroupBox
)

import pyqtgraph as pg

# ----------------------------
# Audio helpers
# ----------------------------
def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=1)

def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    if len(x) == 0:
        return x.astype(np.float32, copy=False)
    duration = len(x) / sr_in
    n_out = max(1, int(round(duration * sr_out)))
    t_in = np.linspace(0.0, duration, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)

def _fit_to_duration(x: np.ndarray, sr: int, duration: float) -> np.ndarray:
    n = max(0, int(round(duration * sr)))
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    x = x.astype(np.float32, copy=False)
    if len(x) == n:
        return x
    if len(x) < n:
        out = np.zeros((n,), dtype=np.float32)
        out[:len(x)] = x
        return out
    return x[:n]

def _normalize_peak(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    if len(x) == 0:
        return x.astype(np.float32, copy=False)
    m = float(np.max(np.abs(x)))
    if m <= 1e-12:
        return x.astype(np.float32, copy=False)
    g = min(1.0, peak / m)
    return (x * g).astype(np.float32)

def load_wav_any(path: str, sr_target: int) -> np.ndarray:
    data, sr_in = sf.read(path, always_2d=False)
    data = _to_mono(np.asarray(data, dtype=np.float32))
    return _resample_linear(data, sr_in, sr_target)

# ----------------------------
# Generator module loader (audio + optional events)
# ----------------------------
class LoadedModule:
    def __init__(self, path: str):
        self.path = path
        self.mod = self._load_module(path)
        self.generate = getattr(self.mod, "generate", None)
        self.event = getattr(self.mod, "event", None)

        if self.generate is not None:
            sig = inspect.signature(self.generate)
            if len(sig.parameters) not in (2, 3):
                raise RuntimeError("generate() must take (sr, duration) or (sr, duration, context)")

        if self.event is not None:
            sig = inspect.signature(self.event)
            if len(sig.parameters) != 1:
                raise RuntimeError("event() must take (context)")

        if self.generate is None and self.event is None:
            raise RuntimeError("Python cell scripts must define generate(...) and/or event(context).")

    @staticmethod
    def _load_module(path: str):
        spec = importlib.util.spec_from_file_location(f"cell_module_{abs(hash(path))}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load script: {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod

# simple cache so we don't re-import repeatedly
_MODULE_CACHE: Dict[str, LoadedModule] = {}

def load_py_module(path: str) -> LoadedModule:
    m = _MODULE_CACHE.get(path)
    if m is None:
        m = LoadedModule(path)
        _MODULE_CACHE[path] = m
    return m

# ----------------------------
# Division generators
# ----------------------------
def divisions_uniform(total: float, n: int) -> List[float]:
    n = max(1, int(n))
    return [total / n] * n

DEFAULT_DIVISION_SNIPPET = """def divisions(total):
    return [total/16.0]*16
"""

def parse_float_list(text: str) -> List[float]:
    pts: List[float] = []
    for part in text.replace(";", ",").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            v = float(s)
            if np.isfinite(v):
                pts.append(v)
        except Exception:
            pass
    pts.sort()
    return pts

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = min(win, len(x))
    if win <= 1:
        return x
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same").astype(np.float32)

def rms_envelope(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) == 0:
        return x.astype(np.float32, copy=False)
    xx = (x.astype(np.float32, copy=False) ** 2)
    sm = moving_average(xx, max(1, win))
    return np.sqrt(np.maximum(sm, 0.0)).astype(np.float32)

# ----------------------------
# Data model
# ----------------------------
@dataclass
class CellSource:
    kind: str  # "empty" | "wav" | "py"
    path: str

@dataclass
class Track:
    name: str = "Track"
    gain_db: float = 0.0
    mode: str = "tempo_bpm"  # "tempo_bpm" | "tempo_spm" | "duration"
    bpm: float = 120.0
    seconds_per_measure: float = 2.0
    beats_per_measure: int = 4
    measures: int = 4
    duration_seconds: float = 8.0

    start_offset_seconds: float = 0.0
    loop_to_master: bool = False

    sync_points_text: str = ""  # master seconds

    mod_source_index: int = -1
    mod_amount: float = 0.0
    mod_smoothing_ms: float = 50.0

    division_mode: str = "uniform"
    uniform_n: int = 16
    python_code: str = DEFAULT_DIVISION_SNIPPET

    cells: List[CellSource] = field(default_factory=list)

    def total_duration(self) -> float:
        if self.mode == "duration":
            return max(0.0, float(self.duration_seconds))
        if self.mode == "tempo_spm":
            return float(self.measures) * max(1e-6, float(self.seconds_per_measure))
        bpm = max(1e-6, float(self.bpm))
        beats = max(1, int(self.beats_per_measure))
        return float(self.measures) * (60.0 / bpm) * beats

    def build_divisions(self) -> List[float]:
        total = self.total_duration()
        if total <= 0:
            return []
        if self.division_mode == "python":
            glb = {"__builtins__": {"range": range, "len": len, "sum": sum, "min": min, "max": max, "abs": abs, "float": float, "int": int}}
            loc: Dict[str, Any] = {}
            exec(self.python_code, glb, loc)
            if "divisions" not in loc:
                raise RuntimeError("Python divisions code must define: divisions(total)")
            divs = loc["divisions"](total)
            out = [float(x) for x in divs]
            s = sum(out)
            if s <= 0:
                return []
            scale = total / s
            return [x * scale for x in out]
        return divisions_uniform(total, self.uniform_n)

    def ensure_cells(self, n: int):
        n = max(0, int(n))
        if len(self.cells) < n:
            self.cells.extend([CellSource("empty", "") for _ in range(n - len(self.cells))])
        elif len(self.cells) > n:
            self.cells = self.cells[:n]

@dataclass
class MasterClock:
    mode: str = "auto"
    fixed_seconds: float = 16.0

    def duration(self, tracks: List[Track]) -> float:
        if self.mode == "fixed_seconds":
            return max(0.01, float(self.fixed_seconds))
        m = 0.0
        for t in tracks:
            m = max(m, max(0.0, float(t.start_offset_seconds)) + max(0.0, t.total_duration()))
        return max(0.01, m)

# ----------------------------
# GUI
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grid Audio — Track restart events")
        self.resize(1500, 900)

        self.sr = 48000
        self.tracks: List[Track] = [Track(name="Track 1")]
        self.current_track_index = 0
        self.master = MasterClock()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)

        master_box = QGroupBox("Master clock")
        master_layout = QFormLayout(master_box)
        self.master_mode = QComboBox()
        self.master_mode.addItems(["Auto (max of tracks)", "Fixed seconds"])
        self.master_mode.currentIndexChanged.connect(self.on_master_changed)
        master_layout.addRow("Mode:", self.master_mode)
        self.master_seconds = QDoubleSpinBox()
        self.master_seconds.setRange(0.01, 36000.0)
        self.master_seconds.setValue(self.master.fixed_seconds)
        self.master_seconds.valueChanged.connect(self.on_master_changed)
        master_layout.addRow("Fixed seconds:", self.master_seconds)
        left_layout.addWidget(master_box)

        self.track_list = QListWidget()
        self.track_list.currentRowChanged.connect(self.on_select_track)
        left_layout.addWidget(QLabel("Tracks"))
        left_layout.addWidget(self.track_list)

        btn_row = QHBoxLayout()
        self.btn_add_track = QPushButton("Add Track")
        self.btn_del_track = QPushButton("Remove Track")
        self.btn_add_track.clicked.connect(self.add_track)
        self.btn_del_track.clicked.connect(self.remove_track)
        btn_row.addWidget(self.btn_add_track)
        btn_row.addWidget(self.btn_del_track)
        left_layout.addLayout(btn_row)

        # Per-track gain (left panel)
        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Track gain (dB):"))
        self.track_gain = QDoubleSpinBox()
        self.track_gain.setRange(-60.0, 12.0)
        self.track_gain.setDecimals(1)
        self.track_gain.setSingleStep(0.5)
        self.track_gain.valueChanged.connect(self.on_track_gain_changed)
        gain_row.addWidget(self.track_gain)
        left_layout.addLayout(gain_row)

        sr_row = QHBoxLayout()
        sr_row.addWidget(QLabel("Sample Rate (Hz):"))
        self.sr_box = QSpinBox()
        self.sr_box.setRange(8000, 192000)
        self.sr_box.setValue(self.sr)
        self.sr_box.valueChanged.connect(lambda v: setattr(self, "sr", int(v)))
        sr_row.addWidget(self.sr_box)
        left_layout.addLayout(sr_row)

        self.normalize_check = QCheckBox("Normalize mixdown peak to 0.98")
        self.normalize_check.setChecked(True)
        left_layout.addWidget(self.normalize_check)

        left_layout.addWidget(QLabel("Tip: .py cells can define event(context) to restart other tracks."))

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        form = QFormLayout()

        self.track_name = QLineEdit()
        self.track_name.editingFinished.connect(self.on_track_name_changed)
        form.addRow("Track name:", self.track_name)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Tempo (BPM)", "Tempo (sec/measure)", "Overall duration (sec)"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        form.addRow("Track clock mode:", self.mode_combo)

        self.bpm = QDoubleSpinBox(); self.bpm.setRange(1,1000); self.bpm.valueChanged.connect(self.on_timing_changed)
        self.beats = QSpinBox(); self.beats.setRange(1,64); self.beats.valueChanged.connect(self.on_timing_changed)
        self.spm = QDoubleSpinBox(); self.spm.setRange(0.01,3600); self.spm.valueChanged.connect(self.on_timing_changed)
        self.measures = QSpinBox(); self.measures.setRange(1,9999); self.measures.valueChanged.connect(self.on_timing_changed)
        self.duration = QDoubleSpinBox(); self.duration.setRange(0.01,36000); self.duration.valueChanged.connect(self.on_timing_changed)
        form.addRow("BPM:", self.bpm)
        form.addRow("Beats/measure:", self.beats)
        form.addRow("Seconds/measure:", self.spm)
        form.addRow("Measures:", self.measures)
        form.addRow("Duration (sec):", self.duration)

        self.offset = QDoubleSpinBox(); self.offset.setRange(-36000,36000); self.offset.setDecimals(3); self.offset.valueChanged.connect(self.on_multiclock_changed)
        self.loop = QCheckBox("Loop track pattern to fill master"); self.loop.stateChanged.connect(self.on_multiclock_changed)
        form.addRow("Start offset (sec):", self.offset)
        form.addRow("Loop to master:", self.loop)

        self.sync_points = QLineEdit(); self.sync_points.editingFinished.connect(self.on_sync_changed)
        form.addRow("Sync points (sec):", self.sync_points)

        self.mod_source = QComboBox(); self.mod_source.currentIndexChanged.connect(self.on_mod_changed)
        self.mod_amount = QDoubleSpinBox(); self.mod_amount.setRange(0,4); self.mod_amount.setDecimals(3); self.mod_amount.setSingleStep(0.05); self.mod_amount.valueChanged.connect(self.on_mod_changed)
        self.mod_smooth = QDoubleSpinBox(); self.mod_smooth.setRange(0,2000); self.mod_smooth.setDecimals(1); self.mod_smooth.setSingleStep(5); self.mod_smooth.valueChanged.connect(self.on_mod_changed)
        form.addRow("Clock mod source:", self.mod_source)
        form.addRow("Mod amount:", self.mod_amount)
        form.addRow("Mod smoothing (ms):", self.mod_smooth)

        self.div_mode = QComboBox(); self.div_mode.addItems(["Uniform","Python code"]); self.div_mode.currentIndexChanged.connect(self.on_div_mode_changed)
        self.uniform_n = QSpinBox(); self.uniform_n.setRange(1,4096); self.uniform_n.valueChanged.connect(self.on_div_params_changed)
        form.addRow("Division mode:", self.div_mode)
        form.addRow("Uniform divisions:", self.uniform_n)

        right_layout.addLayout(form)

        self.python_code = QPlainTextEdit(); self.python_code.setPlainText(DEFAULT_DIVISION_SNIPPET); self.python_code.textChanged.connect(self.on_python_code_changed)
        right_layout.addWidget(QLabel("Python division code (only if Division mode = Python code):"))
        right_layout.addWidget(self.python_code, stretch=1)

        right_layout.addWidget(QLabel("Grid (click a cell to assign WAV or PY; right-click to clear):"))
        self.grid = QTableWidget(1,1)
        self.grid.cellClicked.connect(self.on_cell_clicked)
        self.grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self.on_grid_context_menu)
        right_layout.addWidget(self.grid, stretch=2)

        btns = QHBoxLayout()
        self.btn_render = QPushButton("Render / Preview"); self.btn_render.clicked.connect(self.render_preview)
        self.btn_export = QPushButton("Export WAV"); self.btn_export.clicked.connect(self.export_wav)
        btns.addWidget(self.btn_render); btns.addWidget(self.btn_export)
        right_layout.addLayout(btns)

        self.plot = pg.PlotWidget()
        self.wave = self.plot.plot([], [])
        right_layout.addWidget(self.plot, stretch=2)

        splitter.addWidget(right)
        splitter.setSizes([360, 1140])
        self.setCentralWidget(splitter)

        file_menu = self.menuBar().addMenu("File")
        act_new = QAction("New", self); act_new.triggered.connect(self.new_project)
        file_menu.addAction(act_new)

        self.refresh_track_list()
        self.track_list.setCurrentRow(0)
        self.refresh_ui_from_track()
        self.rebuild_grid()
        self.apply_master_visibility()

    def err(self, title, e):
        QMessageBox.critical(self, title, f"{e}\n\n{traceback.format_exc()}")

    def apply_master_visibility(self):
        self.master_seconds.setEnabled(self.master_mode.currentIndex()==1)

    def on_master_changed(self):
        self.master.mode = "fixed_seconds" if self.master_mode.currentIndex()==1 else "auto"
        self.master.fixed_seconds = float(self.master_seconds.value())
        self.apply_master_visibility()

    def refresh_track_list(self):
        self.track_list.blockSignals(True)
        self.track_list.clear()
        for t in self.tracks:
            self.track_list.addItem(QListWidgetItem(t.name))
        self.track_list.blockSignals(False)

    def refresh_mod_source_combo(self):
        self.mod_source.blockSignals(True)
        self.mod_source.clear()
        self.mod_source.addItem("None")
        for t in self.tracks:
            self.mod_source.addItem(t.name)
        self.mod_source.blockSignals(False)

    def on_select_track(self, idx):
        if idx<0 or idx>=len(self.tracks): return
        self.current_track_index = idx
        self.refresh_ui_from_track()
        self.rebuild_grid()

    def current_track(self): return self.tracks[self.current_track_index]

    def add_track(self):
        self.tracks.append(Track(name=f"Track {len(self.tracks)+1}"))
        self.refresh_track_list(); self.refresh_mod_source_combo()
        self.track_list.setCurrentRow(len(self.tracks)-1)

    def remove_track(self):
        if len(self.tracks)<=1: return
        self.tracks.pop(self.current_track_index)
        for t in self.tracks:
            if t.mod_source_index >= len(self.tracks): t.mod_source_index = -1
        self.current_track_index = max(0, min(self.current_track_index, len(self.tracks)-1))
        self.refresh_track_list(); self.refresh_mod_source_combo()
        self.track_list.setCurrentRow(self.current_track_index)

    def new_project(self):
        self.tracks=[Track(name="Track 1")]; self.current_track_index=0; self.master=MasterClock()
        self.master_mode.setCurrentIndex(0); self.master_seconds.setValue(self.master.fixed_seconds)
        self.refresh_track_list(); self.refresh_mod_source_combo()
        self.track_list.setCurrentRow(0); self.refresh_ui_from_track(); self.rebuild_grid()
        self.wave.setData([], [])

    def refresh_ui_from_track(self):
        t=self.current_track()
        self.track_gain.blockSignals(True)
        self.track_gain.setValue(float(getattr(t,'gain_db',0.0)))
        self.track_gain.blockSignals(False)
        self.track_name.setText(t.name)
        self.mode_combo.setCurrentIndex(0 if t.mode=="tempo_bpm" else (1 if t.mode=="tempo_spm" else 2))
        self.bpm.setValue(float(t.bpm)); self.beats.setValue(int(t.beats_per_measure))
        self.spm.setValue(float(t.seconds_per_measure)); self.measures.setValue(int(t.measures))
        self.duration.setValue(float(t.duration_seconds))
        self.offset.setValue(float(t.start_offset_seconds)); self.loop.setChecked(bool(t.loop_to_master))
        self.sync_points.setText(t.sync_points_text)

        self.refresh_mod_source_combo()
        self.mod_source.setCurrentIndex((t.mod_source_index+1) if t.mod_source_index>=0 else 0)
        self.mod_amount.setValue(float(t.mod_amount)); self.mod_smooth.setValue(float(t.mod_smoothing_ms))

        self.div_mode.setCurrentIndex(0 if t.division_mode=="uniform" else 1)
        self.uniform_n.setValue(int(t.uniform_n))
        self.python_code.blockSignals(True); self.python_code.setPlainText(t.python_code); self.python_code.blockSignals(False)
        self.apply_mode_visibility()

    def apply_mode_visibility(self):
        idx=self.mode_combo.currentIndex()
        bpm_on=idx==0; spm_on=idx==1; dur_on=idx==2
        self.bpm.setEnabled(bpm_on); self.beats.setEnabled(bpm_on)
        self.measures.setEnabled(bpm_on or spm_on)
        self.spm.setEnabled(spm_on); self.duration.setEnabled(dur_on)

        div_py = (self.div_mode.currentIndex()==1)
        self.uniform_n.setEnabled(not div_py)
        self.python_code.setEnabled(div_py)

    def on_track_name_changed(self):
        t=self.current_track(); t.name=self.track_name.text().strip() or "Track"
        self.refresh_track_list(); self.refresh_mod_source_combo()

    def on_track_gain_changed(self):
        self.current_track().gain_db = float(self.track_gain.value())

    def on_mode_changed(self):
        t=self.current_track(); idx=self.mode_combo.currentIndex()
        t.mode = "tempo_bpm" if idx==0 else ("tempo_spm" if idx==1 else "duration")
        self.apply_mode_visibility(); self.rebuild_grid()

    def on_timing_changed(self):
        t=self.current_track()
        t.bpm=float(self.bpm.value()); t.beats_per_measure=int(self.beats.value())
        t.seconds_per_measure=float(self.spm.value()); t.measures=int(self.measures.value())
        t.duration_seconds=float(self.duration.value())
        self.rebuild_grid()

    def on_multiclock_changed(self):
        t=self.current_track()
        t.start_offset_seconds=float(self.offset.value()); t.loop_to_master=bool(self.loop.isChecked())

    def on_sync_changed(self):
        self.current_track().sync_points_text=self.sync_points.text().strip()

    def on_mod_changed(self):
        t=self.current_track()
        sel=self.mod_source.currentIndex()-1
        t.mod_source_index = sel if sel>=0 else -1
        t.mod_amount=float(self.mod_amount.value())
        t.mod_smoothing_ms=float(self.mod_smooth.value())

    def on_div_mode_changed(self):
        t=self.current_track()
        t.division_mode = "uniform" if self.div_mode.currentIndex()==0 else "python"
        self.apply_mode_visibility(); self.rebuild_grid()

    def on_div_params_changed(self):
        self.current_track().uniform_n=int(self.uniform_n.value()); self.rebuild_grid()

    def on_python_code_changed(self):
        self.current_track().python_code=self.python_code.toPlainText()

    def rebuild_grid(self):
        t=self.current_track()
        try:
            divs=t.build_divisions()
        except Exception as e:
            self.err("Division error", e)
            divs=divisions_uniform(max(t.total_duration(),0.01),16)
        t.ensure_cells(len(divs))
        self.grid.blockSignals(True)
        self.grid.setRowCount(1)
        self.grid.setColumnCount(max(1,len(divs)))
        starts=np.cumsum([0.0]+divs[:-1]) if divs else [0.0]
        for i in range(self.grid.columnCount()):
            if i < len(divs):
                s=float(starts[i]); e=float(s+divs[i])
                hdr=f"{i}\n{round(s,3)}–{round(e,3)}s"
            else:
                hdr=f"{i}"
            self.grid.setHorizontalHeaderItem(i,QTableWidgetItem(hdr))
        self.grid.setVerticalHeaderItem(0,QTableWidgetItem("Cells"))
        for i in range(self.grid.columnCount()):
            item=QTableWidgetItem()
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            if i < len(t.cells):
                c=t.cells[i]
                if c.kind=="empty": item.setText("")
                elif c.kind=="wav": item.setText("WAV:\n"+os.path.basename(c.path))
                else: item.setText("PY:\n"+os.path.basename(c.path))
            self.grid.setItem(0,i,item)
        self.grid.resizeColumnsToContents()
        self.grid.blockSignals(False)

    def on_cell_clicked(self,row,col):
        if row!=0: return
        t=self.current_track()
        if col>=len(t.cells): return
        path,_=QFileDialog.getOpenFileName(self,"Assign cell source (.wav or .py)","",
                                          "Audio or Python (*.wav *.py);;WAV (*.wav);;Python (*.py);;All (*)")
        if not path: return
        ext=os.path.splitext(path)[1].lower()
        if ext==".wav":
            t.cells[col]=CellSource("wav",path)
        elif ext==".py":
            t.cells[col]=CellSource("py",path)
        else:
            QMessageBox.warning(self,"Unsupported","Choose a .wav or .py file.")
            return
        self.rebuild_grid()

    def on_grid_context_menu(self,pos):
        idx=self.grid.indexAt(pos)
        if not idx.isValid(): return
        if idx.row()!=0: return
        t=self.current_track()
        if 0<=idx.column()<len(t.cells):
            t.cells[idx.column()]=CellSource("empty","")
            self.rebuild_grid()

    def render_preview(self):
        try:
            y=self.render_mixdown()
            if self.normalize_check.isChecked():
                y=_normalize_peak(y,0.98)
            self.wave.setData(np.arange(len(y)), y)
        except Exception as e:
            self.err("Render error", e)

    def export_wav(self):
        try:
            y=self.render_mixdown()
            if self.normalize_check.isChecked():
                y=_normalize_peak(y,0.98)
            out,_=QFileDialog.getSaveFileName(self,"Export WAV","mixdown.wav","WAV (*.wav)")
            if not out: return
            sf.write(out, y, self.sr)
            QMessageBox.information(self,"Export",f"Saved:\n{out}")
        except Exception as e:
            self.err("Export error", e)

    # -------- Track restart events --------
    def _collect_restart_events(self, master_dur: float) -> List[set]:
        """
        Returns: restart_samples_per_track[ti] = set(sample indices on master timeline)
        Any event emitted by any cell can request restarts of other tracks.
        """
        n_tracks = len(self.tracks)
        restarts: List[set] = [set() for _ in range(n_tracks)]
        max_events = 20000
        events_count = 0

        # name->index map (current names)
        name_map = {t.name: i for i, t in enumerate(self.tracks)}

        for src_ti, t in enumerate(self.tracks):
            divs = t.build_divisions()
            if not divs:
                continue
            t.ensure_cells(len(divs))
            pat_dur = float(sum(divs))
            if pat_dur <= 1e-9:
                continue

            starts = np.cumsum([0.0] + divs[:-1])
            # occurrences of this track pattern on master (if looping)
            start0 = float(t.start_offset_seconds)
            if t.loop_to_master:
                if pat_dur <= 1e-9:
                    occs = 0
                else:
                    occs = int(math.ceil(max(0.0, master_dur - start0) / pat_dur)) + 1
            else:
                occs = 1

            occs = max(0, min(occs, 10000))  # safety cap

            for occ in range(occs):
                occ_start = start0 + occ * pat_dur
                if occ_start > master_dur:
                    break

                for ci, (cell, cs) in enumerate(zip(t.cells, starts)):
                    if cell.kind != "py" or not cell.path:
                        continue
                    try:
                        mod = load_py_module(cell.path)
                    except Exception:
                        continue
                    if mod.event is None:
                        continue

                    master_time = occ_start + float(cs)
                    if master_time < 0.0 or master_time > master_dur:
                        continue

                    ctx = {
                        "track_index": src_ti,
                        "track_name": t.name,
                        "cell_index": ci,
                        "cells_total": len(divs),
                        "cell_start": float(cs),
                        "cell_duration": float(divs[ci]),
                        "track_pattern_duration": float(pat_dur),
                        "track_offset": float(t.start_offset_seconds),
                        "track_loop_to_master": bool(t.loop_to_master),
                        "track_sync_points_master": parse_float_list(t.sync_points_text),
                        "master_time": float(master_time),
                        "master_duration": float(master_dur),
                        "tracks": [{"index": i, "name": tt.name} for i, tt in enumerate(self.tracks)],
                    }
                    try:
                        ev = mod.event(ctx)
                    except Exception:
                        continue
                    if not isinstance(ev, dict):
                        continue

                    targets = ev.get("restart_tracks", [])
                    if targets == "all":
                        target_idx = list(range(n_tracks))
                    elif targets == "all_except_self":
                        target_idx = [i for i in range(n_tracks) if i != src_ti]
                    else:
                        target_idx = []
                        if isinstance(targets, (list, tuple)):
                            for it in targets:
                                if isinstance(it, int) and 0 <= it < n_tracks:
                                    target_idx.append(it)
                                elif isinstance(it, str) and it in name_map:
                                    target_idx.append(name_map[it])

                    # optional delay (sec)
                    delay = 0.0
                    try:
                        delay = float(ev.get("delay", 0.0) or 0.0)
                    except Exception:
                        delay = 0.0

                    sidx = int(round((master_time + delay) * self.sr))
                    if 0 <= sidx < int(round(master_dur * self.sr)) + 1:
                        for ti in target_idx:
                            restarts[ti].add(sidx)
                        events_count += 1
                        if events_count >= max_events:
                            return restarts

        return restarts

    def render_mixdown(self)->np.ndarray:
        master_dur=self.master.duration(self.tracks)
        n_total=int(round(master_dur*self.sr))
        mix=np.zeros((n_total,),dtype=np.float32)
        rendered_tracks: List[np.ndarray] = []

        # precompute restart samples requested by event-cells
        restarts = self._collect_restart_events(master_dur)

        for ti, t in enumerate(self.tracks):
            divs=t.build_divisions()
            if not divs:
                rendered_tracks.append(np.zeros((n_total,),dtype=np.float32))
                continue
            t.ensure_cells(len(divs))
            pat_dur=float(sum(divs))
            if pat_dur<=1e-9:
                rendered_tracks.append(np.zeros((n_total,),dtype=np.float32))
                continue

            sync_pts = parse_float_list(t.sync_points_text)
            # render pattern with per-cell context
            pat = self._render_track_pattern(ti, t, divs, pat_dur, sync_pts)
            pat_n=len(pat)

            # modulation (directed: only earlier tracks)
            speed=None
            if t.mod_source_index>=0 and t.mod_amount>0 and t.mod_source_index < ti:
                src_audio = rendered_tracks[t.mod_source_index]
                win=int(round(max(0.0,t.mod_smoothing_ms)*0.001*self.sr))
                env=rms_envelope(src_audio, max(1,win))
                m=float(np.max(env)) if len(env) else 0.0
                if m>1e-12:
                    env=(env/m).astype(np.float32)
                speed = np.clip(1.0 + float(t.mod_amount)*env, 0.25, 4.0).astype(np.float32)

            # phase resets = own sync points + external restart events
            reset = set(int(round(p*self.sr)) for p in sync_pts if p>=0.0)
            reset |= restarts[ti]

            y=np.zeros((n_total,),dtype=np.float32)
            self._render_track_to_master(y, pat, pat_n, t.start_offset_seconds, bool(t.loop_to_master), speed, reset)
            gain_lin = float(10.0 ** (float(getattr(t,'gain_db',0.0)) / 20.0))
            if gain_lin != 1.0:
                y = (y * gain_lin).astype(np.float32, copy=False)
            mix += y
            rendered_tracks.append(y)

        return np.clip(mix, -1.0, 1.0).astype(np.float32, copy=False)

    def _render_track_pattern(self, ti:int, t:Track, divs:List[float], pat_dur:float, sync_pts_master:List[float])->np.ndarray:
        pat_n=max(1,int(round(pat_dur*self.sr)))
        pat=np.zeros((pat_n,),dtype=np.float32)
        starts=np.cumsum([0.0]+divs[:-1])
        total_cells=len(divs)

        for ci,(cell,dur,st) in enumerate(zip(t.cells, divs, starts)):
            if cell.kind=="empty": 
                continue
            start_samp=int(round(float(st)*self.sr))
            ctx = {
                "track_index": ti,
                "track_name": t.name,
                "cell_index": ci,
                "cells_total": total_cells,
                "cell_start": float(st),
                "cell_duration": float(dur),
                "track_pattern_duration": float(pat_dur),
                "track_offset": float(t.start_offset_seconds),
                "track_loop_to_master": bool(t.loop_to_master),
                "track_sync_points_master": list(sync_pts_master),
            }
            seg=self.render_cell_audio(cell, float(dur), ctx)
            end_samp=min(pat_n, start_samp+len(seg))
            if end_samp>start_samp:
                pat[start_samp:end_samp] += seg[:end_samp-start_samp]
        return np.clip(pat, -1.0, 1.0).astype(np.float32, copy=False)

    def _render_track_to_master(self, out:np.ndarray, pat:np.ndarray, pat_n:int,
                               start_offset_seconds:float, loop_to_master:bool,
                               speed:Optional[np.ndarray], reset_samples:set):
        n_total=len(out)
        start_idx=int(round(start_offset_seconds*self.sr))
        phase=0.0
        if start_idx < 0:
            pre=-start_idx
            if speed is None:
                phase=float(pre)
            else:
                phase=float(np.sum(speed[:min(pre,len(speed))]))
        for i in range(n_total):
            if i in reset_samples:
                phase=0.0
            local=i-start_idx
            if local<0:
                continue
            if not loop_to_master and local>=pat_n:
                break
            inc=1.0
            if speed is not None and i < len(speed):
                inc=float(speed[i])
            idx = int(phase) % pat_n if loop_to_master else int(phase)
            if 0<=idx<pat_n:
                out[i] += float(pat[idx])
            phase += inc
            if not loop_to_master and phase>=pat_n:
                break

    def render_cell_audio(self, cell:CellSource, duration:float, context:Dict[str,Any])->np.ndarray:
        """Renders audio (if any). Event-only scripts just return silence."""
        duration=max(0.0,float(duration))
        if duration<=0:
            return np.zeros((0,),dtype=np.float32)
        if cell.kind=="wav":
            x=load_wav_any(cell.path, self.sr)
            return _fit_to_duration(x, self.sr, duration)
        if cell.kind=="py":
            mod=load_py_module(cell.path)
            if mod.generate is None:
                return np.zeros((int(round(duration*self.sr)),), dtype=np.float32)
            try:
                if len(inspect.signature(mod.generate).parameters)==3:
                    x=mod.generate(self.sr, duration, context)
                else:
                    x=mod.generate(self.sr, duration)
            except TypeError:
                x=mod.generate(self.sr, duration)
            x=_to_mono(np.asarray(x,dtype=np.float32))
            return _fit_to_duration(x, self.sr, duration)
        return np.zeros((int(round(duration*self.sr)),),dtype=np.float32)

def main():
    app=QApplication(sys.argv)
    w=MainWindow(); w.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
