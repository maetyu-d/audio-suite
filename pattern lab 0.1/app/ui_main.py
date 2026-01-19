from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QUrl
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QPushButton, QDoubleSpinBox, QSpinBox,
    QFileDialog, QGroupBox, QMessageBox, QSplitter, QTabWidget,
    QLineEdit, QDialog, QTextEdit
)

import pyqtgraph as pg
import importlib

try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
    HAVE_MEDIA = True
except Exception:
    HAVE_MEDIA = False

from .events import RenderConfig
from . import patterns
from .preset_io import load_preset, save_preset
from .renderer import render
from .script_host import invalidate_cache


class PresetJsonEditor(QDialog):
    def __init__(self, parent: QWidget, path: Path):
        super().__init__(parent)
        self.setWindowTitle(f"Edit preset JSON — {path.name}")
        self.resize(860, 620)
        self.path = path

        lay = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        lay.addWidget(self.text, 1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_save = QPushButton("Save")
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_save)
        lay.addLayout(btns)

        try:
            self.text.setPlainText(path.read_text(encoding='utf-8'))
        except Exception as e:
            self.text.setPlainText(f"{{\n  \"error\": \"Failed to read preset: {type(e).__name__}: {e}\"\n}}")

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._save)

    def _save(self):
        raw = self.text.toPlainText()
        try:
            obj = json.loads(raw)
        except Exception as e:
            QMessageBox.critical(self, "Invalid JSON", f"This preset is not valid JSON.\n\n{type(e).__name__}: {e}")
            return
        if not isinstance(obj, dict):
            QMessageBox.critical(self, "Invalid preset", "Preset JSON must be an object (a JSON dict).")
            return
        try:
            self.path.write_text(json.dumps(obj, indent=2), encoding='utf-8')
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not write preset file.\n\n{type(e).__name__}: {e}")
            return
        self.accept()


class ScriptEditor(QDialog):
    def __init__(self, parent: QWidget, path: Path):
        super().__init__(parent)
        self.setWindowTitle(f"Edit code — {path.name}")
        self.resize(900, 680)
        self.path = path

        lay = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        lay.addWidget(self.text, 1)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_save = QPushButton("Save")
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_save)
        lay.addLayout(btns)

        try:
            self.text.setPlainText(path.read_text(encoding='utf-8'))
        except Exception as e:
            self.text.setPlainText(f"# Failed to read file: {type(e).__name__}: {e}\n")

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._save)

    def _save(self):
        raw = self.text.toPlainText()
        try:
            compile(raw, str(self.path), 'exec')
        except Exception as e:
            QMessageBox.critical(self, "Syntax error", f"This file has a Python syntax error.\n\n{type(e).__name__}: {e}")
            return
        try:
            self.path.write_text(raw, encoding='utf-8')
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not write file.\n\n{type(e).__name__}: {e}")
            return
        self.accept()


class RenderWorker(QObject):
    finished = pyqtSignal(np.ndarray, object)  # audio, events
    failed = pyqtSignal(str)

    def __init__(self, generator: str, gen_kwargs: dict, cfg: RenderConfig):
        super().__init__()
        self.generator = str(generator)
        self.gen_kwargs = dict(gen_kwargs or {})
        self.cfg = cfg

    def run(self):
        try:
            ev = patterns.generate(self.generator, self.cfg, **self.gen_kwargs)
            y, ev2 = render(ev, self.cfg)
            self.finished.emit(y, ev2)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pattern Lab")
        self.resize(1200, 720)

        self._tmp_wav: Optional[Path] = None
        self._player = None
        self._audio_out = None
        self._audio: Optional[np.ndarray] = None
        self._audio_sr: Optional[int] = None
        self._events = None
        self._current_preset_path: Optional[Path] = None
        self._current_script_path: Optional[Path] = None
        self._current_script_entry: str = 'generate'

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)
        main.addWidget(splitter)

        # LEFT: controls
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(8, 8, 8, 8)
        left_l.setSpacing(6)
        splitter.addWidget(left)

        # RIGHT: plots
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 8, 8, 8)
        right_l.setSpacing(6)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # ---------- Presets ----------
        preset_box = QGroupBox("Presets")
        pb = QVBoxLayout(preset_box)
        pb.setSpacing(6)
        self.preset_combo = QComboBox()
        self._preset_paths = self._scan_presets()
        self.preset_combo.addItems([p.name for p in self._preset_paths])
        btn_row = QHBoxLayout()
        self.btn_load_preset = QPushButton("Load")
        self.btn_save_preset = QPushButton("Save As…")
        self.btn_edit_preset = QPushButton("Edit JSON…")
        self.btn_edit_script = QPushButton("Edit Pattern Code…")
        btn_row.addWidget(self.btn_load_preset)
        btn_row.addWidget(self.btn_save_preset)
        btn_row.addWidget(self.btn_edit_preset)
        btn_row.addWidget(self.btn_edit_script)
        pb.addWidget(self.preset_combo)
        pb.addLayout(btn_row)
        left_l.addWidget(preset_box)

        # ---------- Generator ----------
        gen_box = QGroupBox("Pattern Generator")
        gb = QVBoxLayout(gen_box)
        gb.setSpacing(6)
        self.gen_combo = QComboBox()
        self.gen_combo.addItems(patterns.list_generators())
        gb.addWidget(self.gen_combo)
        left_l.addWidget(gen_box)

        # ---------- Render config ----------
        cfg_box = QGroupBox("Render")
        form = QFormLayout(cfg_box)
        form.setVerticalSpacing(6)
        form.setHorizontalSpacing(10)
        self.sr = QSpinBox(); self.sr.setRange(8000, 192000); self.sr.setValue(44100)
        self.seconds = QDoubleSpinBox(); self.seconds.setRange(1.0, 600.0); self.seconds.setDecimals(2); self.seconds.setValue(22.0)
        self.bpm = QDoubleSpinBox(); self.bpm.setRange(20.0, 400.0); self.bpm.setDecimals(2); self.bpm.setValue(126.0)
        self.swing = QDoubleSpinBox(); self.swing.setRange(0.0, 0.5); self.swing.setSingleStep(0.01); self.swing.setDecimals(3); self.swing.setValue(0.08)
        self.stretch = QDoubleSpinBox(); self.stretch.setRange(0.25, 8.0); self.stretch.setSingleStep(0.05); self.stretch.setDecimals(3); self.stretch.setValue(1.0)
        self.jitter = QDoubleSpinBox(); self.jitter.setRange(0.0, 0.05); self.jitter.setSingleStep(0.0005); self.jitter.setDecimals(4); self.jitter.setValue(0.0)
        self.gain = QDoubleSpinBox(); self.gain.setRange(0.05, 1.5); self.gain.setSingleStep(0.01); self.gain.setDecimals(3); self.gain.setValue(0.9)
        self.seed = QSpinBox(); self.seed.setRange(0, 2**31-1); self.seed.setValue(3)

        form.addRow("Sample rate", self.sr)
        form.addRow("Seconds", self.seconds)
        form.addRow("BPM", self.bpm)
        form.addRow("Swing", self.swing)
        form.addRow("Time stretch", self.stretch)
        form.addRow("Micro jitter (s)", self.jitter)
        form.addRow("Master gain", self.gain)
        form.addRow("Seed", self.seed)
        left_l.addWidget(cfg_box)

        # ---------- Generator params ----------
        self.params_box = QGroupBox("Generator Parameters")
        self.params_form = QFormLayout(self.params_box)
        self.params_form.setVerticalSpacing(6)
        self.params_form.setHorizontalSpacing(10)
        left_l.addWidget(self.params_box)
        self._param_widgets: dict[str, QWidget] = {}
        self._build_param_form(self.gen_combo.currentText())

        # ---------- Actions ----------
        act_box = QGroupBox("Actions")
        ab = QVBoxLayout(act_box)
        ab.setSpacing(6)
        self.btn_render = QPushButton("Render")
        self.btn_export = QPushButton("Export WAV…"); self.btn_export.setEnabled(False)
        play_row = QHBoxLayout()
        self.btn_play = QPushButton("Play"); self.btn_stop = QPushButton("Stop")
        self.btn_play.setEnabled(False); self.btn_stop.setEnabled(False)
        play_row.addWidget(self.btn_play); play_row.addWidget(self.btn_stop)
        self.status = QLabel("Ready"); self.status.setWordWrap(True)
        ab.addWidget(self.btn_render)
        ab.addWidget(self.btn_export)
        ab.addLayout(play_row)
        ab.addWidget(self.status)
        left_l.addWidget(act_box)
        left_l.addStretch(1)

        # ---------- Plots ----------
        tabs = QTabWidget()
        right_l.addWidget(tabs)

        self.wave = pg.PlotWidget(title="Waveform")
        self.wave.showGrid(x=True, y=True, alpha=0.25)
        self.wave.setLabel('bottom', 'Time', units='s')
        self.wave.setLabel('left', 'Amplitude')
        tabs.addTab(self.wave, "Waveform")

        self.roll = pg.PlotWidget(title="Piano Roll")
        self.roll.showGrid(x=True, y=True, alpha=0.25)
        self.roll.setLabel('bottom', 'Time', units='s')
        self.roll.setLabel('left', 'MIDI')
        tabs.addTab(self.roll, "Piano Roll")

        self.spec = pg.ImageView(view=pg.PlotItem())
        self.spec.getView().setLabel('bottom', 'Time')
        self.spec.getView().setLabel('left', 'Freq bin')
        tabs.addTab(self.spec, "Spectrogram")

        # connections
        self.gen_combo.currentTextChanged.connect(self._on_generator_changed)
        self.btn_render.clicked.connect(self._on_render)
        self.btn_export.clicked.connect(self._on_export)
        self.btn_load_preset.clicked.connect(self._on_load_preset)
        self.btn_save_preset.clicked.connect(self._on_save_preset)
        self.btn_edit_preset.clicked.connect(self._on_edit_preset_json)
        self.btn_edit_script.clicked.connect(self._on_edit_script)
        self.btn_play.clicked.connect(self._on_play)
        self.btn_stop.clicked.connect(self._on_stop)

        self._update_media_buttons()
        self._update_script_button()

    # ---------------- presets ----------------
    def _scan_presets(self) -> list[Path]:
        preset_dir = Path(__file__).resolve().parent.parent / 'presets'
        preset_dir.mkdir(exist_ok=True)
        return sorted(preset_dir.glob('*.json'))

    def _on_load_preset(self):
        if not self._preset_paths:
            QMessageBox.information(self, "Presets", "No presets found in ./presets")
            return
        p = self._preset_paths[self.preset_combo.currentIndex()]
        self._current_preset_path = p
        preset = load_preset(p)
        self._apply_preset(preset)

    def _on_edit_preset_json(self):
        if not self._preset_paths:
            QMessageBox.information(self, "Presets", "No presets found in ./presets")
            return
        p = self._preset_paths[self.preset_combo.currentIndex()]
        dlg = PresetJsonEditor(self, p)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        cur_name = p.name
        self._preset_paths = self._scan_presets()
        self.preset_combo.clear()
        self.preset_combo.addItems([pp.name for pp in self._preset_paths])
        idx = 0
        for i, pp in enumerate(self._preset_paths):
            if pp.name == cur_name:
                idx = i
                break
        self.preset_combo.setCurrentIndex(idx)

        if self._current_preset_path is not None and self._current_preset_path.resolve() == p.resolve():
            try:
                preset = load_preset(p)
                self._apply_preset(preset)
                self.status.setText(f"Edited preset: {p.name}")
            except Exception as e:
                QMessageBox.warning(self, "Preset", f"Saved, but failed to reload: {type(e).__name__}: {e}")
        else:
            self.status.setText(f"Edited preset: {p.name}")

    def _apply_preset(self, preset: dict):
        try:
            gen = preset.get('generator', 'Glass Cells')
            cfg = preset.get('cfg', {})
            gen_kwargs = preset.get('gen', {})

            idx = self.gen_combo.findText(gen)
            if idx >= 0:
                self.gen_combo.setCurrentIndex(idx)

            self.sr.setValue(int(cfg.get('sample_rate', self.sr.value())))
            self.seconds.setValue(float(cfg.get('seconds', self.seconds.value())))
            self.bpm.setValue(float(cfg.get('bpm', self.bpm.value())))
            self.swing.setValue(float(cfg.get('swing', self.swing.value())))
            self.stretch.setValue(float(cfg.get('time_stretch', self.stretch.value())))
            self.jitter.setValue(float(cfg.get('micro_jitter', self.jitter.value())))
            self.gain.setValue(float(cfg.get('master_gain', self.gain.value())))
            self.seed.setValue(int(cfg.get('seed', self.seed.value())))

            for k, w in self._param_widgets.items():
                if k in gen_kwargs:
                    self._set_widget_value(w, gen_kwargs[k])

            self._current_script_path = None
            self._current_script_entry = str(gen_kwargs.get('entry', 'generate'))
            if str(gen).lower().startswith('python'):
                sp = gen_kwargs.get('script_path', '')
                if sp:
                    self._current_script_path = self._resolve_script_path(str(sp))
            self._update_script_button()
            self.status.setText(f"Loaded preset: {preset.get('name', gen)}")
        except Exception as e:
            QMessageBox.warning(self, "Preset", f"Failed to apply preset: {type(e).__name__}: {e}")

    def _resolve_script_path(self, script_path: str) -> Path:
        p = Path(script_path)
        if not p.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            p = (project_root / p).resolve()
        return p

    def _current_script_from_ui(self) -> Optional[Path]:
        gen = self.gen_combo.currentText().strip().lower()
        if 'python' not in gen:
            return None
        w = self._param_widgets.get('script_path')
        if w is None:
            return None
        sp = str(self._widget_value(w) or '').strip()
        if not sp:
            return None
        return self._resolve_script_path(sp)

    def _update_script_button(self):
        self.btn_edit_script.setEnabled(True)

    # ---------------- generator params ----------------
    def _on_generator_changed(self, text: str):
        self._build_param_form(text)
        self._update_script_button()

    def _param_schema(self, gen_name: str) -> dict:
        g = gen_name.lower()
        if 'glass' in g:
            return {
                'root_midi': {'type': 'int', 'label': 'Root MIDI', 'min': 24, 'max': 96, 'value': 60},
                'scale': {'type': 'choice', 'label': 'Scale', 'choices': list(patterns.SCALES.keys()), 'value': 'glass'},
                'cell_len': {'type': 'int', 'label': 'Cell length', 'min': 4, 'max': 32, 'value': 8},
                'voices': {'type': 'int', 'label': 'Voices', 'min': 1, 'max': 6, 'value': 2},
                'drift': {'type': 'float', 'label': 'Drift', 'min': 0.0, 'max': 2.0, 'step': 0.01, 'value': 0.0},
            }
        if 'fibonacci' in g:
            return {
                'root_midi': {'type': 'int', 'label': 'Root MIDI', 'min': 24, 'max': 96, 'value': 60},
                'scale': {'type': 'choice', 'label': 'Scale', 'choices': list(patterns.SCALES.keys()), 'value': 'minor'},
                'steps': {'type': 'int', 'label': 'Steps', 'min': 8, 'max': 128, 'value': 34},
                'pulse_every': {'type': 'int', 'label': 'Pulse every', 'min': 1, 'max': 13, 'value': 2},
            }
        if 'prime' in g:
            return {
                'root_midi': {'type': 'int', 'label': 'Root MIDI', 'min': 24, 'max': 96, 'value': 57},
                'scale': {'type': 'choice', 'label': 'Scale', 'choices': list(patterns.SCALES.keys()), 'value': 'minor'},
                'base_step': {'type': 'float', 'label': 'Step (beats)', 'min': 0.125, 'max': 2.0, 'step': 0.01, 'value': 0.5},
                'prime_a': {'type': 'int', 'label': 'Prime A', 'min': 3, 'max': 199, 'value': 23},
                'prime_b': {'type': 'int', 'label': 'Prime B', 'min': 3, 'max': 199, 'value': 31},
            }
        if 'pythag' in g:
            return {
                'base_midi': {'type': 'int', 'label': 'Base MIDI', 'min': 24, 'max': 96, 'value': 52},
                'fifth_steps': {'type': 'text', 'label': 'Fifth steps', 'value': '0,1,2,3,2,1,4,5,4,3,2,1'},
            }
        if 'python' in g:
            return {
                'script_path': {'type': 'text', 'label': 'Script path (.py)', 'value': 'examples/glass_canon_5_8.py'},
                'entry': {'type': 'text', 'label': 'Entry function', 'value': 'generate'},
            }
        return {}

    def _build_param_form(self, gen_name: str):
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_widgets.clear()
        schema = self._param_schema(gen_name)
        for key, spec in schema.items():
            w = self._make_widget(spec)
            self._param_widgets[key] = w
            self.params_form.addRow(spec['label'], w)

    def _make_widget(self, spec: dict):
        t = spec['type']
        if t == 'int':
            w = QSpinBox(); w.setRange(int(spec.get('min', -999999)), int(spec.get('max', 999999)))
            w.setValue(int(spec.get('value', 0)))
            return w
        if t == 'float':
            w = QDoubleSpinBox(); w.setRange(float(spec.get('min', -1e9)), float(spec.get('max', 1e9)))
            w.setSingleStep(float(spec.get('step', 0.01)))
            w.setDecimals(4)
            w.setValue(float(spec.get('value', 0.0)))
            return w
        if t == 'choice':
            w = QComboBox(); w.addItems([str(c) for c in spec.get('choices', [])])
            val = str(spec.get('value', ''))
            idx = w.findText(val)
            if idx >= 0:
                w.setCurrentIndex(idx)
            return w
        if t == 'text':
            w = QLineEdit(); w.setText(str(spec.get('value', '')))
            return w
        w = QLineEdit(); w.setText(str(spec.get('value', '')))
        return w

    def _widget_value(self, w):
        if isinstance(w, QSpinBox):
            return int(w.value())
        if isinstance(w, QDoubleSpinBox):
            return float(w.value())
        if isinstance(w, QComboBox):
            return str(w.currentText())
        if isinstance(w, QLineEdit):
            return str(w.text())
        return None

    def _set_widget_value(self, w, v):
        if isinstance(w, QSpinBox):
            w.setValue(int(v))
        elif isinstance(w, QDoubleSpinBox):
            w.setValue(float(v))
        elif isinstance(w, QComboBox):
            idx = w.findText(str(v))
            if idx >= 0:
                w.setCurrentIndex(idx)
        elif isinstance(w, QLineEdit):
            w.setText(str(v))

    def _gen_kwargs(self) -> dict:
        kwargs = {k: self._widget_value(w) for k, w in self._param_widgets.items()}
        if 'fifth_steps' in kwargs:
            txt = str(kwargs['fifth_steps'])
            try:
                kwargs['fifth_steps'] = [int(x.strip()) for x in txt.split(',') if x.strip()]
            except Exception:
                kwargs['fifth_steps'] = [0, 1, 2, 3, 2, 1, 4, 5]
        return kwargs

    # ---------------- render/export ----------------
    def _cfg(self) -> RenderConfig:
        return RenderConfig(
            sample_rate=int(self.sr.value()),
            seconds=float(self.seconds.value()),
            bpm=float(self.bpm.value()),
            swing=float(self.swing.value()),
            time_stretch=float(self.stretch.value()),
            micro_jitter=float(self.jitter.value()),
            master_gain=float(self.gain.value()),
            seed=int(self.seed.value()),
        )

    def _on_render(self):
        self.status.setText("Rendering…")
        self.btn_render.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)

        cfg = self._cfg()
        gen = self.gen_combo.currentText()
        gen_kwargs = self._gen_kwargs()

        self._thread = QThread()
        self._worker = RenderWorker(gen, gen_kwargs, cfg)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_render_done)
        self._worker.failed.connect(self._on_render_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_render_done(self, y: np.ndarray, events):
        self._audio = y
        self._audio_sr = int(self._cfg().sample_rate)
        self._events = events

        self._plot_wave(y, self._audio_sr)
        self._plot_roll(events)
        self._plot_spec(y, self._audio_sr)

        self._invalidate_tmp()
        self.btn_export.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_render.setEnabled(True)
        self.status.setText("Rendered")

    def _on_render_failed(self, msg: str):
        self.btn_render.setEnabled(True)
        QMessageBox.critical(self, "Render failed", msg)
        self.status.setText("Render failed")

    def _invalidate_tmp(self):
        self._tmp_wav = None

    def _ensure_tmp(self):
        if self._audio is None or self._audio_sr is None:
            return None
        if self._tmp_wav is not None and self._tmp_wav.exists():
            return self._tmp_wav
        fd, p = tempfile.mkstemp(prefix='patternlab_', suffix='.wav')
        Path(p).unlink(missing_ok=True)
        sf.write(p, self._audio, int(self._audio_sr), subtype='PCM_16')
        self._tmp_wav = Path(p)
        return self._tmp_wav

    def _on_play(self):
        if not HAVE_MEDIA:
            QMessageBox.information(self, "Play", "PyQt multimedia not available in this environment.")
            return
        p = self._ensure_tmp()
        if p is None:
            return
        if self._player is None:
            self._player = QMediaPlayer()
            self._audio_out = QAudioOutput()
            self._player.setAudioOutput(self._audio_out)
        self._player.setSource(QUrl.fromLocalFile(str(p)))
        self._player.play()
        self.btn_stop.setEnabled(True)
        self.status.setText("Playing")

    def _on_stop(self):
        if self._player is not None:
            self._player.stop()
        self.btn_stop.setEnabled(False)
        self.status.setText("Stopped")

    def _update_media_buttons(self):
        if not HAVE_MEDIA:
            self.btn_play.setEnabled(False)
            self.btn_stop.setEnabled(False)

    def _on_export(self):
        if self._audio is None or self._audio_sr is None:
            return
        name, ok = QFileDialog.getSaveFileName(self, "Export WAV", "pattern_lab.wav", "WAV (*.wav)")
        if not ok or not name:
            return
        sf.write(name, self._audio, int(self._audio_sr), subtype='PCM_16')
        self.status.setText(f"Exported: {Path(name).name}")

    def _on_edit_script(self):
        p = self._current_script_path or self._current_script_from_ui()
        title = "Edit Pattern Code"
        if p is None:
            p = (Path(__file__).resolve().parent / 'patterns.py')
        if not p.exists():
            QMessageBox.warning(self, title, f"Code file not found:\n\n{p}")
            return
        dlg = ScriptEditor(self, p)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            if p.name == 'patterns.py':
                importlib.invalidate_caches()
                importlib.reload(patterns)
                self.status.setText("Edited patterns.py — reloaded pattern generators")
                # rebuild generator list may change? keep current
            else:
                invalidate_cache(p)
                self.status.setText(f"Edited script: {p.name}")
        except Exception as e:
            QMessageBox.warning(self, title, f"Saved, but reload/invalidate failed.\n\n{type(e).__name__}: {e}\n\nIf changes don't apply, restart the app.")

    def _on_save_preset(self):
        name, ok = QFileDialog.getSaveFileName(self, "Save preset", str((Path(__file__).resolve().parent.parent / 'presets') / 'my_preset.json'), "JSON (*.json)")
        if not ok or not name:
            return
        preset = {
            'name': Path(name).stem,
            'generator': self.gen_combo.currentText(),
            'cfg': self._cfg_dict(),
            'gen': self._gen_kwargs(),
        }
        save_preset(name, preset)
        self._preset_paths = self._scan_presets()
        self.preset_combo.clear()
        self.preset_combo.addItems([p.name for p in self._preset_paths])
        self.status.setText(f"Saved preset: {Path(name).name}")

    def _cfg_dict(self) -> dict:
        c = self._cfg()
        return {
            'sample_rate': c.sample_rate,
            'seconds': c.seconds,
            'bpm': c.bpm,
            'swing': c.swing,
            'time_stretch': c.time_stretch,
            'micro_jitter': c.micro_jitter,
            'master_gain': c.master_gain,
            'seed': c.seed,
        }

    # ---------------- plotting ----------------
    def _plot_wave(self, y: np.ndarray, sr: int):
        self.wave.clear()
        n = y.shape[0]
        t = np.arange(n, dtype=np.float32) / float(sr)
        self.wave.plot(t, y, pen=pg.mkPen(width=1))

    def _plot_roll(self, events):
        self.roll.clear()
        if not events:
            return
        xs = []
        ys = []
        for e in events:
            xs.append(float(getattr(e, 't0', 0.0)))
            ys.append(float(getattr(e, 'midi', 60.0)))
        self.roll.plot(xs, ys, pen=None, symbol='o', symbolSize=5)

    def _plot_spec(self, y: np.ndarray, sr: int):
        # simple magnitude spectrogram
        y = np.asarray(y, dtype=np.float32)
        nfft = 1024
        hop = 256
        if y.shape[0] < nfft:
            self.spec.setImage(np.zeros((1,1), dtype=np.float32))
            return
        win = np.hanning(nfft).astype(np.float32)
        frames = 1 + (y.shape[0] - nfft) // hop
        spec = np.empty((nfft//2+1, frames), dtype=np.float32)
        for i in range(frames):
            seg = y[i*hop:i*hop+nfft] * win
            fft = np.fft.rfft(seg)
            mag = np.abs(fft).astype(np.float32)
            spec[:, i] = 20.0*np.log10(mag + 1e-8)
        self.spec.setImage(spec.T, autoLevels=True)

