import sys
import time
from collections import deque

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

from model import ForestFireModel, ModelParams
from viewer_qt import FireViewWidget
from osc_out import OSCSender, OSCConfig
from watchers import WatchEngine, ThresholdRule
from led import LedIndicator


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forest Fire — Qt (Balanced View) + Rule Editor + OSC (Pure Data)")
        self.resize(1120, 720)

        params = ModelParams()
        self.model = ForestFireModel(params=params, seed=int(time.time()) & 0xFFFF_FFFF)

        # OSC
        self.osc = OSCSender(OSCConfig(host="127.0.0.1", port=9000, enabled=True))
        self.watch = WatchEngine()

        # Starter rules
        self.watch.set_rules([
            ThresholdRule(enabled=True, metric_key="burning", op=">", threshold=250, hysteresis=10,
                          edge="rising", cooldown_s=0.5, osc_address="/fire/burning_hi"),
            ThresholdRule(enabled=True, metric_key="ignitions", op=">", threshold=40, hysteresis=2,
                          edge="rising", cooldown_s=0.25, osc_address="/fire/ignitions_spike"),
        ])

        # ---- Central layout ----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Left controls in scroll area
        left_wrap = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_wrap)
        left_layout.setContentsMargins(8, 8, 8, 8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_wrap)

        # Right split: top fire view, bottom plots (two stacked) — balanced 50/50
        self.fire_view = FireViewWidget(self.model)

        pg.setConfigOptions(antialias=True)
        self.plot_counts = pg.PlotWidget(title="Cell counts")
        self.plot_counts.addLegend(offset=(10, 10))
        self.plot_rates = pg.PlotWidget(title="Ignitions / Embers / Rain")

        # Distinguish counts by colour (requested)
        self.curve_trees = self.plot_counts.plot(
            pen=pg.mkPen((80, 210, 110), width=2),
            name="trees"
        )
        self.curve_burn  = self.plot_counts.plot(
            pen=pg.mkPen((255, 140, 40), width=2),
            name="burning"
        )
        self.curve_ash   = self.plot_counts.plot(
            pen=pg.mkPen((190, 190, 190), width=2),
            name="ash"
        )
        self.curve_empty = self.plot_counts.plot(
            pen=pg.mkPen((120, 80, 80), width=2),
            name="empty"
        )

        self.plot_rates.addLegend(offset=(10, 10))
        # Distinguish rates by colour (requested)
        self.curve_ign = self.plot_rates.plot(
            pen=pg.mkPen((255, 120, 40), width=2),
            name="ignitions"
        )
        self.curve_emb = self.plot_rates.plot(
            pen=pg.mkPen((80, 200, 255), width=2),
            name="embers"
        )
        self.curve_rain = self.plot_rates.plot(
            pen=pg.mkPen((120, 140, 255), width=2),
            name="rain"
        )

        self.plots_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.plots_splitter.addWidget(self.plot_counts)
        self.plots_splitter.addWidget(self.plot_rates)

        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.right_splitter.addWidget(self.fire_view)
        self.right_splitter.addWidget(self.plots_splitter)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_splitter.addWidget(scroll)
        main_splitter.addWidget(self.right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([360, 760])

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_splitter)

        # ---------- Controls ----------
        row = QtWidgets.QHBoxLayout()
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_random = QtWidgets.QPushButton("Randomize")
        row.addWidget(self.btn_pause)
        row.addWidget(self.btn_reset)
        row.addWidget(self.btn_random)
        left_layout.addLayout(row)

        self._paused = False

        def add_slider(title, vmin, vmax, init, step):
            box = QtWidgets.QGroupBox(title)
            lay = QtWidgets.QVBoxLayout(box)
            s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            scale = int(round(1.0/step))
            s.setMinimum(int(round(vmin*scale)))
            s.setMaximum(int(round(vmax*scale)))
            s.setValue(int(round(init*scale)))
            lbl = QtWidgets.QLabel(f"{init:.6g}")
            lay.addWidget(s)
            lay.addWidget(lbl)
            left_layout.addWidget(box)
            return s, lbl, scale

        self.s_spread, self.lbl_spread, self.sc_spread = add_slider("Base spread", 0.01, 1.0, self.model.params.base_spread, 0.005)
        self.s_light,  self.lbl_light,  self.sc_light  = add_slider("Lightning rate", 0.0, 2e-5, self.model.params.lightning_rate, 1e-7)
        self.s_ember,  self.lbl_ember,  self.sc_ember  = add_slider("Ember rate", 0.0, 0.15, self.model.params.ember_rate, 0.001)
        self.s_rainc,  self.lbl_rainc,  self.sc_rainc  = add_slider("Rain chance", 0.0, 0.08, self.model.params.rain_chance, 0.001)
        self.s_rains,  self.lbl_rains,  self.sc_rains  = add_slider("Rain strength", 0.0, 1.0, self.model.params.rain_strength, 0.01)
        self.s_winds,  self.lbl_winds,  self.sc_winds  = add_slider("Wind strength", 0.0, 1.5, self.model.params.wind_strength, 0.01)

        wind_box = QtWidgets.QGroupBox("Wind direction (deg)")
        wind_lay = QtWidgets.QVBoxLayout(wind_box)
        self.dial_wind = QtWidgets.QDial()
        self.dial_wind.setRange(0, 359)
        self.dial_wind.setValue(int(self.model.params.wind_dir_deg) % 360)
        self.lbl_wind = QtWidgets.QLabel(f"{self.model.params.wind_dir_deg:.0f}°")
        wind_lay.addWidget(self.dial_wind)
        wind_lay.addWidget(self.lbl_wind)
        left_layout.addWidget(wind_box)

        # OSC Config
        osc_box = QtWidgets.QGroupBox("OSC Output (Pure Data)")
        osc_lay = QtWidgets.QGridLayout(osc_box)
        self.chk_osc = QtWidgets.QCheckBox("Enabled")
        self.chk_osc.setChecked(True)
        self.osc_host = QtWidgets.QLineEdit("127.0.0.1")
        self.osc_port = QtWidgets.QSpinBox()
        self.osc_port.setRange(1, 65535)
        self.osc_port.setValue(9000)
        osc_lay.addWidget(self.chk_osc, 0, 0, 1, 2)
        osc_lay.addWidget(QtWidgets.QLabel("Host"), 1, 0)
        osc_lay.addWidget(self.osc_host, 1, 1)
        osc_lay.addWidget(QtWidgets.QLabel("Port"), 2, 0)
        osc_lay.addWidget(self.osc_port, 2, 1)
        left_layout.addWidget(osc_box)

        def apply_osc_cfg():
            self.osc.cfg.enabled = self.chk_osc.isChecked()
            self.osc.set_target(self.osc_host.text().strip(), int(self.osc_port.value()))

        self.chk_osc.toggled.connect(apply_osc_cfg)
        self.osc_host.editingFinished.connect(apply_osc_cfg)
        self.osc_port.valueChanged.connect(lambda _: apply_osc_cfg())

        # Rule editor
        self.metric_keys = ["trees", "burning", "ash", "empty", "ignitions", "embers", "rain"]
        self._rule_leds = []
        self._rule_lines = []  # list of dicts: {"lo":line,"hi":line|None,"plot":PlotWidget}

        rule_box = QtWidgets.QGroupBox("Rule Editor (Threshold → OSC)")
        rule_lay = QtWidgets.QVBoxLayout(rule_box)

        btns = QtWidgets.QHBoxLayout()
        self.btn_add_rule = QtWidgets.QPushButton("Add rule")
        self.btn_del_rule = QtWidgets.QPushButton("Remove selected")
        btns.addWidget(self.btn_add_rule)
        btns.addWidget(self.btn_del_rule)
        btns.addStretch(1)
        rule_lay.addLayout(btns)

        self.rule_table = QtWidgets.QTableWidget()
        self.rule_table.setColumnCount(10)
        self.rule_table.setHorizontalHeaderLabels([
            "LED", "On", "Metric", "Op", "Thr", "Hi", "Hyst", "Edge", "Cool(s)", "OSC Address"
        ])
        self.rule_table.verticalHeader().setVisible(False)
        self.rule_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.rule_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.rule_table.horizontalHeader().setStretchLastSection(True)
        self.rule_table.setAlternatingRowColors(True)
        self.rule_table.setMinimumHeight(240)

        hh = self.rule_table.horizontalHeader()
        for c in range(0, 9):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(9, QtWidgets.QHeaderView.ResizeMode.Stretch)

        rule_lay.addWidget(self.rule_table)
        left_layout.addWidget(rule_box)

        def make_spin(minv, maxv, step, init):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setDecimals(6)
            sp.setRange(minv, maxv)
            sp.setSingleStep(step)
            sp.setValue(init)
            sp.setAccelerated(True)
            return sp

        def metric_to_plot(metric: str):
            if metric in ("trees", "burning", "ash", "empty"):
                return self.plot_counts
            return self.plot_rates

        def clear_rule_lines():
            for entry in self._rule_lines:
                try:
                    if entry.get("lo") is not None:
                        entry["plot"].removeItem(entry["lo"])
                    if entry.get("hi") is not None:
                        entry["plot"].removeItem(entry["hi"])
                except Exception:
                    pass
            self._rule_lines = []

        def rebuild_rules_from_table():
            rules = []
            n = self.rule_table.rowCount()
            for r in range(n):
                chk = self.rule_table.cellWidget(r, 1)
                cmb_metric = self.rule_table.cellWidget(r, 2)
                cmb_op = self.rule_table.cellWidget(r, 3)
                sp_thr = self.rule_table.cellWidget(r, 4)
                sp_hi  = self.rule_table.cellWidget(r, 5)
                sp_hys = self.rule_table.cellWidget(r, 6)
                cmb_edge = self.rule_table.cellWidget(r, 7)
                sp_cool = self.rule_table.cellWidget(r, 8)
                le_addr = self.rule_table.cellWidget(r, 9)

                addr = le_addr.text().strip()
                if not addr.startswith("/"):
                    addr = "/" + addr

                rules.append(ThresholdRule(
                    enabled=chk.isChecked(),
                    metric_key=cmb_metric.currentText(),
                    op=cmb_op.currentText(),
                    threshold=float(sp_thr.value()),
                    threshold_hi=float(sp_hi.value()),
                    hysteresis=float(sp_hys.value()),
                    edge=cmb_edge.currentText(),
                    cooldown_s=float(sp_cool.value()),
                    osc_address=addr,
                    send_state=True,
                    send_value=True,
                ))
            self.watch.set_rules(rules)
            update_threshold_lines()

        def update_threshold_lines():
            # rebuild all lines to keep implementation simple + correct
            clear_rule_lines()
            pen = pg.mkPen((210, 210, 210), width=1, style=QtCore.Qt.PenStyle.DashLine)

            for i, rule in enumerate(self.watch.rules):
                plot = metric_to_plot(rule.metric_key)
                lo = pg.InfiniteLine(pos=float(rule.threshold), angle=0, movable=False, pen=pen)
                lo.setZValue(10)
                try:
                    lo.label = pg.InfLineLabel(lo, text=f"{i}:{rule.metric_key} {rule.op} {rule.threshold:g}", position=0.98,
                                              anchors=[(1,1), (1,1)])
                except Exception:
                    pass
                plot.addItem(lo)

                hi = None
                if rule.op == "band":
                    hi = pg.InfiniteLine(pos=float(rule.threshold_hi), angle=0, movable=False, pen=pen)
                    hi.setZValue(10)
                    plot.addItem(hi)

                self._rule_lines.append({"plot": plot, "lo": lo, "hi": hi})

        def add_rule_row(rule: ThresholdRule):
            r = self.rule_table.rowCount()
            self.rule_table.insertRow(r)

            led = LedIndicator(14)
            self._rule_leds.append(led)
            self.rule_table.setCellWidget(r, 0, led)

            chk = QtWidgets.QCheckBox()
            chk.setChecked(rule.enabled)
            chk.setStyleSheet("QCheckBox{margin-left:6px;}")
            self.rule_table.setCellWidget(r, 1, chk)

            cmb_metric = QtWidgets.QComboBox()
            cmb_metric.addItems(self.metric_keys)
            if rule.metric_key in self.metric_keys:
                cmb_metric.setCurrentText(rule.metric_key)
            self.rule_table.setCellWidget(r, 2, cmb_metric)

            cmb_op = QtWidgets.QComboBox()
            cmb_op.addItems([">", "<", "band"])
            cmb_op.setCurrentText(rule.op)
            self.rule_table.setCellWidget(r, 3, cmb_op)

            sp_thr = make_spin(0.0, 1e9, 1.0, float(rule.threshold))
            sp_hi  = make_spin(0.0, 1e9, 1.0, float(rule.threshold_hi))
            sp_hys = make_spin(0.0, 1e9, 0.5, float(rule.hysteresis))
            sp_cool= make_spin(0.0, 60.0, 0.05, float(rule.cooldown_s))
            self.rule_table.setCellWidget(r, 4, sp_thr)
            self.rule_table.setCellWidget(r, 5, sp_hi)
            self.rule_table.setCellWidget(r, 6, sp_hys)

            cmb_edge = QtWidgets.QComboBox()
            cmb_edge.addItems(["rising", "falling", "both", "level"])
            cmb_edge.setCurrentText(rule.edge)
            self.rule_table.setCellWidget(r, 7, cmb_edge)

            self.rule_table.setCellWidget(r, 8, sp_cool)

            le_addr = QtWidgets.QLineEdit(rule.osc_address)
            self.rule_table.setCellWidget(r, 9, le_addr)

            def refresh_hi_enabled():
                sp_hi.setEnabled(cmb_op.currentText() == "band")

            refresh_hi_enabled()

            def on_any_change(*_):
                refresh_hi_enabled()
                rebuild_rules_from_table()

            chk.toggled.connect(on_any_change)
            cmb_metric.currentTextChanged.connect(on_any_change)
            cmb_op.currentTextChanged.connect(on_any_change)
            sp_thr.valueChanged.connect(on_any_change)
            sp_hi.valueChanged.connect(on_any_change)
            sp_hys.valueChanged.connect(on_any_change)
            cmb_edge.currentTextChanged.connect(on_any_change)
            sp_cool.valueChanged.connect(on_any_change)
            le_addr.editingFinished.connect(on_any_change)

        def remove_selected_rule():
            row = self.rule_table.currentRow()
            if row < 0:
                return
            self.rule_table.removeRow(row)
            if 0 <= row < len(self._rule_leds):
                self._rule_leds.pop(row)
            rebuild_rules_from_table()

        def add_default_rule():
            add_rule_row(ThresholdRule(
                enabled=True, metric_key="burning", op=">", threshold=200.0, threshold_hi=400.0,
                hysteresis=10.0, edge="rising", cooldown_s=0.25, osc_address="/fire/burning_hi"
            ))
            rebuild_rules_from_table()

        self.btn_add_rule.clicked.connect(add_default_rule)
        self.btn_del_rule.clicked.connect(remove_selected_rule)

        for rule in self.watch.rules:
            add_rule_row(rule)
        rebuild_rules_from_table()

        left_layout.addStretch(1)

        # ---- Plot history buffers ----
        self.hist = 900
        self.ts = deque(maxlen=self.hist)
        self.trees = deque(maxlen=self.hist)
        self.burning = deque(maxlen=self.hist)
        self.ash = deque(maxlen=self.hist)
        self.empty = deque(maxlen=self.hist)
        self.ign = deque(maxlen=self.hist)
        self.emb = deque(maxlen=self.hist)
        self.rain = deque(maxlen=self.hist)

        # ---- Wiring ----
        def update_params():
            self.model.params.base_spread = self.s_spread.value() / self.sc_spread
            self.lbl_spread.setText(f"{self.model.params.base_spread:.6g}")

            self.model.params.lightning_rate = self.s_light.value() / self.sc_light
            self.lbl_light.setText(f"{self.model.params.lightning_rate:.6g}")

            self.model.params.ember_rate = self.s_ember.value() / self.sc_ember
            self.lbl_ember.setText(f"{self.model.params.ember_rate:.6g}")

            self.model.params.rain_chance = self.s_rainc.value() / self.sc_rainc
            self.lbl_rainc.setText(f"{self.model.params.rain_chance:.6g}")

            self.model.params.rain_strength = self.s_rains.value() / self.sc_rains
            self.lbl_rains.setText(f"{self.model.params.rain_strength:.6g}")

            self.model.params.wind_strength = self.s_winds.value() / self.sc_winds
            self.lbl_winds.setText(f"{self.model.params.wind_strength:.6g}")

        for s in [self.s_spread, self.s_light, self.s_ember, self.s_rainc, self.s_rains, self.s_winds]:
            s.valueChanged.connect(update_params)

        def wind_changed(v):
            self.model.params.wind_dir_deg = float(v)
            self.lbl_wind.setText(f"{v:.0f}°")
        self.dial_wind.valueChanged.connect(wind_changed)

        def toggle_pause():
            self._paused = not self._paused
            self.btn_pause.setText("Resume" if self._paused else "Pause")
        self.btn_pause.clicked.connect(toggle_pause)

        self.btn_reset.clicked.connect(lambda: self.model.reset())
        self.btn_random.clicked.connect(lambda: self.model.randomize())

        # Timers
        self.timer = QtCore.QTimer()
        self.timer.setInterval(33)  # ~30 Hz
        self.timer.timeout.connect(self.tick)
        self.timer.start()

        # Balance splitters after show
        QtCore.QTimer.singleShot(0, self._lock_balanced_split)

    def _lock_balanced_split(self):
        h = max(2, self.right_splitter.height())
        self.right_splitter.setSizes([h // 2, h // 2])

        ph = max(2, self.plots_splitter.height())
        self.plots_splitter.setSizes([ph // 2, ph // 2])

    def tick(self):
        if not self._paused:
            self.model.step()

        stats = self.model.get_stats()

        # update plots
        self.ts.append(stats["t"])
        self.trees.append(stats["trees"])
        self.burning.append(stats["burning"])
        self.ash.append(stats["ash"])
        self.empty.append(stats["empty"])
        self.ign.append(stats["ignitions"])
        self.emb.append(stats["embers"])
        self.rain.append(stats["rain"])

        x = np.array(self.ts, dtype=np.float32)
        self.curve_trees.setData(x, np.array(self.trees))
        self.curve_burn.setData(x, np.array(self.burning))
        self.curve_ash.setData(x, np.array(self.ash))
        self.curve_empty.setData(x, np.array(self.empty))

        self.curve_ign.setData(x, np.array(self.ign))
        self.curve_emb.setData(x, np.array(self.emb))
        self.curve_rain.setData(x, np.array(self.rain))

        # view
        self.fire_view.set_frame(self.model.render_rgb())
        self.fire_view.update()

        # OSC + LEDs
        led_states = self.watch.update(stats, self.osc.send)
        for i, (en, active) in enumerate(led_states):
            if i < len(self._rule_leds):
                self._rule_leds[i].set_state(en, active)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
