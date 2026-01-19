from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np


class FireViewWidget(QtWidgets.QWidget):
    """Qt top-down fire view that always fits its frame.

    - The simulation image is scaled to fit the widget while preserving aspect ratio.
    - Mouse painting correctly maps through the fitted transform.
    - Mouse wheel adjusts an additional zoom multiplier (fit * zoom).
    """
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model

        # zoom multiplier on top of fit-to-frame scale
        self._zoom = 1.0
        self._zoom_min = 0.5
        self._zoom_max = 8.0

        self._brush = 2
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)

        self._pix = None
        self._hud_font = QtGui.QFont("Consolas", 11)
        self._hud_font.setBold(True)

        # Cached transform: map model->widget
        self._draw_scale = 1.0
        self._draw_offx = 0.0
        self._draw_offy = 0.0

    def _recalc_transform(self):
        if self.model is None:
            self._draw_scale = 1.0
            self._draw_offx = 0.0
            self._draw_offy = 0.0
            return

        H, W = self.model.state.shape
        if W <= 0 or H <= 0:
            self._draw_scale = 1.0
            self._draw_offx = 0.0
            self._draw_offy = 0.0
            return

        # Reserve a little room for HUD text
        avail = self.rect().adjusted(0, 24, 0, 0)
        aw = max(1, avail.width())
        ah = max(1, avail.height())

        fit = min(aw / float(W), ah / float(H))
        scale = max(0.01, fit * float(self._zoom))

        tw = float(W) * scale
        th = float(H) * scale

        self._draw_scale = scale
        self._draw_offx = avail.left() + (aw - tw) * 0.5
        self._draw_offy = avail.top() + (ah - th) * 0.5

    def wheelEvent(self, e: QtGui.QWheelEvent):
        delta = e.angleDelta().y()
        if delta == 0:
            return
        step = 1.1
        if delta > 0:
            self._zoom = min(self._zoom_max, self._zoom * step)
        else:
            self._zoom = max(self._zoom_min, self._zoom / step)
        self.update()

    def _widget_to_grid(self, wx: float, wy: float):
        self._recalc_transform()
        gx = int((wx - self._draw_offx) / self._draw_scale)
        gy = int((wy - self._draw_offy) / self._draw_scale)
        H, W = self.model.state.shape
        if 0 <= gx < W and 0 <= gy < H:
            return gx, gy
        return None

    def _apply_brush(self, gx: int, gy: int, buttons, shift: bool):
        if buttons & QtCore.Qt.MouseButton.LeftButton:
            if shift:
                self.model.set_tree_at(gx, gy, radius=self._brush)
            else:
                self.model.ignite_at(gx, gy, radius=self._brush)
        elif buttons & QtCore.Qt.MouseButton.RightButton:
            self.model.clear_at(gx, gy, radius=self._brush)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        self.mouseMoveEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.model is None:
            return
        if e.buttons() == QtCore.Qt.MouseButton.NoButton:
            return

        pos = self._widget_to_grid(e.position().x(), e.position().y())
        if pos is None:
            return

        gx, gy = pos
        shift = bool(e.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        self._apply_brush(gx, gy, e.buttons(), shift)
        self.update()

    def set_frame(self, frame_rgb: np.ndarray):
        h, w, _ = frame_rgb.shape
        qimg = QtGui.QImage(frame_rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888).copy()
        self._pix = QtGui.QPixmap.fromImage(qimg)

    def paintEvent(self, e):
        self._recalc_transform()

        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(12, 12, 12))

        if self._pix is not None and self.model is not None:
            H, W = self.model.state.shape
            tw = int(round(W * self._draw_scale))
            th = int(round(H * self._draw_scale))

            # Draw scaled image centered in its frame
            target = QtCore.QRect(int(round(self._draw_offx)), int(round(self._draw_offy)), tw, th)
            p.drawPixmap(target, self._pix)

            # Frame border
            p.setPen(QtGui.QPen(QtGui.QColor(70, 70, 70), 1))
            p.drawRect(target.adjusted(0, 0, -1, -1))

        # HUD
        p.setFont(self._hud_font)
        p.setPen(QtGui.QColor(240, 240, 240))
        stats = self.model.get_stats() if self.model is not None else {}
        if stats:
            hud = (
                f"t={stats['t']}  burning={stats['burning']}  ignitions={stats['ignitions']}  "
                f"embers={stats['embers']}  zoom={self._zoom:.2f}  "
                "(LMB ignite, Shift+LMB tree, RMB clear)"
            )
        else:
            hud = "No model"
        p.drawText(10, 18, hud)

        p.end()
