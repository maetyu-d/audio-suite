from PyQt6 import QtCore, QtGui, QtWidgets


class LedIndicator(QtWidgets.QWidget):
    """Small LED circle:
    - grey when disabled
    - red when enabled but inactive
    - green when active
    """
    def __init__(self, diameter: int = 14, parent=None):
        super().__init__(parent)
        self._enabled = True
        self._active = False
        self._diameter = int(diameter)
        self.setFixedSize(self._diameter, self._diameter)

    def set_state(self, enabled: bool, active: bool):
        enabled = bool(enabled)
        active = bool(active)
        if enabled == self._enabled and active == self._active:
            return
        self._enabled = enabled
        self._active = active
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)

        if not self._enabled:
            fill = QtGui.QColor(120, 120, 120)
            edge = QtGui.QColor(70, 70, 70)
        else:
            if self._active:
                fill = QtGui.QColor(60, 220, 90)
                edge = QtGui.QColor(30, 140, 50)
            else:
                fill = QtGui.QColor(230, 70, 70)
                edge = QtGui.QColor(150, 35, 35)

        p.setPen(QtGui.QPen(edge, 1.5))
        p.setBrush(QtGui.QBrush(fill))
        p.drawEllipse(rect)

        # subtle highlight
        hi = rect.adjusted(3, 3, -7, -7)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(255, 255, 255, 70))
        p.drawEllipse(hi)
        p.end()
