import sys
import time
import requests
import pyautogui
from os import environ
from PySide2.QtCore import (
    QCoreApplication,
    QEvent,
    QObject,
    QPointF,
    Qt,
    QTimer,
    QUrl,
    Slot,
)
from PySide2.QtGui import QGuiApplication, QMouseEvent, QCursor
from PySide2.QtQuick import QQuickItem, QQuickView
from PySide2 import QtWidgets as qtw
from PySide2 import QtCore as qtc
from PySide2 import QtQml as qtm

class HandGest(qtc.QObject):

    # Property Signals
    handXChanged = qtc.Signal()
    handYChanged = qtc.Signal()
    handZChanged = qtc.Signal()
    gestChanged = qtc.Signal()
    
    # Initialize the head tracker with 
    def __init__(self, parent=None):
        super(HandGest, self).__init__(parent)
        self.x = 0
        self.y = 0
        self.z = 0
        self.old_gest = 0
        self._gest = 0
        self.cursor = QCursor()
    
    # Sets the current gesture data
    @qtc.Slot(result='QVariant')
    def set_gest_data(self):
        
        res = requests.post("http://127.0.0.1:5000/determine-gesture/").json()
        self.old_gest = self._gest
        self.setGest(res['gesture'])
        self.setX(res['x'])
        self.setY(res['y'])
        self.setZ(res['z'])
        self.x = res['x']
        self.y = res['y']
        self.z = res['z']
        
        return QPointF(self.x, self.y)
    
    @qtc.Slot(result='QVariant')
    def setMouse(self):
        # Set the cursor position
        self.cursor.setPos(self.x, self.y)
        
        # Case for mouse up after performing a mouse down
        if self._gest == 0 and (self.old_gest == 1 or self.old_gest == 2):
            if(self.old_gest == 2):
                pyautogui.keyUp("alt")

        # Case for mouse down and drag
        elif self._gest == 2 and self.old_gest != 2:
            pyautogui.keyDown("alt")
        
        '''
        # Case for clicking
        elif self._gest == 3 and self.old_gest != 3:
        #    pyautogui.click()

        # Case mouse down
        elif self._gest == 1 and self.old_gest != 1:
        #   pyautogui.mouseDown()
        '''
        return [self.gest, self.old_gest]

    @qtc.Property(int, notify=gestChanged)
    def gest(self):
        return self._gest
    
    @gest.setter
    def setGest(self, newGest):
        if self._gest == newGest:
            return
        self._gest = newGest
        self.gestChanged.emit()
    
    @qtc.Property(float, notify=handXChanged)
    def handX(self):
        return self.x

    @handX.setter
    def setX(self, newX):
        # Do not emit a changed signal if the icon is the same
        if self.x == newX:
            return
        self.x = newX
        self.handXChanged.emit()
    
    @qtc.Property(float, notify=handYChanged)
    def handY(self):
        return self.y
    
    @handY.setter
    def setY(self, newY):
        # Do not emit a changed signal if the icon is the same
        if self.y == newY:
            return
        self.y = newY
        self.handYChanged.emit()
    
    @Slot(QQuickItem, Qt.MouseButton)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier, QPointF)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier, QPointF, int)
    def mouseClick(self, item, button, modifier=Qt.NoModifier, pos=QPointF(), delay=-1):
        self.mousePress(item, button, modifier, pos, delay)
        self.mouseRelease(item, button, modifier, pos, 2 * delay)

    @Slot(QQuickItem, Qt.MouseButton)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier, QPointF)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier, QPointF, int)
    def mousePress(self, item, button, modifier=Qt.NoModifier, pos=QPointF(), delay=-1):
        self._send_mouse_events(
            QEvent.MouseButtonPress, item, button, modifier, pos, delay
        )

    @Slot(QQuickItem, Qt.MouseButton)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier, QPointF)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier, QPointF, int)
    def mouseRelease(
        self, item, button, modifier=Qt.NoModifier, pos=QPointF(), delay=-1
    ):
        self._send_mouse_events(
            QEvent.MouseButtonRelease, item, button, modifier, pos, delay
        )

    @Slot(QQuickItem, Qt.MouseButton)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier, QPointF)
    @Slot(QQuickItem, Qt.MouseButton, Qt.KeyboardModifier, QPointF, int)
    def mouseDClick(
        self, item, button, modifier=Qt.NoModifier, pos=QPointF(), delay=-1
    ):
        self.mousePress(item, button, modifier, pos, delay)
        self.mouseRelease(item, button, modifier, pos, 2 * delay)
        self.mousePress(item, button, modifier, pos, 3 * delay)
        self._send_mouse_events(
            QEvent.MouseButtonDblClick, item, button, pos, 4 * delay
        )
        self.mouseRelease(item, button, modifier, pos, 2 * delay)

    def _send_mouse_events(self, type_, item, button, modifier, pos, delay):
        window = item.window()
        if pos.isNull():
            pos = item.boundingRect().center()
        sp = item.mapToScene(pos).toPoint()
        event = QMouseEvent(
            type_, pos, window.mapToGlobal(sp), button, button, modifier
        )
        if delay < 0:
            delay = 0

        def on_timeout():
            QCoreApplication.instance().notify(window, event)

        QTimer.singleShot(delay, on_timeout)
    

if __name__ == '__main__':

    # Initializes the app, engine, and classes 
    app = QGuiApplication(sys.argv)
    engine = qtm.QQmlApplicationEngine()
    root_context = engine.rootContext()
    
    handgest = HandGest()

    root_context.setContextProperty('handgest', handgest)

    engine.load(qtc.QUrl.fromLocalFile('../test.qml'))
    sys.exit(app.exec_())
