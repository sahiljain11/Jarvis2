import sys
import time
import requests
from os import environ
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2 import QtCore as qtc
from PySide2 import QtQuick as qtq
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
        self._gest = 0
        self.cursor = qtg.QCursor()
    
    # Sets the current gesture data
    def set_gest_data(self):
        res = requests.post("http://127.0.0.1:5000/determine-gesture/", json=send).json()
        self.setGest(res['gesture'])
        '''self.setX(res['x'])
        self.setY(res['y'])
        self.setZ(res['z'])'''
        self.x = res['x']
        self.y = res['y']
        self.z = res['z']
        return
    
    @qtc.Slot()
    def setMouse(self):
        #self.set_gest_data()
        self.cursor.setPos(0, 0)

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
    
    @qtc.Property(float, notify=handZChanged)
    def handZ(self):
        return self.z

    @handZ.setter
    def setZ(self, newZ):
        # Do not emit a changed signal if the icon is the same
        if self.z == newZ:
            return
        self.z = newZ
        self.handZChanged.emit()

if __name__ == '__main__':

    # Initializes the app, engine, and classes 
    app = qtg.QGuiApplication(sys.argv)
    engine = qtm.QQmlApplicationEngine()
    root_context = engine.rootContext()
    
    handgest = HandGest()

    root_context.setContextProperty('handgest', handgest)

    engine.load(qtc.QUrl.fromLocalFile('../test.qml'))
    sys.exit(app.exec_())
