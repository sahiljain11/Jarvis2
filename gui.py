import numpy as np
import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc

# Creates the QApplication object to initialize state of gui
app = qtw.QApplication([])
window = qtw.QWidget(windowTitle='Jarvis 2.0')
window.show()
app.exec_()

