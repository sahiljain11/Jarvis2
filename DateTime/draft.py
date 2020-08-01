from PySide2.QtCore import QUrl
from PySide2 import QtGui
from PySide2 import QtQml


def main():
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

    app = QtGui.QGuiApplication(sys.argv)

    engine = QtQml.QQmlApplicationEngine()

    filename = os.path.join(CURRENT_DIR, "draft.qml")
    engine.load(QUrl.fromLocalFile(filename))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

# import sys
# from PyQt4 import QtGui, QtCore
#
# from time import strftime
#
#
# class Main(QtGui.QMainWindow):
#
#     def __init__(self):
#         QtGui.QMainWindow.__init__(self)
#         self.initUI()
#
#     def initUI(self):
#         self.timer = QtCore.QTimer(self)
#         self.timer.timeout.connect(self.Time)
#         self.timer.start(1000)
#
#         self.lcd = QtGui.QLCDNumber(self)
#         self.lcd.display(strftime("%H" + ":" + "%M"))
#
#         self.setCentralWidget(self.lcd)
#
#         # ---------Window settings --------------------------------
#
#         self.setGeometry(300, 300, 250, 100)
#         self.setWindowTitle("Clock")
#
#     # -------- Slots ------------------------------------------
#
#     def Time(self):
#         self.lcd.display(strftime("%H" + ":" + "%M"))
#
#
# def main():
#     app = QtGui.QApplication(sys.argv)
#     main = Main()
#     main.show()
#
#     sys.exit(app.exec_())
#
#
# if __name__ == "__main__":
#     main()


#
# from PySide2.QtWidgets import QApplication, QWidget, QLCDNumber
# from PySide2.QtCore import QTime, QTimer, SIGNAL
# import sys
# from PySide2.QtGui import QIcon
#
#
# class DigitalClock(QLCDNumber):
#     def __init__(self, parent=None):
#         super(DigitalClock, self).__init__(parent)
#
#         self.setSegmentStyle(QLCDNumber.Filled)
#
#         timer = QTimer(self)
#         self.connect(timer, SIGNAL('timeout()'), self.showTime)
#         timer.start(1000)
#         self.showTime()
#         self.setWindowTitle("Digital Clock")
#         self.resize(300, 200)
#
#         self.setIcon()
#
#     def setIcon(self):
#         appIcon = QIcon("icon.png")
#         self.setWindowIcon(appIcon)
#
#     def showTime(self):
#         time = QTime.currentTime()
#         text = time.toString('hh:mm')
#         if (time.second() % 2) == 0:
#             text = text[:2] + ' ' + text[3:]
#
#         self.display(text)
#
#
# myapp = QApplication(sys.argv)
# window = DigitalClock()
# window.show()
# myapp.exec_()
# sys.exit()