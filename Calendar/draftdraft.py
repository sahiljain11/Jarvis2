from PySide2.QtCore import QUrl
from PySide2 import QtGui
from PySide2 import QtQml

def main():
    import os
    import sys
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    app = QtGui.QGuiApplication(sys.argv)
    engine = QtQml.QQmlApplicationEngine()
    filename = os.path.join(CURRENT_DIR, "draftdraft.qml")
    engine.load(QUrl.fromLocalFile(filename))
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()