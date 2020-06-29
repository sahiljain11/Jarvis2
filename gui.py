import math
import numpy as np
import sys
import os
from Spotify_Module import SpotipyModule
from GmailModule import GmailModule
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2 import QtCore as qtc
from PySide2 import QtQuick as qtq
from PySide2 import QtQml as qtm


class Spotify(qtc.QObject):

    #Signals
    iconUrlChanged = qtc.Signal()
    
    def __init__(self, parent=None):
        super(Spotify, self).__init__(parent)
        self.m_iconUrl = qtc.QUrl()

    @qtc.Slot(str)
    def test(self, word):
        print(word)

    @qtc.Property(qtc.QUrl, notify=iconUrlChanged)
    def iconUrl(self):
        return self.m_iconUrl

    @iconUrl.setter
    def setIcon(self, url):
        # Do not emit a changed signal if the icon is the same
        if self.m_iconUrl == url:
            return
        self.m_iconUrl = url
        self.iconUrlChanged.emit()
    

if __name__ == '__main__':

    # Initializes the app, engine, and classes 
    app = qtg.QGuiApplication(sys.argv)
    engine = qtm.QQmlApplicationEngine()
    spotify = SpotipyModule(os.environ.get('USER'), os.environ.get('CLIENT_ID'), os.environ.get('CLIENT_SECRET'), os.environ.get('REDIRECT_URI'))
    gmail = GmailModule()
    # Load classes and main qml file 
    engine.rootContext().setContextProperty("spotify", spotify)
    engine.rootContext().setContextProperty("gmail",gmail)
    engine.load(qtc.QUrl.fromLocalFile('gmail.qml'))


    # Exit if we have no classes
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec_())