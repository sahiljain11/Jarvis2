import math
import numpy as np
import sys
import os
import pyautogui
from Spotify.Spotify_Module import SpotipyModule
from Gmail.GmailModule import GmailModule
from Weather.Weather import WeatherWrapper
from Calendar.API_setup import CalendarProvider
from HandGest import HandGest
from CovidGraphing.Covid19Data import Stats
from PySide2.QtQuick import QQuickItem, QQuickView 
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
    app = qtw.QApplication(sys.argv)
    qtm.qmlRegisterType(CalendarProvider, "MyCalendar", 1, 0, "CalendarProvider")
    view = QQuickView()
    root_context = view.rootContext()
    #engine = qtm.QQmlApplicationEngine()
    #root_context = engine.rootContext()

    #Load Spotify
    spotify = SpotipyModule(os.environ.get('USER'), os.environ.get('CLIENT_ID'), os.environ.get('CLIENT_SECRET'), os.environ.get('REDIRECT_URI'),os.environ.get(('USERNAME')))
    root_context.setContextProperty("spotify", spotify)
    root_context.setContextProperty("searchList", spotify.search_list)
    
    #Load Simulator
    #mouse_emulator = MouseEmulator()
    #view.rootContext().setContextProperty("mouse_emulator", mouse_emulator)

    # Load Gmail
    #gmail = GmailModule()
    #root_context.setContextProperty("gmail", gmail)
    #root_context.setContextProperty("emailPreview", gmail.currentEmailList)

    # Load HandGest
    hand = HandGest()
    root_context.setContextProperty("hand", hand)

    # Load Weather
    weather = WeatherWrapper()
    weather.api_key = "API_HERE"
    root_context.setContextProperty("weather", weather)

    # Load Covid
    covid = Stats()
    root_context.setContextProperty("corona", covid)


    # Load the main gui qml
    #engine.load(qtc.QUrl.fromLocalFile('./gui.qml'))
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(CURRENT_DIR, "gui.qml")
    view.setSource(qtc.QUrl.fromLocalFile(filename))
    view.show()
    

    # Load specific widgets
    #engine.load(qtc.QUrl.fromLocalFile('./Gmail/Gmail.qml'))
    #engine.load(qtc.QUrl.fromLocalFile('./Spotify/Spotify.qml'))
    
    # Exit if we have no classes
    #if not engine.rootObjects():
     #   sys.exit(-1)
    sys.exit(app.exec_())