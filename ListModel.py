
import sys
import os
import time
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2 import QtCore as qtc
from PySide2 import QtQuick as qtq
from PySide2 import QtQml as qtm


# QtWrapper is a wrapper that wraps an object in a Qt object
class SongWrapper(qtc.QObject):

    objectChanged = qtc.Signal()

    # Initialize the wrapper
    def __init__(self, song, artist, image_url):
        super(QtWrapper, self).__init__()
        self._song = song
        self._artist = artist
        self._image_url = image_url
    
    # Returns the song name

    def song(self):
        return self._song

    def artist(self):
        return self._artist
    
    def image_url(self):
        return self._image_url
    
    # Returns the song 
    
    
    

    
