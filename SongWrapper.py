import sys
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2 import QtCore as qtc
from PySide2 import QtQuick as qtq
from PySide2 import QtQml as qtm

#QtWrapper is a wrapper that wraps an object in a Qt object
class SongWrapper(qtc.QObject):

    # Dictionary of roles for SongWrapper
    roles = {
        qtc.Qt.UserRole + 1: b'song',
        qtc.Qt.UserRole + 2: b'artist',
        qtc.Qt.UserRole + 3: b'image_url'
    }

    # Signals
    songChanged = qtc.Signal()
    artistChanged = qtc.Signal()
    image_urlChanged = qtc.Signal()

    # Initialize the wrapper
    def __init__(self, song, artist, image_url, parent=None):
        super(SongWrapper, self).__init__()
        self._data = {b'song': song, b'artist': artist, b'image_url': image_url}
    
    # Retrieves the given role of the SongWrapper (i.e. song, artist, or image_url)
    def data(self, key):
        return self._data[self.roles[key]]

    @qtc.Property(str, notify=songChanged)
    def song(self):
        return self._data[b'song']
    
    @qtc.Property(str, notify=artistChanged)
    def artist(self):
        return self._data[b'artist']
    
    @qtc.Property(qtc.QUrl, notify=image_urlChanged)
    def image_url(self):
        return self._data[b'image_url']

    def __str__(self):
        return "[" + str(self.song) + ", " + str(self.artist)  + "]" 
    
    def __repr__(self):
        return str(self)