import numpy as np
import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from PyQt5.QtCore import Qt
import spotipy as spy
import spotipy.util as util
from os import environ

#Notes: Widgets work independently- play/pause needs to be pressed down when pressing skip
#button or else GUI fails when play/pause button is pressed afterwards

#Note: https://pypi.org/project/qtwidgets/
# The widgets here look clean af and I am interesting in creating something like that- Matt


# allows API to access user stuffs
scope = 'user-library-read user-read-playback-state streaming' \
        ' user-modify-playback-state playlist-modify-private ' \
        'user-read-playback-position user-read-currently-playing'

#checks for number of arguments from script- arguments used in token auth
if len(sys.argv) > 1:
    username = sys.argv[1]
    print(sys.argv[0])
    print(username)
    print(sys.argv[2])
    print(sys.argv[3])

else:
    #exits script if no args are put in
    print("Usage: %s username" % (sys.argv[0],))
    sys.exit()
#creates token based on credentials
token = util.prompt_for_user_token(username,scope, environ.get('CLIENT_ID'),environ.get('CLIENT_SECRET'),environ.get('REDIRECT_URL'))



class MainWindow(qtw.QWidget):
    def __init__(self):
        #Call super class constructor
        super(MainWindow, self).__init__()
        self.show()
        #Sets the layout type
        self.layout = qtw.QVBoxLayout()
        self.setLayout(self.layout)
        #Spotify Play/Pause Code
        self.button = qtw.QPushButton("Play/Pause Me", self, checkable=True, shortcut=qtg.QKeySequence('='))
        self.button.clicked.connect(self.but_state)
        self.layout.addWidget(self.button)
        #Spotify Slider
        self.slider = qtw.QSlider(minimum=0, maximum =100, orientation=qtc.Qt.Horizontal)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.slider_state)
        self.layout.addWidget(self.slider)
        #Spotify Move Forward
        self.forward = qtw.QPushButton("Next Song", self, checkable=False)
        self.forward.clicked.connect(self.next_state)
        self.layout.addWidget(self.forward)
        #Spotify Move Backward
        self.backward = qtw.QPushButton("Previous Song", self, checkable=False)
        self.backward.clicked.connect(self.previous_state)
        self.layout.addWidget(self.backward)
        #Spotify Displays Song
        #self.song_display = qtw.QLabel(self.display_current_song())
        #self.layout.addWidget(self.song_display)
        #https: // www.youtube.com / watch?v = xewHDkCKVoQ
        #Hopefully can get song title to move like on spotify
        self.how_you_feel = qtw.QLabel(self)
        self.how_you_feel.setText('How are you feeling today?')
        self.how_you_feel.setAlignment(Qt.AlignHCenter)
        self.layout.addWidget(self.how_you_feel)

        self.search_bar = qtw.QLineEdit(self)
        self.search_bar.setFont(qtg.QFont("Sanserif",15))
        self.search_bar.setAlignment(Qt.AlignHCenter)
        self.search_bar.returnPressed.connect(self.find_music)
        self.layout.addWidget(self.search_bar)

        #Displays Current song title




    #creates a spotify object with token provided
    def set_token(self):
        if token:
            authorized_user = spy.Spotify(auth=token)
        return authorized_user

    # PLACE CODE FOR BUTTON LOGIC IN HERE
    def but_state(self):
        if self.button.isChecked():
            print("You pressed the button")

            results = self.set_token().current_user_saved_tracks()


            self.set_token().start_playback()

        else:
            print("Released button")

            self.set_token().pause_playback()
    #allows user to go to next song in queue
    def next_state(self):
        if not self.button.isChecked():
            self.button.setChecked(True)
            self.set_token().next_track()
        else:
            self.set_token().next_track()
            print("SKIP")

    #allows user to go to previous song in queue
    def previous_state(self):
        if not self.button.isChecked():
            self.button.setChecked(True)
            self.set_token().previous_track()
        else:
            self.set_token().previous_track()
            print("Back one")

    def display_current_song(self):
        self.set_token().currently_playing()
        print(self.set_token().current_playback())


    #PLACE CODE FOR SLIDER LOGIC IN HERE
    #Sets the volume variable to the slider position
    #Currently the max slider value is 100 and the min is 0    
    def slider_state(self):
        sp = spy.Spotify(auth=token)
        volume_percent = self.slider.value()

        if sp.current_user_playing_track() is not None:
            sp.volume(volume_percent)
            print("Volume: ", volume_percent)

    def find_music(self):
        #print(self.search_bar.text())
        string_inputted = self.search_bar.text()
        search_query =string_inputted.replace(" ", "%20")
        print(self.set_token().search(search_query,1,0,'track'))
        #b =self.set_token().search(search_query, 1, 0, 'track').get()
        #print(type(b))
        #print(self.set_token().search(search_query, 1, 0, 'track').values())
        a =(self.set_token().search(search_query, 1, 0, 'track').values())
        print(len(a))
        print(type(a))
        for p in a:
            print(p)
            print(type(p))
        #print(self.set_token().search(search_query, 1, 0, 'track').keys(self.set_token().search(search_query, 1, 0, 'track').keys()))
        #print(search_query)


        self.search_bar.setText('')
        #print(self.set_token().current_user_playing_track())
        #I want to access both individual songs and playlists/albums


#intializes gui- exit by closing GUI
if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()

    sys.exit(app.exec_())