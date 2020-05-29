import numpy as np
import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import spotipy as spy
import spotipy.util as util
from os import environ


# allows API to access user stuffs
scope = 'user-library-read user-read-playback-state streaming user-modify-playback-state'

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
        #Spotify Button Code
        self.button = qtw.QPushButton("Push Me", self, checkable=True, shortcut=qtg.QKeySequence('='))
        self.button.clicked.connect(self.but_state)
        self.layout.addWidget(self.button)
        #Spotify Slider
        self.slider = qtw.QSlider(minimum=0, maximum =100,orientation=qtc.Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_state)
        self.layout.addWidget(self.slider)
    
    # PLACE CODE FOR BUTTON LOGIC IN HERE
    def but_state(self):
        if(self.button.isChecked()):
            print("You pressed the button")

            sp = spy.Spotify(auth=token)
            results = sp.current_user_saved_tracks()

            # need device id and permissions
            sp.start_playback()

        else:
            print("Released button")
            sp = spy.Spotify(auth=token)
            sp.pause_playback()

    #PLACE CODE FOR SLIDER LOGIC IN HERE
    #Sets the volume variable to the slider position
    #Currently the max slider value is 100 and the min is 0    
    def slider_state(self):
        sp = spy.Spotify(auth=token)
        volume_percent = self.slider.value()
        sp.volume(volume_percent)
        print("Volume: ", volume_percent)

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()

    sys.exit(app.exec_())