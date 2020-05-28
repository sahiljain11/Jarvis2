import numpy as np
import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc

class MainWindow(qtw.QWidget):
    def __init__(self):
        #Call super class constructor
        super(MainWindow, self).__init__()
        self.show()
        #Sets the layout type
        self.layout = qtw.QVBoxLayout()
        self.setLayout(self.layout)
        #Spotify Button Code
        self.button = qtw.QPushButton("Push Me", self, checkable=True, shortcut=qtg.QKeySequence('+'))
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

        else:
            print("Released button")

    #PLACE CODE FOR SLIDER LOGIC IN HERE
    #Sets the volume variable to the slider position
    #Currently the max slider value is 100 and the min is 0    
    def slider_state(self):
        volume = self.slider.value()
        print("Volume: ", volume)

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec_())