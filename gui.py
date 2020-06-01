import math
import numpy as np
import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc

class MainWindow(qtw.QWidget):
    def __init__(self):
        # Call super class constructor
        super(MainWindow, self).__init__()

        # Set title and dimensions of the window
        self.title = 'Jarvis 2.0'
        self.left = 10
        self.top = 10
        self.width = 500
        self.height = 500

        # Misc debugging variables
        #self.counter = 0

        # Click and drag functionality
        self.source = None    # Stores the position of the widget we are currently dragging
        self.setAcceptDrops(True)
        # Initialize gui and display it 
        self.initUI()
        self.show()

    # Initialize the gui
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.create_test_grid_layout(16, 4, 4)

    # Creates a square grid layout of dummy widget for gui testing 
    def create_test_grid_layout(self, numWidgets, num_rows, num_columns):
        # Make sure the grid can hold the widgets
        if (num_rows * num_columns < numWidgets):
            print("There are more widgets than the available grid space")
            return
        self.layout = qtw.QGridLayout()
        counter = 0;
        #Add numWidgets dummy widgets
        for i in range(0,num_rows):
            for j in range(0, num_columns):
                widget = DummyWidget(str(counter))
                widget.installEventFilter(self)
                self.layout.addWidget(widget, i, j)
                counter += 1
        # Set the grid as the layout
        self.setLayout(self.layout)

    # Event filter used to overwrite the widget's mouse events
    def event_filter(self, watched, event):
        if event.type() == QEvent.MouseButtonPress:
            self.mousePressEvent(event)
        elif event.type() == QEvent.MouseMove:
            self.mouseMoveEvent(event)
        elif event.type() == QEvent.MouseButtonReleased:
            self.mouseReleaseEvent(event)
        return super(MainWindow, self).eventFilter(watched, event)

    # Grabs the index of the widget and the given position
    def get_index(self, pos):
        for i in range(self.layout.count()):
            #Return the index of the widget if we find it
            if(self.layout.itemAt(i).geometry().contains(pos) and  i != self.source):
                return i

    # When the mouse is pressed set self.source to the position of the mouse
    def mousePressEvent(self, event):
        # Grab the index of the widget at the mouse pos when the left button is pressed
        if event.button() == qtc.Qt.LeftButton:
            self.source = self.get_index(event.windowPos().toPoint())
            print("Position: ",event.globalPos())
            print("Index: ", self.source)
        # Reset the target if we did not click on a widget
        else:
            self.source = None

    # Reset self.source on release
    def mouseReleaseEvent(self, event):
        self.source = None
    
    
    # Creates an image of the widget we are moving
    def mouseMoveEvent(self, event):
        # If the left button was pressed and we have a target, displat an image of the widget at mousePos
        if event.buttons() & qtc.Qt.LeftButton and self.source is not None:
            target_widget = self.layout.itemAt(self.source)
            drag = qtg.QDrag(target_widget.widget())
            widget_image = target_widget.widget().grab()
            # Set the data that we need to transfer over
            mimeData = qtc.QMimeData()
            mimeData.setImageData(widget_image)
            drag.setMimeData(mimeData)
            #Display an image of the widget as we drag it
            drag.setPixmap(widget_image)
            mousePos = event.globalPos()
            drag.setHotSpot(qtc.QPoint(drag.pixmap().width()/2, drag.pixmap().height()/2))
            drag.exec_()

    # Checks if a drag event warrants moving a widget
    def dragEnterEvent(self, event):
        # if we have image data, try to move a widget
        if event.mimeData().hasImage():
            event.accept()
        else:
            event.ignore()
    
    # te 
    def dropEvent(self, event):
        # Only perform the drop if the target is not the same as the source
        if not event.source().geometry().contains(event.pos()):
            drag_target = self.get_index(event.pos())
            # Do nothing our mouse is not above a widget
            if drag_target is None:
                return
            widget1, widget2 = max(self.source, drag_target), min(self.source, drag_target)
            pos1, pos2 = self.layout.getItemPosition(widget1), self.layout.getItemPosition(widget2)
            self.layout.addItem(self.layout.takeAt(widget1), *pos2)
            self.layout.addItem(self.layout.takeAt(widget2), *pos1)

# Dummy widget class with just box 
class DummyWidget(qtw.QLabel):
    #Constructor for dummy widget
    def __init__(self, text):
        super(DummyWidget, self).__init__()
        self.setStyleSheet('background-color: #229977; color: yellow;')
        self.setText(text)
        self.setAlignment(qtc.Qt.AlignCenter)
        
# Class that contains other widgets 
# More functionality to be added later 
class JarvisCon(qtw.QWidget):
    # Constructor for container 
    def __init__(self, widget):
        self.widget = widget
        self.setAcceptDrops()

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec_())