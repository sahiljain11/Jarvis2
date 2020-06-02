import numpy as np
import sys
import random
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from PyQt5 import QtNetwork as qtn

class Client(qtw.QWidget):
    def __init__(self, parent=None):
        super(Client, self).__init__(parent)
        self.block_size = 0
        self.data = ''

        # Initialize request button and label      
        self.data_label = qtw.QLabel()

        self.button = qtw.QPushButton("Connect")
        self.button.setDefault(True)
        # self.button.setEnabled(False)

        self.socket = qtn.QTcpSocket(self)

        # Connect Signals
        self.button.clicked.connect(self.connect_to_server)
        self.socket.readyRead.connect(self.read_data)
        self.socket.error.connect(self.displayError)

        # Set layout
        self.layout = qtw.QVBoxLayout()
        self.layout.addWidget(self.data_label)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

        self.show()

    # Connect to the server
    def connect_to_server(self):
        # self.button.setEnabled(False)
        self.block_size = 0
        self.socket.abort()
        self.socket.connectToHost("0.0.0.0",41441)

    # Read the data from the server
    def read_data(self):
        recv_data = qtc.QDataStream(self.socket)
        recv_data.setVersion(qtc.QDataStream.Qt_5_0)

        if self.block_size == 0:
            if self.socket.bytesAvailable() < 2:
                return
            
        self.block_size = recv_data.readUInt16()

        if self.socket.bytesAvailable() < self.block_size:
            return

        nextData = recv_data.readString()

        if nextData == self.data:
            qtc.QTimer.singleShot(0, self.connect_to_server)
            return
        
        self.data = nextData
        self.data_label.setText(self.data)
        self.button.setEnabled(True)

    def displayError(self, socketError):
        if socketError == qtn.QAbstractSocket.RemoteHostClosedError:
            pass
        elif socketError == qtn.QAbstractSocket.HostNotFoundError:
            qtw.QMessageBox.information(self, "Fortune Client",
                    "The host was not found. Please check the host name and "
                    "port settings.")
        elif socketError == qtn.QAbstractSocket.ConnectionRefusedError:
            qtw.QMessageBox.information(self, "Fortune Client",
                    "The connection was refused by the peer. Make sure the "
                    "fortune server is running, and check that the host name "
                    "and port settings are correct.")
        else:
            qtw.QMessageBox.information(self, "Fortune Client",
                    "The following error occurred: %s." % self.socket.errorString())
        self.button.setEnabled(True)



    





if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    server = Client()
    random.seed(None)
    sys.exit(app.exec_())