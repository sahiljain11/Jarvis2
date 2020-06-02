import numpy as np
import sys
import random
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from PyQt5 import QtNetwork as qtn

class Server(qtw.QWidget):

    # Creates a server that sends data
    def __init__(self):
        super(Server, self).__init__()
        self.server = qtn.QTcpServer()
        self.status = qtw.QLabel()
        self.address = 1234
        self.port = 80
        # Close the server if we cannot start it
        if not self.server.listen():
            qtw.QMessageBox.critical(self, "Fortune Server", "Unable to start the server: %s." % self.server.errorString())
            self.close()
            return
        # Create a widget with status in the middle
        self.status.setText("The server is running at address %s on port %d.\nRun the Fortune Client example now." % (self.server.serverAddress().toString(), self.server.serverPort()))
        mainLayout = qtw.QVBoxLayout()
        mainLayout.addWidget(self.status)
        self.setLayout(mainLayout)
        # Send data to the client if we have a new connection
        self.server.newConnection.connect(self.send_data)
        # Set the title of the window
        self.setWindowTitle("Server")

        self.show()

    # Implements sending of data
    def send_data(self):
        block = qtc.QByteArray()
        out = qtc.QDataStream(block, qtc.QIODevice.WriteOnly)
        out.setVersion(qtc.QDataStream.Qt_5_0)
        # Write the string to out
        out.writeUInt16(0)
        out.writeString("Welcome to the server")
        out.device().seek(0)
        out.writeUInt16(block.size() - 2)
        # Delete the connection if they disconnect
        client_connection = self.server.nextPendingConnection()
        client_connection.disconnected.connect(client_connection.deleteLater)
        # Write to the client
        client_connection.write(block)
        client_connection.disconnectFromHost()

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    server = Server()
    random.seed(None)
    sys.exit(app.exec_())
