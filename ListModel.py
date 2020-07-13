
import sys
import os
import time
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2 import QtCore as qtc
from PySide2 import QtQuick as qtq
from PySide2 import QtQml as qtm
from SongWrapper import SongWrapper


class ListModel(qtc.QAbstractListModel):

    # Initializes a python list to contain the elements and sets the role
    # element_type must subclass QObject and must have a variable "roles"
    def __init__(self, element_type, parent=None):
        super(ListModel, self).__init__()
        self._list = []
        self.roles = element_type.roles
    
    def roleNames(self):
        return self.roles
    
    # Apppends the item to the list
    def appendRow(self, item, parent=qtc.QModelIndex()):
        self.beginInsertRows(parent, len(self._list), len(self._list))
        self._list.append(item)
        self.endInsertRows()
    
    # Clears the list
    def clear(self, parent=qtc.QModelIndex()):
        self.beginResetModel()
        self._list.clear()
        self.endResetModel()

    # Returns the row count
    def rowCount(self, parent=qtc.QModelIndex()):
        return len(self._list)
    
    # Returns the data if it fits a given role
    def data(self, index, role):
        row = index.row()
        if index.isValid() and 0 <= row < self.rowCount():
            return self._list[index.row()].data(role)
    
    def __str__(self):
        return str(self._list)

    
