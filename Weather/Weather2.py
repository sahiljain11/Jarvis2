from werkzeug.utils import cached_property
import json
import requests

from PySide2.QtCore import Property, Signal, Slot, QObject, QUrl, QUrlQuery, Qt, QDate, QTime, QDateTime
from PySide2 import QtNetwork
from PySide2 import QtGui
from PySide2 import QtQml

import logging
from pytz import timezone
import datetime

logging.basicConfig(level=logging.DEBUG)



class WeatherWrapper(QObject):
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

    dataChanged = Signal()


    def __init__(self, api_key="", parent: QObject = None) -> None:
        super().__init__(parent)
        self._data = dict()
        self._has_error = False
        self._api_key = api_key

    @cached_property
    def manager(self) -> QtNetwork.QNetworkAccessManager:
        return QtNetwork.QNetworkAccessManager(self)

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, key):
        self._api_key = "60e3e2927b60361a8a0fcae0f2042eab"

    @Property("QVariantMap", notify=dataChanged)
    def data(self) -> dict:
        return self._data

    @Slot(result=bool)
    def hasError(self):
        return self._has_error

    @Slot(str)
    def update_by_city(self, city: str) -> None:

        url = QUrl(WeatherWrapper.BASE_URL)
        query = QUrlQuery()
        query.addQueryItem("q", city)
        query.addQueryItem("appid", self.api_key)
        url.setQuery(query)

        request = QtNetwork.QNetworkRequest(url)
        reply: QtNetwork.QNetworkReply = self.manager.get(request)
        reply.finished.connect(self._handle_reply)

    def _handle_reply(self) -> None:
        has_error = False
        reply: QtNetwork.QNetworkReply = self.sender()
        if reply.error() == QtNetwork.QNetworkReply.NoError:
            data = reply.readAll().data()
            logging.debug(f"data: {data}")
            d = json.loads(data)
            code = d["cod"]
            if code != 404:
                del d["cod"]
                self._data = d
            else:
                #self._data = dict()
                has_error = True
                logging.debug("error: {code}")
                self.data = "AHHH"
        else:
            #self.data = dict()
            has_error = True
            self.data = "AHHH"
            logging.debug(f"error: {reply.errorString()}")
        self.dataChanged.emit()
        self._has_error = has_error
        reply.deleteLater()


def main():
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

    app = QtGui.QGuiApplication(sys.argv)

    API_KEY = "API_HERE"

    weather = WeatherWrapper()
    weather.api_key = API_KEY

    engine = QtQml.QQmlApplicationEngine()
    engine.rootContext().setContextProperty("weather", weather)

    filename = os.path.join(CURRENT_DIR, "Weatherdraft.qml")
    engine.load(QUrl.fromLocalFile(filename))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()