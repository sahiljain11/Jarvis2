from pytz import timezone
from datetime import datetime
from werkzeug.utils import cached_property
import json
import requests
from PySide2.QtCore import Property, Signal, Slot, QObject, QUrl, QUrlQuery, Qt, QDate, QTime, QDateTime
from PySide2 import QtNetwork
from PySide2 import QtGui
from PySide2 import QtQml

import logging

logging.basicConfig(level=logging.DEBUG)




class WeatherDate(QObject):
    signalName = Signal()
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
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

    @Property("QVariantMap", notify=signalName)
    def data(self) -> dict:
        return self._data

    @Slot(result=bool)
    def hasError(self):
        return self._has_error

    @Slot(str)
    def time_by_city(self, city:str) -> None:
        url = QUrl(WeatherDate.BASE_URL)
        query = QUrlQuery()
        query.addQueryItem("q", city)
        query.addQueryItem("appid", self.api_key)
        url.setQuery(query)
        request = QtNetwork.QNetworkRequest(url)
        reply: QtNetwork.QNetworkReply = self.manager.get(request)
        reply.finished.connect(self.tim)

    def tim(self) -> None:
        has_error = False
        reply: QtNetwork.QNetworkReply = self.sender()
        if reply.error() == QtNetwork.QNetworkReply.NoError:
            data = reply.readAll().data()
            logging.debug(f"data: {data}")
            d = json.loads(data)
            code = d["cod"]
            if code != 404:
                del d["cod"]
                data = reply.readAll().data()
                logging.debug(f"data: {data}")
                d = json.loads(data)

                latitude = d['coord']['lat']
                longitude = d['coord']['lon']
                url = "http://api.geonames.org/timezoneJSON?formatted=true&lat={}&lng={}&username=demo".format(latitude, longitude)
                r = requests.get(url)
                timezonee = r.json()['timezoneID']
                ourcity = timezone(timezonee)
                timeoutput = datetime.now(ourcity)
                oof = timeoutput.strftime('%Y-%m-%d_%H-%M-%S')
                self._data = oof
            else:
                self._data = dict()
                has_error = True
                logging.debug("error: {code}")

        else:
            self.data = dict()
            has_error = True
            logging.debug(f"error: {reply.errorString()}")
        self.signalName.emit()
        self._has_error = has_error
        reply.deleteLater()


def main():
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

    app = QtGui.QGuiApplication(sys.argv)

    time = WeatherDate()

    engine = QtQml.QQmlApplicationEngine()
    engine.rootContext().setContextProperty("time", time)

    filename = os.path.join(CURRENT_DIR, "Weather.qml")
    engine.load(QUrl.fromLocalFile(filename))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


