#imports
import logging
import os
import pickle
import sys
import threading
import datetime
import functools
from PySide2 import QtCore, QtGui, QtQml

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request




SCOPES = ["https://www.googleapis.com/auth/calendar"]
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


logging.basicConfig(level=logging.DEBUG)

def qdatetime_to_string(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, QtCore.QDateTime):
                x[k] = v.toString(QtCore.Qt.ISODate)
            else:
                qdatetime_to_string(v)
    elif isinstance(x, list):
        for i, e in enumerate(x):
            if isinstance(e, QtCore.QDateTime):
                x[i] = e.toString(QtCore.Qt.ISODate)
            else:
                qdatetime_to_string(v)

class Reply(QtCore.QObject):
    finished = QtCore.Signal()

    def __init__(self, func, args=(), kwargs=None, parent=None):
        super().__init__(parent)
        self._results = None
        self._is_finished = False
        self._error_str = ""
        threading.Thread(
            target=self._execute, args=(func, args, kwargs), daemon=True
        ).start()

    @property
    def results(self):
        return self._results

    @property
    def error_str(self):
        return self._error_str

    def is_finished(self):
        return self._is_finished

    def has_error(self):
        return bool(self._error_str)

    def _execute(self, func, args, kwargs):
        if kwargs is None:
            kwargs = {}
        try:
            self._results = func(*args, **kwargs)
        except Exception as e:
            self._error_str = str(e)
        self._is_finished = True
        self.finished.emit()


def convert_to_reply(func):
    def wrapper(*args, **kwargs):
        reply = Reply(func, args, kwargs)
        return reply

    return wrapper


class CalendarBackend(QtCore.QObject):
    eventsChanged = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._service = None

    @property
    def service(self):
        if self._service is None:
            reply = self._update_credentials()
            loop = QtCore.QEventLoop()
            reply.finished.connect(loop.quit)
            loop.exec_()
            if not reply.has_error():
                self._service = reply.results
            else:
                logging.debug(reply.error_str)
        return self._service

    def updateListEvents(self, kw):
        # threading creates separate flow of execution
        threading.Thread(target=self._update_list_events, args=(kw,)).start()

    #gathers next 10 events from google calendar and appends to qt_events
    def _update_list_events(self, kw):
        self._update_credentials()

        events_result = self.service.events().list(**kw).execute()
        events = events_result.get("items", [])

        qt_events = []
        if not events:
            logging.debug("No upcoming events found.")
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            end = event["end"].get("dateTime", event["end"].get("date"))
            logging.debug(f"From {start} - To {end}:  {event['summary']}")

            start_dt = QtCore.QDateTime.fromString(start, QtCore.Qt.ISODate)
            end_dt = QtCore.QDateTime.fromString(end, QtCore.Qt.ISODate)
            summary = event["summary"]

            e = {"start": start_dt, "end": end_dt, "summary": summary}
            qt_events.append(e)

        self.eventsChanged.emit(qt_events)

    #OAuth with Google API
    @convert_to_reply
    def _update_credentials(self):
        creds = None
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open("token.pickle", "wb") as token:
                pickle.dump(creds, token)
        return build("calendar", "v3", credentials=creds, cache_discovery=False)

    @convert_to_reply
    def insert(self, **kwargs):
        return self.service.events().insert(**kwargs).execute()

#connects events to calendar window
class CalendarProvider(QtCore.QObject):
    loaded = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._cache_events = []
        self._backend = CalendarBackend()
        self._backend.eventsChanged.connect(self._handle_events)

    @QtCore.Slot("QVariant")
    def createEvent(self, parameters):
        kw = parameters.toVariant()
        if isinstance(kw, dict):
            qdatetime_to_string(kw)
            reply = self._backend.insert(**kw)
            wrapper = functools.partial(self.handle_finished_create_event, reply)
            reply.finished.connect(wrapper)

    def handle_finished_create_event(self, reply):
        if reply.has_error():
            logging.debug(reply.error_str)
        else:
            event = reply.results
            link = event.get("htmlLink", "")
            logging.debug("Event created: %s" % (link,))
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(link))


    @QtCore.Slot("QVariant")
    def updateListEvents(self, parameters):
        d = dict()
        for k, v in parameters.toVariant().items():
            if isinstance(v, QtCore.QDateTime):
                v = v.toTimeSpec(QtCore.Qt.OffsetFromUTC).toString(
                    QtCore.Qt.ISODateWithMs
                )
            d[k] = v
        self._backend.updateListEvents(d)

    @QtCore.Slot(QtCore.QDate, result="QVariantList")
    def eventsForDate(self, date):
        events = []
        for event in self._cache_events:
            start = event["start"]
            if start.date() == date:
                events.append(event)
        return events

    @QtCore.Slot(list)
    def _handle_events(self, events):
        self._cache_events = events
        self.loaded.emit()
        logging.debug("Loaded")

#adding an event to calendar
class AddToCalendar(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = dict()
        #using instance var from another class
        self.A = CalendarBackend()
    @QtCore.Slot(str)
    def createevent(self, eventinfo: str, eventstart: str, eventend: str):
        starttime = str(datetime.datetime.strptime(eventstart, "%m/%d/%Y %H:%M:%S"))

        endtime = str(datetime.datetime.strptime(eventend, "%m/%d/%Y %H:%M:%S"))
        try:
            event = {
                'summary': eventinfo,
                'start': {
                    'dateTime':  (starttime[0:10]+"T"+starttime[11:]+"-06:00"),
                    'timeZone': 'America/Chicago',
                },
                'end': {
                    'dateTime': (endtime[0:10]+"T"+endtime[11:]+"-06:00"),
                    'timeZone': 'America/Chicago',
                }
            }
        except:
            pass

        event = self.A._service.events().insert(calendarId='primary', body=event).execute()
        print('Event created: %s' % (event.get('htmlLink')))



if __name__ == "__main__":
    app = QtGui.QGuiApplication(sys.argv)

    QtQml.qmlRegisterType(CalendarProvider, "MyCalendar", 1, 0, "CalendarProvider")
    cal2 = AddToCalendar()
    engine = QtQml.QQmlApplicationEngine()
    engine.rootContext().setContextProperty("cal2", cal2)
    filename = os.path.join(CURRENT_DIR, "CalendarDraft.qml")
    engine.load(QtCore.QUrl.fromLocalFile(filename))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())