from __future__ import print_function
import datetime
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import apiclient.errors
import os.path
import os
from PySide2 import QtCore
from PySide2.QtCore import Property, Signal, Slot, QObject, QUrl, QUrlQuery
from PySide2 import QtGui
from PySide2 import QtQml
from PySide2 import QtNetwork



class CalendarModule(QtCore.QObject):
    
    def __init__(self):
        super(CalendarModule, self).__init__()
        self.scopes = ['https://www.googleapis.com/auth/calendar']
        self.service = self.use_token_pickle_to_get_service()


    def use_token_pickle_to_get_service(self):
        """Shows basic usage of the Google Calendar API.
        Prints the start and name of the next 10 events on the user's calendar.
        """
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        service = build('calendar', 'v3', credentials=creds)

        return service
        
    @Slot(result='QVariant')
    def getevents(self):
        # Call the Calendar API
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        #print('Getting the upcoming 10 events')
        events_result = self.service.events().list(calendarId='primary', timeMin=now,
                                              maxResults=10, singleEvents=True,
                                              orderBy='startTime').execute()
        events = events_result.get('items', [])
        hey = {}
        if not events:
            return 'No upcoming events found.'

        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            hey[start] = event['summary']
        return hey



def main():
    import os
    import sys

    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

    app = QtGui.QGuiApplication(sys.argv)

    engine = QtQml.QQmlApplicationEngine()
    filename = os.path.join(CURRENT_DIR, "calendar2.qml")

    cal = CalendarModule()
    engine.rootContext().setContextProperty("Cal", cal)
    engine.load(QUrl.fromLocalFile(filename))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())




if __name__ == '__main__':
    print(CalendarModule().getevents())
    main()





# SCOPES = ['https://www.googleapis.com/auth/calendar.events.readonly']
#
#
# def main():
#     """Shows basic usage of the Google Calendar API.
#     Prints the start and name of the next 10 events on the user's calendar.
#     """
#     creds = None
#     # The file token.pickle stores the user's access and refresh tokens, and is
#     # created automatically when the authorization flow completes for the first
#     # time.
#     if os.path.exists('token.pickle'):
#         with open('token.pickle', 'rb') as token:
#             creds = pickle.load(token)
#     # If there are no (valid) credentials available, let the user log in.
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 'credentials.json', SCOPES)
#             creds = flow.run_local_server(port=0)
#         # Save the credentials for the next run
#         with open('token.pickle', 'wb') as token:
#             pickle.dump(creds, token)
#
#     service = build('calendar', 'v3', credentials=creds)
#
#     # Call the Calendar API
#     now = datetime.datetime.utcnow().isoformat() + 'Z' # 'Z' indicates UTC time
#     print('Getting the upcoming 10 events')
#     events_result = service.events().list(calendarId='primary', timeMin=now,
#                                         maxResults=10, singleEvents=True,
#                                         orderBy='startTime').execute()
#     events = events_result.get('items', [])
#
#     if not events:
#         print('No upcoming events found.')
#     for event in events:
#         start = event['start'].get('dateTime', event['start'].get('date'))
#         print(start, event['summary'])
#
#
# if __name__ == '__main__':
#     main()