from __future__ import print_function
import pickle
import os.path
import base64
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
import os
from PySide2 import QtCore as qtc
from PySide2 import QtGui as qtg
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import apiclient.errors
import re

''' 
Gmail Module Functionality Description

-allows user to log in
-get labels from a users account
-gets a list of most recent messages (in terms of id)
-get a information from message (input is message id)
-list message ids based on label search
-list message ids based on a query search
-create messages with or without attachments
-send these messages to another email 
'''


class GmailModule(qtc.QObject):

    def __init__(self):
        super(GmailModule, self).__init__()
        self.scopes = ['https://mail.google.com/']
        self.service = self.use_token_pickle_to_get_service()
        self.message_ids = self.get_list_of_users_message_ids()

    '''
    Accesses a file to gain saved credentials
    if no file exists the file is generated and user is asked to put in creds
    '''
    def use_token_pickle_to_get_service(self):
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
                    'credentials.json', self.scopes)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        service = build('gmail', 'v1', credentials=creds)

        return service
    '''
    gets a list of labels from email account
    '''
    @qtc.Slot()
    def get_labels(self):
        results = self.service.users().labels().list(userId='me').execute()

        labels = results.get('labels', [])
        if not labels:
            return
        else:
            label_list = [None] * len(labels)
            label_list_index = 0
            for label in labels:
                label_list[label_list_index] = (label['name'])
                label_list_index += 1
            return label_list

    def get_list_of_users_message_ids(self):

        emails = self.service.users().messages().list(userId='me').execute()
        list_of_msg_ids = [None] * len(emails['messages'])
        index = 0
        for email in range(0, len(emails['messages'])):
            list_of_msg_ids[index] = (emails['messages'][email])
            index += 1
        return list_of_msg_ids

    @qtc.Slot(int, result=str)
    def get_message_id(self, i):
        return self.message_ids[i]['id']

    @qtc.Slot(str,result=str)
    def GetMessage(self, msg_id):
        try:
            message = self.service.users().messages().get(userId='me', id=msg_id, format='full').execute()


            try:

                    raw_msg = message['payload']['parts'][0]['body']['data']
                    msg_str = base64.urlsafe_b64decode(raw_msg.encode('ASCII'))

            #print(msg_str)
            # print ('Message snippet: %s' % message['payload']['body']['data'])

            except KeyError:
                if 'parts' in message['payload'] and 'data' not in message['payload']['parts'][0]['body']:

                    raw_msg = ((message['payload']['parts'][0]['parts'][1]['body']['data']))
                    msg_str = base64.urlsafe_b64decode(raw_msg.encode('ASCII'))
                elif 'parts' in message['payload'] and 'data' in (message['payload']['parts'][0]['body']):
                    raw_msg=((message['payload']['parts'][0]['body']['data']))
                    msg_str = base64.urlsafe_b64decode(raw_msg.encode('ASCII'))
                else:
                    raw_msg = message['payload']['body']['data']
                    msg_str = base64.urlsafe_b64decode(raw_msg.encode('ASCII'))

                return msg_str

            return msg_str
        except apiclient.errors.HttpError:

            return

    @qtc.Slot(str,result=str)
    def GetSender(self,msg_id):
        try:
            message = self.service.users().messages().get(userId='me', id=msg_id, format='metadata').execute()
            
            for i in range(0,len((message['payload']['headers']))):
                if (message['payload']['headers'][i]['name']) == 'From':
                    sender = ((message['payload']['headers'][i]['value']))
                    return sender
            return
        except apiclient.errors.HttpError:

            return

    @qtc.Slot(str,result=str)
    def GetSubjectTitle(self, msg_id):
        try:
            message = self.service.users().messages().get(userId='me', id=msg_id, format='metadata').execute()
            for i in range(0,len((message['payload']['headers']))):
                if (message['payload']['headers'][i]['name']) == 'Subject':
                   subject = ((message['payload']['headers'][i]['value']))
                   return subject
        except apiclient.errors.HttpError:
            return

    @qtc.Slot(str,result=str)
    def GetSnippet(self,msg_id):
        try:
            message = self.service.users().messages().get(userId='me', id=msg_id, format='metadata').execute()
            snippet = message['snippet']
            return snippet
        except apiclient.errors.HttpError:
            return
    @qtc.Slot()
    def ListMessagesMatchingQuery(self, user_id,query=''):

        try:
            response = self.service.users().messages().list(userId='me',
                                                            q=query).execute()
            messages = []
            if 'messages' in response:
                messages.extend(response['messages'])

            while 'nextPageToken' in response:
                page_token = response['nextPageToken']
                response = self.service.users().messages().list(userId=user_id, q=query,
                                                                pageToken=page_token).execute()
                messages.extend(response['messages'])

            return messages
        except apiclient.errors.HttpError as error:
            return

    @qtc.Slot()
    def ListMessagesWithLabels(self, user_id, label_ids=''):
        """
        List all Messages of the user's mailbox with label_ids applied.

        Args:
          service: Authorized Gmail API service instance.
          user_id: User's email address. The special value "me"
          can be used to indicate the authenticated user.
          label_ids: Only return Messages with these labelIds applied.

        Returns:
          List of Messages that have all required Labels applied. Note that the
          returned list contains Message IDs, you must use get with the
          appropriate id to get the details of a Message.
        """

        try:
            response = self.service.users().messages().list(userId=user_id,
                                                            labelIds=label_ids).execute()
            messagess = []
            if 'messages' in response:
                messagess.extend(response['messages'])

            while 'nextPageToken' in response:
                page_token = response['nextPageToken']
                response = self.service.users().messages().list(userId=user_id,
                                                           labelIds=label_ids,
                                                           pageToken=page_token).execute()
                messagess.extend(response['messages'])

            return messagess
        except apiclient.errors.HttpError as error:
            print('An error occurred: %s' % error)
    @qtc.Slot(str)
    def get_messages_from_query(self, query=''):
        try:
            response = self.service.users().messages().list(userId='me',
                                                            q=query).execute()
            messages = []
            if 'messages' in response:
                messages.extend(response['messages'])

            while 'nextPageToken' in response:
                page_token = response['nextPageToken']
                response = self.service.users().messages().list(userId='me', q=query,
                                                                pageToken=page_token).execute()
                messages.extend(response['messages'])

            return messages
        except apiclient.errors.HttpError as error:

            return

    def list_message_ids(self):
        emails = self.service.users().messages().list(userId='me').execute()

        return emails


    def create_email(self, sender, to, subject, message_text, file_dir, filename):
        message = MIMEMultipart()
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject

        msg = MIMEText(message_text)
        message.attach(msg)

        path = os.path.join(file_dir, filename)
        content_type, encoding = mimetypes.guess_type(path)

        if content_type is None or encoding is not None:
            content_type = 'application/octet-stream'
        main_type, sub_type = content_type.split('/', 1)
        if main_type == 'text':
            fp = open(path, 'rb')
            msg = MIMEText(fp.read(), _subtype=sub_type)
            fp.close()
        elif main_type == 'image':
            fp = open(path, 'rb')
            msg = MIMEImage(fp.read(), _subtype=sub_type)
            fp.close()
        elif main_type == 'audio':
            fp = open(path, 'rb')
            msg = MIMEAudio(fp.read(), _subtype=sub_type)
            fp.close()
        else:
            fp = open(path, 'rb')
            msg = MIMEBase(main_type, sub_type)
            msg.set_payload(fp.read())
            fp.close()

        msg.add_header('Content-Disposition', 'attachment', filename=filename)
        message.attach(msg)

        return {'raw': base64.urlsafe_b64encode(message.as_string())}




    # create email- use message returned as input for send_message

    @qtc.Slot(str,str,str,str,result=dict)
    def create_basic_email(self, sender, to, subject, message_text):
        message = MIMEText(message_text)
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject

        return {'raw': base64.urlsafe_b64encode((message.as_bytes())).decode()}



    @qtc.Slot(str,str,str,str)
    def send_message(self, sender,to,subject,message_text):
        premessage= self.create_basic_email(sender,to,subject,message_text)
        try:
            message = (self.service.users().messages().send(userId=sender, body=premessage)
                       .execute())
            print(message)
            return message
        except apiclient.errors.HttpError as error:
            print('error')
            return

'''
# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://mail.google.com/']
# https://www.googleapis.com/auth/gmail.compose'
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)


def labels():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
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

    service = build('gmail', 'v1', credentials=creds)

    # Call the Gmail API
    results = service.users().labels().list(userId='me').execute()
    print(results['labels'][1]['id'])
    labels = results.get('labels', [])

def messages():
    """Lists Users messages
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

    service = build('gmail', 'v1', credentials=creds)

    emails = service.users().messages().list(userId='me').execute()
    for email in range(0, len(emails['messages'])):
        print(emails['messages'][email])


from apiclient import errors


def ListMessagesMatchingQuery(service, user_id, query=''):
    """List all Messages of the user's mailbox matching the query.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    query: String used to filter messages returned.
    Eg.- 'from:user@some_domain.com' for Messages from a particular sender.

  Returns:
    List of Messages that match the criteria of the query. Note that the
    returned list contains Message IDs, you must use get with the
    appropriate ID to get the details of a Message.
  """
    try:
        response = service.users().messages().list(userId=user_id,
                                                   q=query).execute()
        messages = []
        if 'messages' in response:
            messages.extend(response['messages'])

        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId=user_id, q=query,
                                                       pageToken=page_token).execute()
            messages.extend(response['messages'])
        print(messages)
        return messages
    except errors.HttpError as error:
        print('An error occurred: %s' % error)


def ListMessagesWithLabels(service, user_id, label_ids='SENT'):
    """List all Messages of the user's mailbox with label_ids applied.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    label_ids: Only return Messages with these labelIds applied.

  Returns:
    List of Messages that have all required Labels applied. Note that the
    returned list contains Message IDs, you must use get with the
    appropriate id to get the details of a Message.
  """
    try:
        response = service.users().messages().list(userId=user_id,
                                                   labelIds=label_ids).execute()
        print(response)
        messagess = []
        if 'messages' in response:
            messagess.extend(response['messages'])

        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId=user_id,
                                                       labelIds=label_ids,
                                                       pageToken=page_token).execute()
            messagess.extend(response['messages'])
        print(messagess)
        return messagess
    except errors.HttpError as error:
        print('An error occurred: %s' % error)


from apiclient import errors


def GetMessage(service, user_id, msg_id):
    """Get a Message with given ID.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    msg_id: The ID of the Message required.

  Returns:
    A Message.
  """
    try:
        message = service.users().messages().get(userId=user_id, id=msg_id, format='full').execute()
        # print(message['snippet'])
        raw_msg = message['payload']['parts'][0]['body']['data']
        msg_str = base64.urlsafe_b64decode(raw_msg.encode('ASCII'))

        print('MESSAGE STRING', msg_str)
        # print ('Message snippet: %s' % message['payload']['body']['data'])

        return message
    except errors.HttpError as error:
        print('An error occurred: %s' % error)


def SendMessage(service, user_id, message):
    """Send an email message.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    message: Message to be sent.

  Returns:
    Sent Message.
  """
    try:
        message = (service.users().messages().send(userId=user_id, body=message)
                   .execute())
        print('Message Id: %s' % message['id'])
        return message
    except errors.HttpError as error:
        print('An error occurred: %s' % error)


def CreateMessage(sender, to, subject, message_text):
    """Create a message for an email.

  Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.

  Returns:
    An object containing a base64url encoded email object.
  """
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    return {'raw': base64.urlsafe_b64encode((message.as_bytes())).decode()}


def CreateMessageWithAttachment(sender, to, subject, message_text, file_dir,
                                filename):
    """Create a message for an email.

  Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.
    file_dir: The directory containing the file to be attached.
    filename: The name of the file to be attached.

  Returns:
    An object containing a base64url encoded email object.
  """
    message = MIMEMultipart()
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject

    msg = MIMEText(message_text)
    message.attach(msg)

    path = os.path.join(file_dir, filename)
    content_type, encoding = mimetypes.guess_type(path)

    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
    main_type, sub_type = content_type.split('/', 1)
    if main_type == 'text':
        fp = open(path, 'rb')
        msg = MIMEText(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'image':
        fp = open(path, 'rb')
        msg = MIMEImage(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'audio':
        fp = open(path, 'rb')
        msg = MIMEAudio(fp.read(), _subtype=sub_type)
        fp.close()
    else:
        fp = open(path, 'rb')
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(fp.read())
        fp.close()

    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)

    return {'raw': base64.urlsafe_b64encode(message.as_string())}


def send_message(service, user_id, message):
    """Send an email message.

  Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    message: Message to be sent.

  Returns:
    Sent Message.
  """
    try:
        message = (service.users().messages().send(userId=user_id, body=message)
                   .execute())
        print('Message Id: %s' % message['id'])
        return message
    except errors.HttpError as error:
        print('An error occurred: %s' % error)
'''

if __name__ == '__main__':
    gmail = GmailModule()
    (gmail.get_labels())
    gmail.get_list_of_users_message_ids()

    array = gmail.get_list_of_users_message_ids()
    print(len(array))
    for i in range(0,100):
        print(i)
        (gmail.GetMessage(array[i]['id']))
        print(gmail.GetSender(array[i]['id']))
        print(gmail.GetSubjectTitle(array[i]['id']))
        print(gmail.GetSnippet(array[i]['id']))

    # ListMessagesMatchingQuery(service,'me','it\'s time to refresh')

