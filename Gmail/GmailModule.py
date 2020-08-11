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
import time
import threading
from PySide2 import QtCore as qtc
from ListModel import ListModel
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import apiclient.errors


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
    loaded = qtc.Signal()
    
    def __init__(self):
        super(GmailModule, self).__init__()
        self.scopes = ['https://mail.google.com/']
        self.time_elapsed = 0
        self.currentEmailList = ListModel(GmailModule.EmailObject)
        self.threadMessages = ListModel(GmailModule.EmailObject)
        #print("Gmail finished")

    # Initializes the gmail widget in the qml file
    @qtc.Slot(int)
    def init_gmail_in_widget(self, parm):
        self.service = self.use_token_pickle_to_get_service()
        self.messages = self.service.users().messages()
        self.message_ids = self.get_list_of_users_message_ids()
        self.label_list = self.get_labels()
        threading.Thread(target=self.get_preview_message_info, args=(parm,)).start()
       
    '''
    Accesses a file to gain saved credentials
    if no file exists the file is generated and user is asked to put in creds
    '''
    def use_token_pickle_to_get_service(self):
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token_gmail.pickle'):
            with open('token_gmail.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials_gmail.json', self.scopes)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token_gmail.pickle', 'wb') as token:
                pickle.dump(creds, token)

        service = build('gmail', 'v1', credentials=creds)

        return service

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
        emails = self.messages.list(userId='me').execute()
        list_of_msg_ids = [None] * len(emails['messages'])
        index = 0
        for email in range(0, len(emails['messages'])):
            list_of_msg_ids[index] = (emails['messages'][email])
            index += 1
        return list_of_msg_ids

    @qtc.Slot(int, result=str)
    def get_message_id(self, i):
        return self.message_ids[i]['id']

    @qtc.Slot(int, result=str)
    def get_label(self,i):
        return self.label_list[i]

    # Takes an integer as an index and retrieves the message's metadata
    @qtc.Slot(int)
    def get_preview_message_info(self, num_messages):
        for index in range(0, num_messages):
            message = self.messages.get(userId='me', id=self.message_ids[index]['id'], format='metadata', metadataHeaders=["To", "From", "Subject", "References", "Message-ID"]).execute()
            self.currentEmailList.appendRow(GmailModule.EmailObject(sender=self.GetSender(message), snippet=self.GetSnippet(message), subject=self.GetSubjectTitle(message), threadid=self.GetThread(message), references=self.GetReferences(message), messageid=self.GetMessageID(message)))
        return 

    @qtc.Slot(str,result=str)
    def GetMessage(self, message):
        try:

            try:

                    raw_msg = message['payload']['parts'][0]['body']['data']
                    msg_str = base64.urlsafe_b64decode(raw_msg.encode('ASCII'))

            #print(msg_str)
            # print ('Message snippet: %s' % message['payload']['body']['data'])

            except KeyError:

                if 'parts' in message['payload'] and 'parts' in message['payload']['parts'][0]:
                    if 'data' in (message['payload']['parts'][0]['parts'][0]['body']):
                        raw_msg = (message['payload']['parts'][0]['parts'][0]['body']['data'])
                        msg_str = base64.urlsafe_b64decode(raw_msg.encode('ASCII'))
                    else:
                        raw_msg = (message['payload']['parts'][0]['parts'][0]['parts'][0]['body']['data'])
                        msg_str =  base64.urlsafe_b64decode(raw_msg.encode('ASCII'))

                elif 'parts' in message['payload'] and 'data' not in message['payload']['parts'][0]['body']:

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
    def GetSender(self,message):
        try:
            for i in range(0,len((message['payload']['headers']))):
                if (message['payload']['headers'][i]['name']) == 'From':
                    sender = ((message['payload']['headers'][i]['value']))
                    return sender
            return
        except apiclient.errors.HttpError:
            return

    @qtc.Slot(str,result=str)
    def GetSubjectTitle(self, message):
        try:
            for i in range(0,len((message['payload']['headers']))):
                if (message['payload']['headers'][i]['name']) == 'Subject':
                   subject = ((message['payload']['headers'][i]['value']))
                   return subject
        except apiclient.errors.HttpError:
            return

    @qtc.Slot(str,result=str)
    def GetSnippet(self,message):
        try:
            snippet = message['snippet']
            return snippet
        except apiclient.errors.HttpError:
            return
    
    @qtc.Slot(str, result=str)
    def GetThread(self, message):
        try: 
            return message['threadId']
        except apiclient.errors.HttpError:
            return
    
    @qtc.Slot(str, result=str)
    def GetReferences(self, message):
        try:
            for i in range(0,len((message['payload']['headers']))):
                if (message['payload']['headers'][i]['name']) == 'References':
                    references = ((message['payload']['headers'][i]['value']))
                    return references
        except apiclient.errors.HttpError:
            return
    
    @qtc.Slot(str, result=str)
    def GetMessageID(self, message):
        try:
            for i in range(0,len((message['payload']['headers']))):
                if (message['payload']['headers'][i]['name']) == 'Message-ID':
                    messageID = ((message['payload']['headers'][i]['value']))
                    return messageID
        except apiclient.errors.HttpError:
            return

    @qtc.Slot()
    def ListMessagesMatchingQuery(self, user_id,query=''):

        try:
            response = self.messages.list(userId='me',
                                                            q=query).execute()
            messages = []
            if 'messages' in response:
                messages.extend(response['messages'])

            while 'nextPageToken' in response:
                page_token = response['nextPageToken']
                response = self.messages.list(userId=user_id, q=query,
                                                                pageToken=page_token).execute()
                messages.extend(response['messages'])

            return messages
        except apiclient.errors.HttpError as error:
            return

    @qtc.Slot(str)
    def get_messages_with_labels(self, label_ids=''):

        try:
            response = self.messages.list(userId='me',
                                                            labelIds=label_ids).execute()
            messages = []
            if 'messages' in response:
                messages.extend(response['messages'])

            while 'nextPageToken' in response and len(messages) <= 50:
                page_token = response['nextPageToken']
                response = self.messages.list(userId='me',
                                                           labelIds=label_ids,
                                                           pageToken=page_token).execute()
                messages.extend(response['messages'])
            
            # Add the response to the list of emails
            self.currentEmailList.clear()
            i = 0
            while i < 20 and i < len(messages):
                message = self.messages.get(userId='me', id=messages[i]['id'], format='metadata', metadataHeaders=["To", "From", "Subject", "References", "Message-ID"]).execute()
                self.currentEmailList.appendRow(GmailModule.EmailObject(sender=self.GetSender(message), snippet=self.GetSnippet(message), subject=self.GetSubjectTitle(message), threadid=self.GetThread(message), references=self.GetReferences(message), messageid=self.GetMessageID(message)))
                i += 1
            return messages
        except apiclient.errors.HttpError as error:
            return

    @qtc.Slot(str)
    def get_messages_from_query(self, query=''):
        try:
            response = self.messages.list(userId='me', maxResults=50,
                                                            q=query).execute()
            messages = []
            if 'messages' in response:
                messages.extend(response['messages'])

            while 'nextPageToken' in response:
                page_token = response['nextPageToken']
                response = self.messages.list(userId='me', q=query,
                                                                pageToken=page_token).execute()
                messages.extend(response['messages'])

            self.currentEmailList.clear()
            #print(messages)]
            i = 0
            while i < 10 and i < len(messages):
                message = self.messages.get(userId='me', id=messages[i]['id'], format='metadata', metadataHeaders=["To", "From", "Subject", "References", "Message-ID"]).execute()
                self.currentEmailList.appendRow(GmailModule.EmailObject(sender=self.GetSender(message), snippet=self.GetSnippet(message), subject=self.GetSubjectTitle(message), threadid=self.GetThread(message), references=self.GetReferences(message), messageid=self.GetMessageID(message)))
                i += 1
            return 

        except apiclient.errors.HttpError as error:
            return

    def list_message_ids(self):
        emails = self.messages.list(userId='me').execute()
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

    @qtc.Slot(str,str,str,str, str,result=dict)
    def create_basic_email(self, sender, to, subject, message_text, ref=None, in_reply=None):
        message = MIMEText(message_text)
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject

        # Add referecena and in reply to headers if they are set
        if(not(ref is None) and not (in_reply is None)):
            message['References'] = ref
            message['In-Reply-To'] = in_reply
            #print(ref)
            #print(in_reply)
        return {'raw': base64.urlsafe_b64encode((message.as_bytes())).decode()}

    @qtc.Slot(str,str,str,str)
    def send_message(self, sender,to,subject,message_text):

        # Return early if there is no sender
        if(sender == "" or sender.isspace() or sender is None):
            return

        premessage = self.create_basic_email(sender,to,subject,message_text)
        try:
            message = (self.messages.send(userId=sender, body=premessage)
                       .execute())

            return message
        except apiclient.errors.HttpError as error:
            print('error')
            return

    def get_thread(self, query=''):
        response = self.service.users().threads().list(userId='me',q=query).execute()
        return

    def get_messages_from_a_thread(self,thread_id):
        try:
            response = self.service.users().threads().get(userId='me', id=thread_id).execute()
        except:
            return
        all_messages = response['messages']
        result = []
        for message in range(0, len(all_messages)):
            try:
                raw_msg = (all_messages[message]['payload']['parts'][0]['body']['data'])
            except KeyError:
                raw_msg = all_messages[message]['payload']['body']['data']
                msg_str = self.GetSender(all_messages[message]) + "\n" + base64.urlsafe_b64decode(raw_msg.encode('ASCII')).decode('utf-8')
                result.append(msg_str)
                return result
            msg_str = (base64.urlsafe_b64decode(raw_msg.encode())).decode('utf-8')
            result.append(msg_str)
        return result

    @qtc.Slot(int)
    def add_thread_messages(self, index):
        self.threadMessages.clear()
        threadid = self.currentEmailList.get(index, qtc.Qt.UserRole + 9)
        thread_messages = self.get_messages_from_a_thread(threadid)
        if(thread_messages is None):
            return
        for msg in thread_messages:
            self.threadMessages.appendRow(GmailModule.EmailObject(sender="", snippet="", subject="", threadid=threadid, message=msg))

    @qtc.Slot(str, str, str, str, str, str, str, result=str)
    def respond_to_thread(self, thread_id, sender, to, subject, msg_id, references, message_text):

        # Return early if there is no sender
        if(sender == "" or sender.isspace() or sender is None):
            return
        ref = references
        if(references != ""):
            ref = references + " " + msg_id
        else:
            ref = msg_id
        premessage = self.create_basic_email(sender, to, subject, message_text, ref=ref, in_reply=msg_id)
        premessage['threadId'] = thread_id
        print("References ", ref)
        print("Message ID ", msg_id)
        print("Subject ", subject)
        try:
            message = (self.messages.send(userId=sender, body=premessage)
                       .execute())

            return message
        except apiclient.errors.HttpError as error:
            print('error')
            return
        return
    
    @qtc.Slot(int, result=str)
    def get_current_threadid(self, index):
        return self.currentEmailList.get(index, qtc.Qt.UserRole + 9)
    
    @qtc.Slot(int, result=str)
    def get_current_sender(self, index):
        return self.currentEmailList.get(index, qtc.Qt.UserRole + 4)

    @qtc.Slot(int, result=str)
    def get_current_subject(self, index):
        return self.currentEmailList.get(index, qtc.Qt.UserRole + 6)
    
    @qtc.Slot(int, result=str)
    def get_current_message_id(self, index):
        return self.currentEmailList.get(index, qtc.Qt.UserRole + 10)
    
    @qtc.Slot(int, result=str)
    def get_current_references(self, index):
        return self.currentEmailList.get(index, qtc.Qt.UserRole + 11)

    class EmailObject(qtc.QObject):

        # Roles
        roles = {
            qtc.Qt.UserRole + 4: b'sender',
            qtc.Qt.UserRole + 5: b'receiver',
            qtc.Qt.UserRole + 6: b'subject',
            qtc.Qt.UserRole + 7: b'snippet',
            qtc.Qt.UserRole + 8: b'message',
            qtc.Qt.UserRole + 9: b'threadid',
            qtc.Qt.UserRole + 10: b'messageid',
            qtc.Qt.UserRole + 11: b'references'
        }

        # Signals
        senderChanged = qtc.Signal()
        receiverChanged = qtc.Signal()
        subjectChanged = qtc.Signal()
        snippetChanged = qtc.Signal()
        messageChanged = qtc.Signal()
        threadidChanged = qtc.Signal()

        # Initializer for email object
        def __init__(self, sender=None, receiver=None, subject=None, snippet=None, message=None, threadid=None, messageid=None, references=None):
            super(GmailModule.EmailObject, self).__init__()
            self._data = {b'sender': sender, b'receiver': receiver, b'subject': subject, b'snippet': snippet, b'message': message, b'threadid': threadid, b'messageid': messageid, b'references': references}
        
        # Retrieves the data
        def data(self, key):
            return self._data[self.roles[key]]
        
        @qtc.Property(str)
        def sender(self):
            self._data[b'sender']
        
        @qtc.Property(str)
        def receiver(self):
            return self._data[b'receiver']
        
        @qtc.Property(str)
        def subject(self):
            return self._data[b'subject']
        
        @qtc.Property(str)
        def snippet(self):
            return self._data[b'snippet']

        @qtc.Property(str)
        def message(self):
            return self._data[b'message']
        
        @qtc.Property(str)
        def threadid(self):
            return self._data[b'threadid']

        @qtc.Property(str)
        def messageid(self):
            return self._data[b'messageid']
        
        @qtc.Property(str)
        def references(self):
            return self._data[b'references']

        def __str__(self):
            return "[" + self.sender + " " + self.receiver + " " + self.subject + ']'
        
        def __repr__(self):
            return str(self)

if __name__ == '__main__':
    gmail = GmailModule()
    import timeit

    start = time.time()
    timer = 0
    '''
        for i in range(0,20):
        starts = time.time()
        gmail.send_message(sender='me', to='conradliste@gmail.com', subject=str(i), message_text=str(i)+'test')
        time.sleep(1)
        ends = time.time()
        timer += ends-starts
    end = time.time()
    print(end-start)
    gmail.get_messages_from_query('google')
    '''
    #gmail.get_thread('dsfdsf')
    gmail.get_messages_from_a_thread('173cb3d9a758c831')
    print(gmail.time_elapsed)
    print(timer)
