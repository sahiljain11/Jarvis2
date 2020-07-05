import sys
import spotipy.util as util
import spellchecker
import time
import spotipy as spy
from os import environ
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2 import QtCore as qtc
from PySide2 import QtQuick as qtq
from PySide2 import QtQml as qtm

scope = 'user-library-read'


class SpotipyModule(qtc.QObject):

    #Signals
    currTitleChanged = qtc.Signal()
    currArtistChanged = qtc.Signal()
    currIconChanged = qtc.Signal()

    def __init__(self, username, client_id, client_secret, redirect_url):
        super(SpotipyModule, self).__init__()
        # initalizes variables
        self.username = username
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_url = redirect_url
        self.scope = 'user-library-read user-read-playback-state streaming' \
                     ' playlist-modify-public user-modify-playback-state playlist-modify-private ' \
                     'user-read-playback-position user-read-currently-playing user-read-private'  \


        self.spellchecker = spellchecker.SpellChecker(language=u'en', distance=2)
        self.queue_id = None
        self.playing = False
        self.queue_uri = None
        self.playlist_ids = None
        self.playlist_names = None
        self.music_choices_for_queue = None
        self.search_results = None
        self.artist = ""
        self.title = ""
        self.picture = qtc.QUrl()
        self.token = self.generate_token()
        self.queue = self.generate_queue()


    # generates access token for authorization to use spotify API
    def generate_token(self):
        token = util.prompt_for_user_token(self.username, self.scope, self.client_id, self.client_secret,
                                           self.redirect_url)
        # returns an authorized spotify object
        if token:
            sp = spy.Spotify(auth=token)
            return sp

    def generate_queue(self):
        playlists = self.token.user_playlists('USERNAME TOKEN')
        for playlist in playlists['items']:
            if (playlist['name']) == 'Queue':
                self.queue_id = playlist['id']
                self.queue_uri = playlist['uri']

        if(self.queue_id is None):
            self.token.user_playlist_create('USERNAME TOKEN','Queue', public=True, description="Queue made by Yours Truly, Jarvis")
            playlists = self.token.user_playlists('USERNAME TOKEN')
            self.queue_id = playlists['items'][0]['id']
            self.queue_uri = playlists['items'][0]['uri']
        return



    @qtc.Slot(str)
    def add_song_to_queue(self, song_title):
        # search_query = song_title.replace(" ", "&20")
        # print(search_query)

        search_results = (self.token.search(song_title, 20, 0, type='track,album'))

        # initalizes variables for loop
        song_number = 1
        song_index = 0
        song_choices_for_queue = [None] * len(search_results['tracks']['items'])
        # loop places song titles into array
        for songz in range(0, len(search_results['tracks']['items'])):
            song_choices_for_queue[song_index] = str(song_number) + ') ' + search_results['tracks']['items'][songz][
                'name']
            song_number += 1
            song_index += 1

        self.music_choices_for_queue = song_choices_for_queue
        self.search_results = search_results
        return

    @qtc.Slot(int)
    def helper_add_song_to_queue(self, number):

        song = self.search_results['tracks']['items'][number]['uri']
        title = self.search_results['tracks']['items'][number]['name']
        picture_image = self.search_results['tracks']['items'][number]['album']['images'][1]['url']
        artist = self.search_results['tracks']['items'][number]['artists'][0]['name']
        time = self.search_results['tracks']['items'][number]['duration_ms']
        song_info = {
            'image': picture_image,
            'artist': artist,
            'song_link': song,
            'song_title': title,
            'song_time': time
        }
        print(song)
        print(self.queue_id)
        self.token.user_playlist_add_tracks('USERNAME TOKEN', self.queue_id, tracks=[song])
        return



    # plays songs from a specific playlist
    @qtc.Slot()
    def queue_music_from_playlist(self,playlist_name):
        # gets a list of tracks from a user inputted name of playlist
        search_results = ((self.token.playlist_tracks(self.find_a_playlist(playlist_title=playlist_name),
                                                                 fields='items,uri,name'))['items'])
        list_of_songs = [None] * len(search_results)
        index = 0
        # populates list with track ids
        for value in search_results:
            list_of_songs[index] = value['track']['uri']
            index += 1
        # starts playing playlist from beginning
        self.token.start_playback(device_id=None, context_uri=None, uris=list_of_songs,
                                             offset=None, )
        return


    # play music
    @qtc.Slot()
    def play_music(self):
        if(self.playing == False):
            self.token.start_playback(context_uri=self.queue_uri)
            self.token.shuffle(False)
            time.sleep(.1)
            self.set_current_song_info()
        else:
            self.token.start_playback()
            self.token.shuffle(False)
        self.playing = True
        return

    # pause music
    @qtc.Slot()
    def pause_music(self):
        self.token.pause_playback()
        self.playing = False
        return

    @qtc.Slot()
    def set_current_song_info(self):
        #print(self.token.current_playback()['progress_ms'])
        #print(self.token.current_playback()['context'])
        #for key in self.token.current_user_playing_track()['item']:
        #print(self.token.current_user_playing_track()['item'][key])
        album_cover = ((self.token.current_user_playing_track()['item']['album']['images'][0]['url']))
        artist_name = ((self.token.current_user_playing_track()['item']['artists'][0]['name']))
        song_title = ((self.token.current_user_playing_track()['item']['name']))
        song_info = {
            'image': album_cover,
            'artist': artist_name,
            'song_title': song_title,
        }
        self.set_currTitle(song_info['song_title'])
        self.set_currIcon(song_info['image'])
        self.set_currArtist(song_info['artist'])
        
        return

    @qtc.Property(str, notify=currTitleChanged)
    def currTitle(self):
        return self.title

    @currTitle.setter
    def set_currTitle(self, new_title):
        if self.title == new_title:
            return
        self.title = new_title
        self.currTitleChanged.emit()
        
    @qtc.Property(qtc.QUrl, notify=currIconChanged)
    def currIcon(self):
        return self.picture
    
    @currIcon.setter
    def set_currIcon(self, new_icon):
        if self.picture == new_icon:
            return
        self.picture = new_icon
        self.currIconChanged.emit()

    @qtc.Property(qtc.QUrl, notify=currArtistChanged)
    def currArtist(self):
        return self.artist

    @currArtist.setter
    def set_currArtist(self, new_artist):
        if self.artist == new_artist:
            return
        self.artist = new_artist
        self.currArtistChanged.emit()

    # changes volume of current song
    @qtc.Slot(int)
    def change_volume(self, value):
        if self.token.current_user_playing_track() is not None:
            self.token.volume(value)

        return value

    # allows user to start playing in middle of song
    def change_time(self, song,time):

        if time * 1000 > int(song):
            return
        else:
            return time * 1000

    @qtc.Slot()
    # skips current song to next in queue/playlist
    def next_song(self):
        self.token.next_track()
        time.sleep(.1)
        self.set_current_song_info()
        return

    @qtc.Slot()
    # goes back 1 song
    def prev_song(self):
        self.token.previous_track()
        time.sleep(.1)
        self.set_current_song_info()
        return

    def pop_song_from_queue(self):
        self.queue.pops()
        return    

    def find_a_playlist(self, playlist_title):

        # format_for_query =  playlist_title.replace(" ", "&20")
        # This needs to be tested
        search_results = self.token.search(playlist_title, 20, 0, 'playlist')
        if len(search_results['playlists']['items']) == 0:
            self.spellchecker.split_words(playlist_title)

        playlist_number = 1
        playlist_index = 0
        playlist_to_be_queued = [None] * len(search_results['playlists']['items'])
        playlist_ids = [None] * len(search_results['playlists']['items'])

        for songz in range(0, len(search_results['playlists']['items'])):
            playlist_to_be_queued[playlist_index] = str(playlist_number) + ') ' + \
                                                    search_results['playlists']['items'][songz]['name']
            playlist_ids[playlist_index] = search_results['playlists']['items'][songz]['uri']
            playlist_number += 1
            playlist_index += 1
        self.playlist_ids = playlist_ids
        self.playlist_names = playlist_to_be_queued

        return

    def current_queue(self):
        return self.queue

    # plays a playlist or 1+ tracks immediately
    def play_now(self, context_uris=None, uris=None):
        if context_uris is not None:
            self.token.start_playback(device_id=None, context_uri=context_uris)
        if uris is not None:
            self.token.start_playback(device_id=None, context_uri=None, uris=uris)

        return


'''
spotify = SpotipyModule(environ.get('USER'), environ.get('CLIENT_ID'), environ.get('CLIENT_SECRET'),environ.get("REDIRECT_URI"))
spotify.add_song_to_queue('mask off')
spotify.helper_add_song_to_queue(0)
spotify.add_song_to_queue('Baby baby')
spotify.helper_add_song_to_queue(0)
spotify.add_song_to_queue('24 hours')
spotify.helper_add_song_to_queue(0)
spotify.add_song_to_queue('Just give me a reason')
spotify.helper_add_song_to_queue(0)
spotify.add_song_to_queue('tequila')
spotify.helper_add_song_to_queue(0)
time.sleep(5)
spotify.play_music()
print(spotify.artist)
print(spotify.title)
print(spotify.picture)
time.sleep(5)
spotify.next_song()
print(spotify.artist)
print(spotify.title)
print(spotify.picture)
time.sleep(5)
spotify.next_song()
print(spotify.artist)
print(spotify.title)
print(spotify.picture)
time.sleep(5)
spotify.next_song()
print(spotify.artist)
print(spotify.title)
print(spotify.picture)
time.sleep(5)
'''

'''
spotify = SpotipyModule(environ.get('USER'), environ.get('CLIENT_ID'), environ.get('CLIENT_SECRET'),environ.get("REDIRECT_URI"))
spotify.add_song_to_queue('beautiful people')
spotify.helper_add_song_to_queue(0)
print('\n')

spotify.set_current_song_info()
'''
'''
Clean up CODE
-Implement anything that requires terminal input as parameters of a function (e.g. querying for a song, going back and forth within a song)
-Implement a function to return an array of the possible song choices after querying it
-Store current song as a class variable (self.current) and just make the function current_song() return self.current
-Store volume as a class variable 
-Account for skipping to next song and skipping to previous song when there is no next or previous song 
-Implement pop after song is done
-Cram play_music_from_queue and play_music_from_playlist into play_music
'''


