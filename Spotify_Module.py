import sys
import spotipy.util as util
import spellchecker
import spotipy as spy
from os import environ
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2 import QtCore as qtc
from PySide2 import QtQuick as qtq
from PySide2 import QtQml as qtm

scope = 'user-library-read'


class SpotipyModule(qtc.QObject):

    def __init__(self, username, client_id, client_secret, redirect_url):
        super(SpotipyModule, self).__init__()
        # initalizes variables
        self.username = username
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_url = redirect_url
        self.scope = 'user-library-read user-read-playback-state streaming' \
                     ' user-modify-playback-state playlist-modify-private ' \
                     'user-read-playback-position user-read-currently-playing user-read-private'

        self.spellchecker = spellchecker.SpellChecker(language=u'en', distance=2)
        self.token = self.generate_token()
        self.queue = self.generate_queue()
        self.playlist_ids = None
        self.playlist_names = None
        self.music_choices_for_queue = None
        self.search_results = None

    # generates access token for authorization to use spotify API
    def generate_token(self):
        token = util.prompt_for_user_token(self.username, self.scope, self.client_id, self.client_secret,
                                           self.redirect_url)
        # returns an authorized spotify object
        if token:
            sp = spy.Spotify(auth=token)
            return sp

    def generate_queue(self):
        self.token.user_playlist_create(self.username, 'Queue', public=False, description="Queue made by Yours Truly, Jarvis")
        playlists = self.token.user_playlists(self.username)
        for playlist in playlists['items']:
            print(playlist)
        return

    @qtc.Slot(str)
    # allows user to search for a song and currently returns an array with all possible songs
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
        self.token.user_playlist_add_tracks(self.client_id,)
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
        self.token.start_playback()
        return

    # pause music
    @qtc.Slot()
    def pause_music(self):
        self.token.pause_playback()
        return

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
        return

    @qtc.Slot()
    # goes back 1 song
    def prev_song(self):
        self.token.previous_track()
        return
    
    @qtc.Slot(str)
    # allows user to search for a song and currently returns an array with all possible songs
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

    def helper_add_song_to_queue(self,number):

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
        self.queue.push(song_info)
        return

    def pop_song_from_queue(self):
        self.queue.pops()
        return

    # hopefully returns song title
    @qtc.Property(str)
    def current_song(self):
        # Note I don't know if this one works/' Will test when Sahil isn't using Spotify
        return self.queue.get(self.queue.number_of_nodes-1)['song_title']
        # return self.generate_token().currently_playing()
    
    @qtc.Property(str)
    def current_artist(self):
        # Note I don't know if this one works/' Will test when Sahil isn't using Spotify
        return self.queue.get(self.queue.number_of_nodes-1)['artist']
        # return self.generate_token().currently_playing()
    
    @qtc.Property(qtc.QUrl)
    def current_image(self):
        # Note I don't know if this one works/' Will test when Sahil isn't using Spotify
        return self.queue.get(self.queue.number_of_nodes-1)['image']
        # return self.generate_token().currently_playing()
    

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



spotify = SpotipyModule(environ.get('USER'), environ.get('CLIENT_ID'), environ.get('CLIENT_SECRET'),environ.get("REDIRECT_URI"))





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


