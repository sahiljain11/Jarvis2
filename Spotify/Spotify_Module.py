import sys
import spotipy.util as util
import spellchecker
import time
import spotipy as spy
from os import environ
from ListModel import ListModel
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2 import QtCore as qtc
from PySide2 import QtQuick as qtq
from PySide2 import QtQml as qtm

class SpotipyModule(qtc.QObject):

    #Signals
    currTitleChanged = qtc.Signal()
    currArtistChanged = qtc.Signal()
    currIconChanged = qtc.Signal()
    durTimeChanged = qtc.Signal()

    def __init__(self, username, client_id, client_secret, redirect_url,user):
        super(SpotipyModule, self).__init__()
        # initalizes variables
        self.username = username
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_url = redirect_url
        self.scope = 'user-library-read user-read-playback-state streaming' \
                     ' playlist-modify-public user-modify-playback-state playlist-modify-private ' \
                     'user-read-playback-position user-read-currently-playing user-read-private'
        self.user = user
        self.spellchecker = spellchecker.SpellChecker(language=u'en', distance=2)
        self.queue_id = None
        self.playing = False
        self.queue_uri = None
        self.queue_changed = False
        self.current_queue = None
        self.playlist_ids = None
        self.playlist_names = None
        self.search_results = None
        self.dur_time = 0
        self.artist = ""
        self.title = ""
        self.picture = qtc.QUrl()
        self.search_list = ListModel(SpotipyModule.SongWrapper)
        self.token = self.generate_token()
        self.queue = self.generate_queue()
        self.devices = self.token.devices()

    # generates access token for authorization to use spotify API
    def generate_token(self):
        token = util.prompt_for_user_token(self.username, self.scope, self.client_id, self.client_secret,
                                           self.redirect_url)
        #returns an authorized Spotify object if token is valid
        if token:
            sp = spy.Spotify(auth=token)
            return sp

    #generates a playlist that acts as a queue
    def generate_queue(self):
        playlists = self.token.user_playlists(self.user)
        #Checks Each Playlist to see if Queue exists
        for playlist in playlists['items']:
            #Sets Class Variables to Current Queue Variables if Queue exists
            if (playlist['name']) == 'Queue':
                self.queue_id = playlist['id']
                self.queue_uri = playlist['uri']
                self.current_queue = self.queue_uri
        #Makes Queue Playlist if not found
        if(self.queue_id is None):
            self.token.user_playlist_create(self.user,'Queue', public=True, description="Queue made by Yours Truly, Jarvis")
            playlists = self.token.user_playlists(self.user)
            self.queue_id = playlists['items'][0]['id']
            self.queue_uri = playlists['items'][0]['uri']
            self.current_queue = self.queue_uri
        return

    '''
    Adds a song to the queue playlist and Obtains title, picture, and artist for QML Display
    Args: str(spotify id of a song)
    '''
    @qtc.Slot(str)
    def add_song_to_queue(self, song_title):
        # search_query = song_title.replace(" ", "&20")
        # print(search_query)

        # Find all songs that show up for this query
        self.search_results = (self.token.search(song_title, 10, 0, type='track,album'))
        self.search_list.clear()

        # loop places song titles into list of possible songs
        for temp in self.search_results['tracks']['items']:
            title = temp['name']
            picture_image = temp['album']['images'][1]['url']
            artist = temp['artists'][0]['name']
            self.search_list.appendRow(SpotipyModule.SongWrapper(title, artist, picture_image, self.search_list))
        return

    '''
    Allows User to Choose a Song Based on Search Results from query.
    Args: Number of Chosen song
    '''
    @qtc.Slot(int)
    def helper_add_song_to_queue(self, number):

        # Change the queue to the default queue
        if self.current_queue != self.queue_uri:
            self.current_queue = self.queue_uri
            self.queue_changed = True

        self.search_list.clear()
        #Obtains specific song Information from Queried Song List based on Number
        temp = self.search_results['tracks']['items'][number]
        song = temp['uri']
        #Adds specifcied song into Playlist
        self.token.user_playlist_add_tracks(self.user, self.queue_id, tracks=[song])
        return

    # Plays Music from Queue Playlist or a Specified Playlist
    @qtc.Slot()
    def play_music(self):
        #Starts Playing Songs in Queue and Sets the current playing song info in Front end
        if(self.playing == False):
            print("Entered")
            self.token.start_playback(context_uri=self.current_queue)
            self.token.shuffle(False)
            time.sleep(.1)
            self.set_current_song_info()


        else:
            # Starts the playback of specified Playlist
            if self.queue_changed:
                self.token.start_playback(context_uri=self.current_queue)
                self.queue_changed = False
            else:
                self.token.start_playback()
            self.token.shuffle(False)

        self.playing = True
        return

    # Pauses music
    @qtc.Slot()
    def pause_music(self):
        self.token.pause_playback()
        return

    #Sets the Current time of the song, Artist, Picture, and Title of the song that is playing
    @qtc.Slot()
    def set_current_song_info(self):
        try:
            #Obtain Current Song Data
            temp = self.token.current_user_playing_track()['item']
            song_title = temp['name']

            #Checks if information is already set
            if(song_title == self.title):
                return
            self.set_durTime(temp['duration_ms'])
            self.set_currArtist(temp['artists'][0]['name'])
            self.set_currIcon(temp['album']['images'][0]['url'])
            self.set_currTitle(song_title)
            return
        # If Spotipy is not playing anything... ie self.token.set_current_song_info = None; return
        except TypeError:
            return

    #Gets the Progression of the Time of the Current Song
    @qtc.Slot(result=int)
    def get_current_time(self):
        #If song is not playing, Time is 0
        if(not self.playing):
            return 0
        songtime = self.token.current_user_playing_track()['progress_ms']
        time.sleep(.03)
        return songtime

    #Returns Title of Current Song
    @qtc.Property(str, notify=currTitleChanged)
    def currTitle(self):
        return self.title

    #Sends Signal when Current Title is Changed
    @currTitle.setter
    def set_currTitle(self, new_title):
        if self.title == new_title:
            return
        self.title = new_title
        self.currTitleChanged.emit()

    #Returns Album/Playlist Cover of Song
    @qtc.Property(qtc.QUrl, notify=currIconChanged)
    def currIcon(self):
        return self.picture

    #Sends Signal when Current Picture is Changed
    @currIcon.setter
    def set_currIcon(self, new_icon):
        if self.picture == new_icon:
            return
        self.picture = new_icon
        self.currIconChanged.emit()

    #Returns Artist of Current Song
    @qtc.Property(qtc.QUrl, notify=currArtistChanged)
    def currArtist(self):
        return self.artist

    #Sends Signal when Current Artist is Changed
    @currArtist.setter
    def set_currArtist(self, new_artist):
        if self.artist == new_artist:
            return
        self.artist = new_artist
        self.currArtistChanged.emit()

    #Returns Time of Progress of Song in seconds
    @qtc.Property(float, notify=durTimeChanged)
    def durTime(self):
        if self.dur_time == 0:
            return 10000
        return self.dur_time

    #Sends Signal when Progression of Song is changed
    @durTime.setter
    def set_durTime(self,time):
        if self.dur_time == time:
            return
        else:
            self.dur_time = time
            self.durTimeChanged.emit()

    # changes volume of current song
    @qtc.Slot(int)
    def change_volume(self, value):
        if self.token.current_user_playing_track() is not None:
            self.token.volume(value)

        return value

    # allows user to start playing in middle of song
    @qtc.Slot(int, result=int)
    def change_time(self, song_time):
        #If User sets a songtime within the time limit, time sets: else, dur_time remains the same
        if song_time >= 0 and song_time < self.dur_time:
            self.token.seek_track(position_ms=song_time)
            return song_time
        elif song_time >= self.dur_time:
            self.token.seek_track(position_ms=self.dur_time)
        else:
            return

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

    '''
    Finds a List of Playlist based on a Query
    Args: Playlist_title: A query for a playlist name
    Returns: A list of playlists that matches closely to query
    '''
    @qtc.Slot(str)
    def find_a_playlist(self, playlist_title):
        #Obtains list of 20 playlist id's
        search_results = self.token.search(playlist_title, 20, 0, 'playlist')
        if len(search_results['playlists']['items']) == 0:
            self.spellchecker.split_words(playlist_title)

        #initalizes list to hold data from each playlist
        playlist_number = 1
        playlist_index = 0
        playlist_to_be_queued = [None] * len(search_results['playlists']['items'])
        playlist_ids = [None] * len(search_results['playlists']['items'])

        # Clear the search list
        self.search_list.clear()
        # Add the playlists to the search list
        for songz in range(0, len(search_results['playlists']['items'])):
            name = str(playlist_number) + ') ' + search_results['playlists']['items'][songz]['name']
            author = search_results['playlists']['items'][songz]['owner']['display_name']
            playlist_ids[playlist_index] = search_results['playlists']['items'][songz]['uri']

            self.search_list.appendRow(SpotipyModule.SongWrapper(name, author, "", self.search_list))
            playlist_number += 1
            playlist_index += 1

        self.playlist_ids = playlist_ids
        self.playlist_names = playlist_to_be_queued
        return self.playlist_ids
    
    # plays songs from a specific playlist
    @qtc.Slot(int)
    def queue_music_from_playlist(self, index):
        
        # Clear the search list
        self.search_list.clear()

        playlist = self.playlist_ids[index]
        self.current_queue = playlist
        self.queue_changed = True
        self.play_music()
        return
        
    def destroyer(self):
        self.token.user_playlist_unfollow(user=self.user, playlist_id=self.queue_id)
        return

    #SongWrapper is a wrapper that wraps an object in a Qt object
    class SongWrapper(qtc.QObject):

        # Dictionary of roles for SongWrapper
        roles = {
            qtc.Qt.UserRole + 1: b'song',
            qtc.Qt.UserRole + 2: b'artist',
            qtc.Qt.UserRole + 3: b'image_url'
        }

        # Signals
        #songChanged = qtc.Signal()
        #artistChanged = qtc.Signal()
        #image_urlChanged = qtc.Signal()

        # Initialize the wrapper
        def __init__(self, song, artist, image_url, parent=None):
            super(SpotipyModule.SongWrapper, self).__init__()
            self._data = {b'song': song, b'artist': artist, b'image_url': image_url}
        
        # Retrieves the given role of the SongWrapper (i.e. song, artist, or image_url)
        def data(self, key):
            return self._data[self.roles[key]]

        @qtc.Property(str)
        def song(self):
            return self._data[b'song']
        
        @qtc.Property(str)
        def artist(self):
            return self._data[b'artist']
        
        @qtc.Property(qtc.QUrl)
        def image_url(self):
            return self._data[b'image_url']

        def __str__(self):
            return "[" + str(self.song) + ", " + str(self.artist)  + "]" 
        
        def __repr__(self):
            return str(self)

if __name__ == "__main__":
    spotify = SpotipyModule(environ.get('USER'), environ.get('CLIENT_ID'), environ.get('CLIENT_SECRET'),environ.get("REDIRECT_URI"), environ.get('USERNAME'))
    print(spotify.find_a_playlist('no u'))
    spotify.queue_music_from_playlist('spotify:playlist:6W1hJjo7L6zbqT0mNQUmFx')
    print(spotify.devices)
    #print('\n')
