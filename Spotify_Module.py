import sys
import spotipy.util as util
import spellchecker
from os import environ

scope = 'user-library-read'


class SpotipyModule:

    def __init__(self, username, client_id, client_secret, redirect_url):
        # initalizes variables
        self.username = username
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_url = redirect_url
        self.scope = 'user-library-read user-read-playback-state streaming' \
                     ' user-modify-playback-state playlist-modify-private ' \
                     'user-read-playback-position user-read-currently-playing user-read-private'
        self.queue = QueueDataStructure()
        self.spellchecker = spellchecker.SpellChecker(language=u'en', distance=2)

        # generates access token for authorization to use spotify API

    def generate_token(self):
        token = util.prompt_for_user_token(self.username, self.scope, self.client_id, self.client_secret,
                                           self.redirect_url)
        # returns an authorized spotify object
        import spotipy as spy
        if token:
            sp = spy.Spotify(auth=token)
            return sp

        # plays music from a queue

    def play_music_from_queue(self):
        # generates lists to hold uri and time of each song
        queue_uri = [None] * self.queue.number_of_nodes
        queue_time = [None] * self.queue.number_of_nodes
        # reverses list order so that first in last out
        queue_info = (self.queue.reversal())
        # puts items into uri and time lists
        for index in range(len(queue_info)):
            queue_uri[index] = queue_info[index]['song_link']
            queue_time[index] = queue_info[index]['song_time']
        # plays songs from queue
        self.generate_token().start_playback(device_id=None, context_uri=None, uris=queue_uri,
                                             offset=None, position_ms=self.change_time(queue_time[0]))
        return

    # plays songs from a specific playlist
    def queue_music_from_playlist(self):
        # gets a list of tracks from a user inputted name of playlist
        search_results = ((self.generate_token().playlist_tracks(self.find_a_playlist(str(input("Enter Playlist: "))),
                                                                 fields='items,uri,name'))['items'])
        list_of_songs = [None] * len(search_results)
        index = 0
        # populates list with track ids
        for value in search_results:
            list_of_songs[index] = value['track']['uri']
            index += 1
        # starts playing playlist from beginning
        self.generate_token().start_playback(device_id=None, context_uri=None, uris=list_of_songs,
                                             offset=None, )
        return

        # pauses music
    def play_music(self):
        self.generate_token().start_playback()

    def pause_music(self):
        self.generate_token().pause_playback()

        return

        # changes volume of current song

    def change_volume(self, value):
        if self.generate_token().current_user_playing_track() is not None:
            self.generate_token().volume(value)

        return value

    # allows user to start playing in middle of song
    def change_time(self, song):
        sec = int(input("time to start in sec?"))

        if sec * 1000 > int(song):
            return
        else:
            return sec * 1000

        # skips current song to next in queue/playlist

    def next_song(self):
        self.generate_token().next_track()

        return

        # goes back 1 song

    def prev_song(self):
        self.generate_token().previous_track()

        return

        # allows user to search for a song and currently returns an array with all possible songs

    def add_song_to_queue(self, song_title):
        # search_query = song_title.replace(" ", "&20")
        # print(search_query)

        search_results = (self.generate_token().search(song_title, 20, 0, type='track,album'))
        print(type(search_results))
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

        # adds choosen song into the end of queue
        index = self.choose_song(song_choices_for_queue)
        song = search_results['tracks']['items'][index]['uri']
        title = search_results['tracks']['items'][index]['name']
        picture_image = search_results['tracks']['items'][index]['album']['images'][1]['url']
        artist = search_results['tracks']['items'][index]['artists'][0]['name']
        time = search_results['tracks']['items'][index]['duration_ms']
        print(time)
        song_info = {
            'image': picture_image,
            'artist': artist,
            'song_link': song,
            'song_title': title,
            'song_time': time
        }

        self.queue.push(song_info)
        # self.generate_token().add_to_queue(song)

        return

    def pop_song_from_queue(self):
        self.queue.pops()
        return
        # helper function for add_song_to_queue
        # allows user to choose which song to add into queue

    def choose_song(self, array_with_songs):
        for value in array_with_songs:
            print(value)
        index = int(input("Which do you want? Enter Number"))
        song_number = index - 1
        return song_number

    # hopefully returns song title
    def current_song(self):
        # Note I don't know if this one works/ Will test when Sahil isn't using Spotify
        return self.queue.get(self.queue.number_of_nodes - 1)
        # return self.generate_token().currently_playing()

    def find_a_playlist(self, playlist_title):

        # format_for_query =  playlist_title.replace(" ", "&20")
        # This needs to be tested
        search_results = self.generate_token().search(playlist_title, 20, 0, 'playlist')
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
        index = self.choose_song(playlist_to_be_queued)

        return playlist_ids[index]

    def current_queue(self):
        return self.queue

    # plays a playlist or 1+ tracks immediately
    def play_now(self, context_uris=None, uris=None):
        if context_uris is not None:
            self.generate_token().start_playback(device_id=None, context_uri=context_uris)
        if uris is not None:
            self.generate_token().start_playback(device_id=None, context_uri=None, uris=uris)

        return

    # functions to create
    '''
    get image info (done)
    get user info
    going to a certain time in song (done)
    implement genius api return 2 lines of lyrics 

    '''


class QueueDataStructure:

    def __init__(self):
        self.header_node = self.Node(None, None, None)
        self.number_of_nodes = 0
        self.pointer = self.header_node
        # O(1) operation
        # return the size of the doubly linked list
        # check by doing len(name_of_linked_list)

    def __len__(self):
        return self.number_of_nodes

        # returns the value at a given index
        # O(N) operation

    def get(self, index):
        # sets variable to beginning of linked list
        init_value = self.header_node
        # allows variable to traverse linked list
        for i in range(0, self.number_of_nodes):
            # gets the value at index of linked list
            if i == index:
                return init_value.get_value()
            init_value = init_value.get_next()
        return

    def push(self, new_data):

        new_node = self.Node(new_data, None, None)
        if self.number_of_nodes == 0:
            self.header_node.set_next(new_node)
            self.header_node.set_prev(new_node)
            new_node.set_next(self.header_node)
            new_node.set_prev(self.header_node)
            self.number_of_nodes += 1
        else:
            self.pointer = self.pointer.get_prev()
            new_node.set_next(self.pointer)
            new_node.set_prev(self.header_node)
            self.pointer.set_prev(new_node)
            self.header_node.set_next(new_node)
            self.number_of_nodes += 1

        return

    def pops(self):
        if self.number_of_nodes == 0:
            return
        self.header_node.get_prev().get_prev().set_next(self.header_node)
        self.header_node.set_prev(self.header_node.get_prev().get_prev())
        self.number_of_nodes -= 1
        return

    # add a value to the end of the linkedlist
    # O(1) operation
    def add(self, value):
        # adds node if only header is present
        if self.number_of_nodes == 0:
            node = self.Node(value, self.header_node, self.header_node)
            self.header_node.set_next(node)
            self.header_node.set_prev(node)
            self.number_of_nodes += 1
        # adds node when header and other nodes are present
        else:
            node = self.Node(value, self.header_node.get_prev(), self.header_node)
            self.header_node.get_prev().set_next(node)
            self.header_node.set_prev(node)
            self.number_of_nodes += 1

        return node.get_value()

        # add a value to the given index value
        # O(N) operation

    def insert(self, value, index):
        # checks to see if index is valid
        if index > self.number_of_nodes or index < 0:
            return "Index is out of range"
        # inserts when only header is present
        if self.number_of_nodes == 0:
            node_to_be_inserted = self.Node(value, self.header_node, self.header_node)
            self.header_node.set_next(node_to_be_inserted)
            self.header_node.set_prev(node_to_be_inserted)
            self.number_of_nodes += 1
            return
        # inserts when header and other nodes are present
        # variable traverses through the list
        node = self.header_node.get_next()
        for i in range(1, index + 1):
            node = node.get_next()
            # inserts the node at index
            if i == index:
                node_to_be_inserted = self.Node(value, node.get_prev(), node)
                node.get_prev().set_next(node_to_be_inserted)
                node.set_prev(node_to_be_inserted)
                self.number_of_nodes += 1

        return

        # remove the value at a given index
        # O(N) operation

    def remove(self, index):
        # initalizes variable to header node
        node = self.header_node
        # tells variable to traverse list until  it gets to index
        for i in range(0, self.number_of_nodes):
            node = node.get_next()

            if i == index:
                node.get_prev().set_next(node.get_next())
                node.get_next().set_prev(node.get_prev())

                self.number_of_nodes -= 1
                return

    def reversal(self):
        node_list = self.number_of_nodes * [None]
        node = self.header_node.get_prev()
        for i in range(0, self.number_of_nodes):
            node_list[i] = node.value
            node = node.get_prev()
        return node_list
        # prints the list in a [1,2,3] format
        # O(N) operation

    def __str__(self):
        # returns empty list if empty
        if self.header_node.get_next() == 'None':
            return '[]'
        # concatonates string to create "array"
        string = '['
        node = self.header_node.get_next()
        string += str(node.get_value())
        node = node.get_next()
        while node != self.header_node:
            string = string + ',' + str(node.get_value())
            node = node.get_next()
        string += ']'

        return string

    class Node:

        def __init__(self, value, prevnode, nextnode):
            self.value = value
            self.prevnode = prevnode
            self.nextnode = nextnode
            return

        def get_value(self):
            return self.value

        def set_prev(self, node):
            self.prevnode = node
            return

        def set_next(self, node):
            self.nextnode = node
            return

        def get_prev(self):
            return self.prevnode

        def get_next(self):
            return self.nextnode


print(environ.get('DEVICE_ID'))
Queue_object = QueueDataStructure()
spy = SpotipyModule(environ.get('USER'), environ.get('CLIENT_ID'), environ.get('CLIENT_SECRET'),
                    'https://Jarvis2_Spotify_Mod_HackAThon.com')
import time
spy.queue_music_from_playlist()
time.sleep(5)
spy.next_song()
time.sleep(5)
spy.pause_music()
time.sleep(5)
spy.play_music()
time.sleep(5)
spy.next_song()
spy.pause_music()
time.sleep(5)
