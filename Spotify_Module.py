import spotipy as spy
import sys
import spotipy.util as util

from os import environ



scope = 'user-library-read'
#checks for number of arguments for script

if len(sys.argv) > 1:
    username = sys.argv[1]
    print(sys.argv[0])
    print(username)
    print(sys.argv[2])
    print(sys.argv[3])

else:
    print("Usage: %s username" % (sys.argv[0],))


    print("len is 1")
    sys.exit()
token = util.prompt_for_user_token(username,scope, environ.get('CLIENT_ID'),environ.get('CLIENT_SECRET'),environ.get('REDIRECT_URL'))

if token:
    sp = spy.Spotify(auth=token)
    results = sp.current_user_saved_tracks()
    for item in results['items']:
        track = item['track']
        print(track['name'] + ' - ' + track['artists'][0]['name'])
    sp.start_playback()
else:
    print("Can't get token for", username)



'''
util.prompt_for_user_token()
spotify = spy.Spotify()

print(spotify)
'''

