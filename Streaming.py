from TweepyListener import Listener
import time, tweepy, sys
from t_auth import *
from geocode import locationbox

## user_authorization
#u_auth     = tweepy.auth.BasicAuthHandler(**twitter_account)
#api      = tweepy.API(u_auth)

# OAuth process, using the keys and tokens, for more info https://dev.twitter.com/docs/auth/oauth
auth = tweepy.OAuthHandler(consumer_key=twitter_auth2['consumer_key'], consumer_secret=twitter_auth2['consumer_secret'])
print auth.set_access_token
auth.set_access_token(key=twitter_auth2['access_token'], secret=twitter_auth2['access_token_secret'])

def StreamLocation(loc):
    
    if loc!='all':
        box = locationbox[loc]
    elif loc:
        box=[]
        for citygrid in locationbox.values():
            box+=citygrid

    listen = Listener(arg=loc)
    stream = tweepy.Stream(auth, listen)

    print "Streaming started at %s location..." % (loc)

    
    stream.filter(locations = box)
    

if __name__ == '__main__':
    if sys.argv[1] in locationbox.keys() or sys.argv[1]=='all':
        StreamLocation(sys.argv[1])
    else:
        raise KeyError("Geocode for location not known, try one of these <all> or %s"%locationbox.keys())