from TweepyListener import Listener
import time, tweepy, sys
from t_auth import *
from utils import locationbox,Placenames

## user_authorization
#u_auth     = tweepy.auth.BasicAuthHandler(**twitter_account)
#api      = tweepy.API(u_auth)

# OAuth process, using the keys and tokens, for more info https://dev.twitter.com/docs/auth/oauth
auth = tweepy.OAuthHandler(consumer_key=twitter_auth2['consumer_key'], consumer_secret=twitter_auth2['consumer_secret'])
print auth.set_access_token
auth.set_access_token(key=twitter_auth2['access_token'], secret=twitter_auth2['access_token_secret'])

def StreamLocation(loc,verbose):
    
    if loc!='all' and loc in locationbox.keys():
        box = locationbox[loc]
    elif loc=='all':
        box=[]
        for citygrid in locationbox.values():
            box+=citygrid
    else:
        raise ValueError("Location not known ; How about these ones %s ?"%locationbox.keys())

    listen = Listener(arg=loc,verbose=verbose)
    stream = tweepy.Stream(auth, listen)

    print "Streaming started at %s location..." % (loc)

    
    stream.filter(locations = box)
    
def StreamTrack(loc,verbose):

    if loc in Placenames.keys():
        words = Placenames[loc]
    else:
        raise ValueError("Location not known ; How about these ones %s ?"%Placenames.keys())
    
    listen = Listener(arg=loc,prefix='placename',verbose=verbose)
    stream = tweepy.Stream(auth, listen)

    print "Streaming started using %s location from Placenames..." % (loc)

    
    stream.filter(track = words)

if __name__ == '__main__':
    
    if sys.argv[1].split('-')[0]=='Location':
        if sys.argv[1].split('-')[1] in locationbox.keys() or sys.argv[1].split('-')[1]=='all':
            if len(sys.argv)>2:
                verbose = True if '-v' in sys.argv[2:] else False
            else:
                verbose=False    
            StreamLocation(loc=sys.argv[1].split('-')[1],verbose=verbose)
        else:
            raise KeyError("Geocode for location not known, try one of these <all> or %s"%locationbox.keys())
    
    elif sys.argv[1].split('-')[0]=='Track':
        if sys.argv[1].split('-')[1] in Placenames.keys():
            if len(sys.argv)>2:
                verbose = True if '-v' in sys.argv[2:] else False
            else:
                verbose=False    
            StreamTrack(loc=sys.argv[1].split('-')[1],verbose=verbose)
        else:
            raise KeyError("location not found in track dictionary, try one of these %s"%Placenames.keys())
    else:
        print 'valid arguments are ex:python Streaming.py Location-Boston, python streaming.py Track-Boston'
