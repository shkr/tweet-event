from tweepy import StreamListener
import time, sys
import MySQLdb
from t_auth import *
import json

class Listener(StreamListener):

    def __init__(self, arg='',prefix = 'streamer'):
        
        if arg=='':
            raise ValueError("Please check Streaming argument, if it is empty")
        else:
            
            self.counter = 0
            self.prefix = prefix

            #ConnectionHook
            self.conn  =  MySQLdb.connect(**mysql_auth)
            
            #DataCursor
            self.curr   = self.conn.cursor()
            self.table = prefix+arg
            self.curr.execute("CREATE TABLE IF NOT EXISTS %sTable (Id INT PRIMARY KEY AUTO_INCREMENT,lat DECIMAL(7,5),lon DECIMAL(8,5),created_at VARCHAR(40),hashtags VARCHAR(200),urls VARCHAR(160),user_mentions VARCHAR(200),media VARCHAR(200),favorite_count INT,filter_level VARCHAR(10),tid BIGINT,in_reply_to_screen_name VARCHAR(20),in_reply_to_status_id BIGINT,in_reply_to_user_id BIGINT,retweet_count INT,source VARCHAR(200),text VARCHAR(160),user_id BIGINT,screen_name VARCHAR(100),user_location VARCHAR(40),retweeted_status_id BIGINT)"%self.table)
            
            #ErrorCursor
            self.errcur = self.conn.cursor()
            self.err_table = prefix+'ERROR_LOG'
            self.errcur.execute("CREATE TABLE IF NOT EXISTS ERROR_LOG (Id INT PRIMARY KEY AUTO_INCREMENT,status_id VARCHAR(50),type VARCHAR(50),time BIGINT)")

    def on_data(self, data):

        if  'in_reply_to_status' in data:
            self.on_status(data)
            return True
        elif 'delete' in data:
            delete = json.loads(data)['delete']['status']
            if self.on_delete(delete['id'], delete['user_id']) is False:
                return False
        elif 'limit' in data:
            if self.on_limit(json.loads(data)['limit']['track']) is False:
                return False
        elif 'warning' in data:
            warning = json.loads(data)['warnings']
            self.on_warning(warning['message'])
            return false

    def on_status(self, data):

        tweet = json.loads(data)

        if tweet['coordinates']!=None:
             lat =  tweet['coordinates']['coordinates'][1]
             lon =  tweet['coordinates']['coordinates'][0]
        else:
             lat = 0
             lon = 0
             
        created_at = tweet['created_at']
        hashtags = "%20".join([ item['text'] for item in tweet['entities']['hashtags']])
        urls = "%20".join([ item['url'] for item in tweet['entities']['urls']])
        user_mentions = "%20".join([ item['id_str']+"%40"+item["screen_name"] for item in tweet['entities']['user_mentions']])
        media = "%20".join([ item['id_str']+"%40"+item["media_url"] for item in tweet['entities']['media']]) if 'media' in tweet['entities'].keys() else ''
        favorite_count = tweet["favorite_count"] if tweet["favorite_count"]!=None else 0
        filter_level = tweet["filter_level"] if 'filter_level' in tweet.keys() else ''
        tid = tweet['id']
        in_reply_to_screen_name = tweet["in_reply_to_screen_name"] if tweet["in_reply_to_screen_name"]!=None else 0
        in_reply_to_status_id = tweet["in_reply_to_status_id"] if tweet["in_reply_to_status_id"]!=None else 0
        in_reply_to_user_id = tweet["in_reply_to_user_id"] if tweet["in_reply_to_user_id"]!=None else 0
        retweet_count = tweet["retweet_count"] if tweet["retweet_count"]!=None else 0
        source = tweet["source"].replace("'","\\'").replace('"','\\"')
        text = tweet["text"].replace("'","\\'").replace('"','\\"')
        user_id = tweet["user"]["id"]
        screen_name = tweet["user"]["screen_name"]
        user_location = tweet["user"]["location"]
        retweeted_status_id = tweet["retweeted_status"]["id"] if "retweeted_status" in tweet.keys() else 0
        query = """INSERT INTO %sTable(lat,lon,created_at,hashtags,urls,user_mentions,media,favorite_count,filter_level,tid,in_reply_to_screen_name,in_reply_to_status_id,in_reply_to_user_id,retweet_count,source,text,user_id,screen_name,user_location,retweeted_status_id) VALUES ("%d","%d","%s","%s","%s","%s","%s","%d","%s","%d","%s","%d","%d","%d","%s","%s","%d","%s","%s","%d")"""%(self.table,lat,lon,created_at,hashtags,urls,user_mentions,media,favorite_count,filter_level,tid,in_reply_to_screen_name,in_reply_to_status_id,in_reply_to_user_id,retweet_count,source,text,user_id,screen_name,user_location,retweeted_status_id)

        self.curr.execute(query)
        print query

        self.counter += 1    
        
        return

    def on_delete(self, status_id, user_id):
        self.errout.execute("INSERT INTO ERROR_LOG(status_id, type,time) VALUES('%s','ON_DELETE','%d')"%(str(status_id),int(time.time())))
        return

    def on_limit(self, track):
        self.errout.execute("INSERT INTO ERROR_LOG(status_id, type, time) VALUES('%s','ON_LIMIT','%d')"%(str(track),int(time.time())))
        return

    def on_error(self, status_code):
        self.errout.execute("INSERT INTO ERROR_LOG(status_id, type, time) VALUES('%s','ERROR_CODE','%d')"%(str(status_code),int(time.time())))
        return False

    def on_timeout(self):
        self.errout.execute("INSERT INTO ERROR_LOG(status_id, type, time) VALUES('SLEEP90','TIME_OUT','%d')"%int(time.time()))
        time.sleep(90)
        return 

    def on_warning(self,warning_msg):
        self.errout.execute("INSERT INTO ERROR_LOG(status_id, type, time) VALUES('%s','WARNING_MSG','%d')"%(warning_msg,int(time.time())))
        return