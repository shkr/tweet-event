import sys
from t_auth import *
import datetime
from TwitterSearch.TwitterSearch import *
import MySQLdb
from urllib import quote
from geocode import location


def collect_tweets_from_city(arg):
     
     if arg in location.keys():
         city = arg
     else:
         raise KeyError("[WARNING CASE-SENSITIVE] %s location geocode are not known, available locations %s"%(arg,location.keys())) 


     try:
          tso = TwitterSearchOrder()
          tso.setCount(100)
          tso.setIncludeEntities(True)
          tso.setResultType('mixed')
          tso.setGeocode(**location[city])
          ts = TwitterSearch(**twitter_auth1)
          
          conn = MySQLdb.connect(**mysql_auth)
          curr = conn.cursor()
          with conn:
               curr.execute("CREATE TABLE IF NOT EXISTS %sTable (Id INT PRIMARY KEY AUTO_INCREMENT,lat DECIMAL(7,5),lon DECIMAL(8,5),created_at VARCHAR(40),hashtags VARCHAR(200),urls VARCHAR(160),user_mentions VARCHAR(200),media VARCHAR(200),favorite_count INT,filter_level VARCHAR(10),tid BIGINT,in_reply_to_screen_name VARCHAR(20),in_reply_to_status_id BIGINT,in_reply_to_user_id BIGINT,retweet_count INT,source VARCHAR(200),text VARCHAR(160),user_id BIGINT,screen_name VARCHAR(100),user_location VARCHAR(40),retweeted_status_id BIGINT)"%city)
               ct=1
               for tweet in ts.searchTweetsIterable(tso):
                    ct+=1
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
                    query = """INSERT INTO %sTable(lat,lon,created_at,hashtags,urls,user_mentions,media,favorite_count,filter_level,tid,in_reply_to_screen_name,in_reply_to_status_id,in_reply_to_user_id,retweet_count,source,text,user_id,screen_name,user_location,retweeted_status_id) VALUES ("%d","%d","%s","%s","%s","%s","%s","%d","%s","%d","%s","%d","%d","%d","%s","%s","%d","%s","%s","%d")"""%(city,lat,lon,created_at,hashtags,urls,user_mentions,media,favorite_count,filter_level,tid,in_reply_to_screen_name,in_reply_to_status_id,in_reply_to_user_id,retweet_count,source,text,user_id,screen_name,user_location,retweeted_status_id)

                    curr.execute(query)
                    print query
                    if ct==20:
                         break
              
     except TwitterSearchException as e:
          print (e)
         
if __name__=='__main__':
     city = sys.argv[1]
     collect_tweets_from_city(city)

     
         
         
