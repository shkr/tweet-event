from t_auth import *
import datetime
from TwitterSearch.TwitterSearch import *
import MySQLdb
from urllib import quote

location = {'Delhi':dict(latitude=28.6353080,longitude= 77.2249600,radius=25),'Mumbai':dict(latitude=19.0759837,longitude=72.8776559,radius=25),'NYC':dict(latitude=40.7143528,longitude=-74.00597309999999,radius=25),
            'Boston':dict(latitude=42.3988669,longitude=-70.9232009999999,radius=25),'San Francisco':dict(latitude=37.7749295,longitude=-122.4194155,radius=25),'London':dict(latitude=51.6723432,longitude=-0.1998244,radius=25),'Los Angeles':dict(latitude=34.337306,longitude=-118.155289,radius=25),'Toronto':dict(latitude=43.5810847000001,longitude=-79.639219,radius=25),'San Diego':dict(latitude=32.7153292,longitude=-117.1572551,radius=25),'Houston':dict(latitude=29.7601927,longitude=-95.36938959999999,radius=25)}

cities = location.keys()

for item in cities:
     try:
          tso = TwitterSearchOrder()
          tso.setCount(100)
          tso.setIncludeEntities(True)
          tso.setResultType('mixed')
          #tso.setUntil(datetime.datetime.strptime('31-12-13','%d-%m-%y').date())
          tso.setGeocode(**location[item])
          ts = TwitterSearch(**twitter_auth)
          
          conn = MySQLdb.connect(**mysql_auth)
          curr = conn.cursor()
          with conn:
               curr.execute("DROP TABLE IF EXISTS TestTable")
               curr.execute("CREATE TABLE TestTable (Id INT PRIMARY KEY AUTO_INCREMENT,lat DECIMAL(7,5),lon DECIMAL(8,5),created_at VARCHAR(40),hashtags VARCHAR(200),urls VARCHAR(160),user_mentions VARCHAR(200),media VARCHAR(200),favorite_count INT,filter_level VARCHAR(10),tid BIGINT,in_reply_to_screen_name VARCHAR(20),in_reply_to_status_id BIGINT,in_reply_to_user_id BIGINT,retweet_count INT,source VARCHAR(200),text VARCHAR(160),user_id BIGINT,screen_name VARCHAR(100),user_location VARCHAR(40),retweeted_status_id BIGINT)")
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

                    #print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) 
                    query = """INSERT INTO TestTable(lat,lon,created_at,hashtags,urls,user_mentions,media,favorite_count,filter_level,tid,in_reply_to_screen_name,in_reply_to_status_id,in_reply_to_user_id,retweet_count,source,text,user_id,screen_name,user_location,retweeted_status_id) VALUES ("%d","%d","%s","%s","%s","%s","%s","%d","%s","%d","%s","%d","%d","%d","%s","%s","%d","%s","%s","%d")"""%(lat,lon,created_at,hashtags,urls,user_mentions,media,favorite_count,filter_level,tid,in_reply_to_screen_name,in_reply_to_status_id,in_reply_to_user_id,retweet_count,source,text,user_id,screen_name,user_location,retweeted_status_id)
                    print query
                    curr.execute(query)
                    if ct==20:
                         break
              
     except TwitterSearchException as e:
          print (e)
         
         
         
         
